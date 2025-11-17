import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.data import Data
from typing import Set

from utils.configs import CONFIGS


# ===== POMDP FEATURE MASKING INDICES =====
# From environment.env.get_node_features():
# [0:4]   One-hot node type (structural, always visible)
# [4]     Fire intensity (hazard)
# [5]     Smoke density (hazard)
# [6]     Length normalized (structural, always visible)
# [7]     People count (occupant - should be masked for unseen nodes)
# [8]     Average HP (occupant - should be masked for unseen nodes)
# [9]     Agent presence (occupant - should be masked for unseen nodes)
# [10]    Distance to fire (global hazard, always visible)

OCCUPANT_FEATURE_INDICES = torch.tensor([7, 8, 9], dtype=torch.long)
HAZARD_FEATURE_INDICES = torch.tensor([4, 5], dtype=torch.long)


# ===== POMDP DATA MASKING FUNCTIONS =====
def build_actor_data(global_data: Data, seen_nodes: Set[int]) -> Data:
    """
    Build partial observation data for actor (policy network).
    
    The actor receives a partial observation graph where occupant features
    on unseen nodes are masked to zero. This implements partial observability:
    agents can only know about people/agents in nodes they've visited.
    
    Args:
        global_data (Data): Full state graph from environment
        seen_nodes (Set[int]): Node indices the agent has seen (visited or searched)
        
    Returns:
        Data: Masked graph with occupant features zeroed for unseen nodes
    """
    x = global_data.x.clone()  # Clone to avoid modifying original
    N = x.size(0)
    device = x.device
    
    # ===== CHANGE: Mask occupant features for unseen nodes =====
    # Unseen nodes: people_count, avg_hp, agent_presence → 0
    # Keeps structural (type, length) and hazard info visible globally
    for node_idx in range(N):
        if node_idx not in seen_nodes:
            # This node is unseen: mask occupant features [7, 8, 9]
            x[node_idx, 7] = 0.0  # people_count
            x[node_idx, 8] = 0.0  # avg_hp
            x[node_idx, 9] = 0.0  # agent_presence
    
    # Return masked data with same edge structure
    return Data(
        x=x,
        edge_index=global_data.edge_index,
        edge_attr=getattr(global_data, 'edge_attr', None)
    )


def build_critic_data(global_data: Data) -> Data:
    """
    Build full observation data for critic (value network).
    
    The critic receives the complete ground-truth state (privileged information)
    to improve value estimation without leaking it into the policy.
    
    Args:
        global_data (Data): Full state graph from environment
        
    Returns:
        Data: Unmasked full state graph
    """
    # ===== CHANGE: Critic sees full state (no masking) =====
    return global_data


class GAT(nn.Module):
    """
    Graph Attention Network for processing building evacuation environment.
    
    ===== MAJOR CHANGES FROM ORIGINAL =====
    1. Fixed input signature to accept PyG Data objects (not separate tensors)
    2. Fixed GraphNorm dimensions to match actual layer outputs
    3. Fixed missing assignment of torch.transpose result
    4. Changed to process single graphs (not batched) to match env output
    5. Added proper handling of edge_index format
    6. Output now returns full node embeddings [N, H] for agent-centric readout
    """
    def __init__(self) -> None:
        super().__init__()
        
        in_dim = 11  # Match FEATURE_DIM from environment
        
        # Define the GAT layers with consistent dimensions
        # ===== CHANGE 2: Defined explicit hidden dimensions for clarity =====
        # Optimized for RTX 5090 - larger hidden dimensions
        self.hidden_dim1 = 64  # Increased from 32
        self.hidden_dim2 = 96  # Increased from 48
        self.hidden_dim3 = 48  # Increased from 24 for richer representations
        
        self.gat1 = GATConv(
            in_channels=in_dim,              # 11 from environment
            out_channels=self.hidden_dim1,    # 32
            heads=CONFIGS['gat']['heads'],
            dropout=CONFIGS['gat']['dropout'],
            concat=True,  # Concatenate attention heads
        )
        self.gat2 = GATConv(
            in_channels=self.hidden_dim1 * CONFIGS['gat']['heads'],  # 32 * heads
            out_channels=self.hidden_dim2,    # 48
            heads=CONFIGS['gat']['heads'],
            dropout=CONFIGS['gat']['dropout'],
            concat=True,
        )
        self.gat3 = GATConv(
            in_channels=self.hidden_dim2 * CONFIGS['gat']['heads'],  # 48 * heads
            out_channels=self.hidden_dim3,    # 24
            heads=1,  # Single head for final layer (standard practice)
            dropout=CONFIGS['gat']['dropout'],
            concat=False,
        )
        
        # ===== CHANGE 3: Fixed GraphNorm dimensions to match layer outputs =====
        # Original had GraphNorm(10) and GraphNorm(32) which were wrong
        # GraphNorm should match the output dimension of each layer
        self.norm1 = GraphNorm(self.hidden_dim1 * CONFIGS['gat']['heads'])  # After gat1
        self.norm2 = GraphNorm(self.hidden_dim2 * CONFIGS['gat']['heads'])  # After gat2
        # No norm after gat3 (common practice for final layer)
        
        # Define helper layers
        self.elu = nn.ELU(CONFIGS['gat']['elu_parameter'])  # Adds nonlinearity
        
    
    def __call__(
        self,
        data: Data,
    ) -> torch.Tensor:
        """Override call for IDE tools."""
        return super().__call__(data)
        
        
    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        """
        Pass building graph through GAT to get node embeddings.
        
        ===== CHANGE 4: Now accepts PyG Data object (not separate tensors) =====
        This matches what env.get_observation() returns.
        
        Args:
            data (Data): PyG Data object containing:
                - x: Node features [N, 11]
                - edge_index: Edge connectivity [2, E]
                - (optional) edge_attr: Edge features
            
        Returns:
            out (Tensor[N, 48]): Node embeddings for all nodes in the building
                                 Shape is [N, hidden_dim3] where N = number of nodes
        
        Example usage:
            obs = env.get_observation()  # Returns Data object
            node_embeddings = gat(obs)   # [N, 48]
            agent_idx = env.get_agent_node_index(0)
            agent_embedding = node_embeddings[agent_idx]  # [48]
        """
        # ===== CHANGE 5: Extract x and edge_index from Data object =====
        # Original code expected separate tensors, now we unpack from Data
        x = data.x              # [N, 11] node features
        edge_index = data.edge_index  # [2, E] edge connectivity
        
        # PyG GATConv expects edge_index in [2, E] format (already correct from env)
        # Original code did torch.transpose but didn't store result - that was a bug
        
        # ===== Layer 1: 11 → 32 * heads =====
        out = self.gat1(x, edge_index)
        out = self.norm1(out)
        out = self.elu(out)
        
        # ===== Layer 2: 32 * heads → 48 * heads =====
        out = self.gat2(out, edge_index)
        out = self.norm2(out)
        out = self.elu(out)
        
        # ===== Layer 3: 48 * heads → 24 =====
        out = self.gat3(out, edge_index)
        # No normalization or activation on final layer (standard practice)
        
        # ===== CHANGE 6: Return full node embeddings [N, 48] =====
        # Original code didn't return anything clearly
        # Now we return embeddings for ALL nodes so Policy can extract agent-specific ones
        return out  # Shape: [N, 48] where N is number of nodes in building
    
    
    def get_global_embedding(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute global building state by pooling over all nodes.
        
        ===== NEW METHOD: Added for global context =====
        Useful for giving agents a sense of overall building state.
        
        Args:
            node_embeddings (Tensor[N, H]): Node embeddings from forward()
            
        Returns:
            global_embedding (Tensor[H]): Mean-pooled global building state
        """
        return torch.mean(node_embeddings, dim=0)  # [H]
    
    
    @staticmethod
    def process_batch_with_pomdp(
        global_data: Data,
        agent_seen_nodes_list: list,
    ) -> tuple:
        """
        Process a batch of agents with POMDP masking for actor and full state for critic.
        
        ===== NEW METHOD: POMDP-aware batch processing =====
        Implements:
        - Decentralized actor: partial observation (masked occupants)
        - Centralized critic: full state (privileged information)
        
        Args:
            global_data (Data): Full environment state graph
            agent_seen_nodes_list (list): List of seen_nodes sets, one per agent
                                          seen_nodes[j] = Set[int] of node indices agent j has visited
            
        Returns:
            tuple:
                actor_data_list (list[Data]): Partial observation graphs for each agent
                critic_data (Data): Full state graph (same for all agents)
        """
        actor_data_list = []
        for seen_nodes in agent_seen_nodes_list:
            # ===== CHANGE: Build masked actor data for each agent =====
            actor_data = build_actor_data(global_data, seen_nodes)
            actor_data_list.append(actor_data)
        
        # ===== CHANGE: Build full critic data (same for all agents) =====
        critic_data = build_critic_data(global_data)
        
        return actor_data_list, critic_data