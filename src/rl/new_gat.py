import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.data import Data
from typing import Set

from utils.configs import CONFIGS


# Feature masking for partial observability
# from environment.env.get_node_features():
# [0:4]   node type (one-hot) - structural, always visible
# [4]     fire intensity - hazard info
# [5]     smoke density - hazard info  
# [6]     length normalized - structural, always visible
# [7]     people count - occupant info, hide for unseen nodes
# [8]     average HP - occupant info, hide for unseen nodes
# [9]     agent presence - occupant info, hide for unseen nodes
# [10]    distance to fire - global hazard, always visible

OCCUPANT_FEATURE_INDICES = torch.tensor([7, 8, 9], dtype=torch.long)
HAZARD_FEATURE_INDICES = torch.tensor([4, 5], dtype=torch.long)


# POMDP masking - what the actor can and can't see
def build_actor_data(global_data: Data, seen_nodes: Set[int]) -> Data:
    """
    Build partial observation for the actor (policy network).
    
    The actor only sees occupant info (people, HP, agent presence) for nodes
    it's actually visited. Everything else gets masked to zero.
    This is how we implement partial observability - you can't know about
    people in rooms you haven't searched yet.
    
    Args:
        global_data: full state graph from environment
        seen_nodes: which node indices the agent has visited
        
    Returns:
        masked graph with occupant features = 0 for unseen nodes
    """
    x = global_data.x.clone()  # clone to avoid messing up the original
    N = x.size(0)
    device = x.device
    
    # mask occupant features for unseen nodes
    # unseen nodes: zero out people_count, avg_hp, agent_presence
    # but keep structural info (type, length) and hazard info visible
    for node_idx in range(N):
        if node_idx not in seen_nodes:
            # haven't seen this node yet - mask occupant features [7, 8, 9]
            x[node_idx, 7] = 0.0  # people_count
            x[node_idx, 8] = 0.0  # avg_hp
            x[node_idx, 9] = 0.0  # agent_presence
    
    # return masked data (same edges, just different node features)
    return Data(
        x=x,
        edge_index=global_data.edge_index,
        edge_attr=getattr(global_data, 'edge_attr', None)
    )


def build_critic_data(global_data: Data) -> Data:
    """
    Build full observation for the critic (value network).
    
    The critic gets to see everything - the complete ground-truth state.
    This is "privileged information" that helps it estimate values better,
    but we don't leak it into the policy.
    
    Args:
        global_data: full state graph from environment
        
    Returns:
        unmasked full state graph
    """
    # critic sees full state, no masking
    return global_data


class GAT(nn.Module):
    """
    Graph Attention Network for building evacuation.
    
    Takes the building graph and outputs embeddings for each node.
    Main fixes from original:
    - accepts PyG Data objects directly (not separate tensors)
    - fixed GraphNorm dimensions to match layer outputs
    - fixed the transpose bug where result wasn't stored
    - processes single graphs (not batches) to match what env gives us
    - returns full node embeddings [N, hidden_dim] for each agent to use
    """
    def __init__(self) -> None:
        super().__init__()
        
        in_dim = 11  # matches FEATURE_DIM from environment
        
        # define GAT layers with consistent dimensions
        # optimized for RTX 5090 - using larger hidden dims
        self.hidden_dim1 = 64  # bumped up from 32
        self.hidden_dim2 = 96  # bumped up from 48
        self.hidden_dim3 = 48  # bumped up from 24 for richer representations
        
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
            heads=1,  # single head for final layer (standard practice)
            dropout=CONFIGS['gat']['dropout'],
            concat=False,
        )
        
        # fixed GraphNorm dimensions to match layer outputs
        # original had wrong dims (10 and 32) - should match actual layer output
        self.norm1 = GraphNorm(self.hidden_dim1 * CONFIGS['gat']['heads'])  # after gat1
        self.norm2 = GraphNorm(self.hidden_dim2 * CONFIGS['gat']['heads'])  # after gat2
        # no norm after gat3 (common practice for final layer)
        
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
        
        Now accepts PyG Data object directly (matches what env.get_observation() returns).
        
        Args:
            data: PyG Data object with:
                - x: node features [N, 11]
                - edge_index: edge connectivity [2, E]
                - (optional) edge_attr: edge features
            
        Returns:
            node embeddings for all nodes [N, 48]
        
        Example:
            obs = env.get_observation()  # returns Data object
            node_embeddings = gat(obs)   # [N, 48]
            agent_idx = env.get_agent_node_index(0)
            agent_embedding = node_embeddings[agent_idx]  # [48]
        """
        # extract x and edge_index from Data object
        # original code expected separate tensors, now we unpack from Data
        x = data.x              # [N, 11] node features
        edge_index = data.edge_index  # [2, E] edge connectivity
        
        # PyG GATConv expects edge_index in [2, E] format (already correct from env)
        # original had a bug where it did torch.transpose but didn't store the result
        
        # layer 1: 11 → 32 * heads
        out = self.gat1(x, edge_index)
        out = self.norm1(out)
        out = self.elu(out)
        
        # layer 2: 32 * heads → 48 * heads
        out = self.gat2(out, edge_index)
        out = self.norm2(out)
        out = self.elu(out)
        
        # layer 3: 48 * heads → 24
        out = self.gat3(out, edge_index)
        # no norm or activation on final layer (standard practice)
        
        # return full node embeddings so Policy can extract agent-specific ones
        return out  # shape: [N, 48] where N = number of nodes in building
    
    
    def get_global_embedding(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute global building state by pooling over all nodes.
        
        Gives agents a sense of overall building state.
        
        Args:
            node_embeddings: node embeddings from forward() [N, H]
            
        Returns:
            mean-pooled global building state [H]
        """
        return torch.mean(node_embeddings, dim=0)  # [H]
    
    
    @staticmethod
    def process_batch_with_pomdp(
        global_data: Data,
        agent_seen_nodes_list: list,
    ) -> tuple:
        """
        Process a batch of agents with POMDP masking.
        
        Implements:
        - decentralized actor: partial observation (masked occupants)
        - centralized critic: full state (privileged information)
        
        Args:
            global_data: full environment state graph
            agent_seen_nodes_list: list of seen_nodes sets, one per agent
                                   seen_nodes[j] = Set[int] of nodes agent j visited
            
        Returns:
            tuple of (actor_data_list, critic_data):
                actor_data_list: partial observation graphs for each agent
                critic_data: full state graph (same for all agents)
        """
        actor_data_list = []
        for seen_nodes in agent_seen_nodes_list:
            # build masked actor data for each agent
            actor_data = build_actor_data(global_data, seen_nodes)
            actor_data_list.append(actor_data)
        
        # build full critic data (same for all agents)
        critic_data = build_critic_data(global_data)
        
        return actor_data_list, critic_data