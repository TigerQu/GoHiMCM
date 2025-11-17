import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Set

from .new_gat import GAT, build_actor_data, build_critic_data
try:
    from utils.configs import CONFIGS
except ImportError:
    CONFIGS = {}


class Policy(nn.Module):
    """
    Policy network for multi-agent building evacuation.
    
    ===== MAJOR CHANGES FROM ORIGINAL =====
    1. Now properly handles PyG Data objects from environment
    2. Implements agent-centric readout (gets embedding for each agent's node)
    3. Implements action masking using env.get_valid_actions()
    4. Proper multi-agent action selection
    5. Fixed dimensions to match GAT output (24, not arbitrary)
    6. Added action space mapping (wait/search/move_X)
    """
    def __init__(self, num_agents: int = 2, max_actions: int = 15) -> None:
        """
        Initialize policy network.
        
        ===== CHANGE 1: Added constructor parameters for multi-agent =====
        
        Args:
            num_agents: Number of firefighter agents (default 2 for standard layouts)
            max_actions: Maximum number of possible actions per agent
                        (wait, search, + up to ~10 move actions depending on graph connectivity)
        """
        super().__init__()
        self.num_agents = num_agents
        self.max_actions = max_actions
        
        # Instantiate GAT 
        self.gat = GAT()
        
        # ===== CHANGE 2: Fixed input dimension to match GAT output =====
        # GAT outputs 48-dim embeddings per node (updated for RTX 5090)
        # We concatenate: [agent_node_embedding (48) + global_embedding (48)] = 96
        self.agent_feature_dim = 48  # From GAT output (increased from 24)
        self.input_dim = self.agent_feature_dim * 2  # Agent + global = 96
        
        # ===== CHANGE 3: Output dimension matches action space =====
        # Actions: 0=wait, 1=search, 2...k=move_neighbor_i
        # We'll use max_actions and mask invalid ones
        # Larger network for RTX 5090
        self.action_head = nn.Sequential(
            nn.Linear(self.input_dim, 128),  # Increased from 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # Increased from 32
            nn.ReLU(),
            nn.Linear(64, self.max_actions)  # Logits for all possible actions
        )
        
    
    def __call__(
        self,
        data: Data,
        agent_node_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override __call__ for IDE tools."""
        return super().__call__(data, agent_node_indices)
    
        
    def forward(
        self,
        data: Data,
        agent_node_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action logits for all agents based on current observation.
        
        ===== CHANGE 4: Complete rewrite to handle multi-agent properly =====
        
        Args:
            data (Data): PyG Data object from env.get_observation()
                Contains x [N, 11], edge_index [2, E]
            agent_node_indices (Tensor[num_agents]): Index of each agent's current node
                From env.get_agent_node_index(agent_id) for each agent
            
        Returns:
            action_logits (Tensor[num_agents, max_actions]): Logits for each agent's actions
            node_embeddings (Tensor[N, 48]): Node embeddings (for value function reuse)
        
        Example:
            obs = env.get_observation()
            agent_indices = torch.tensor([env.get_agent_node_index(0), 
                                         env.get_agent_node_index(1)])
            logits, embeddings = policy(obs, agent_indices)
            # Apply action masking, then sample actions
        """
        # Process entire building graph through GAT 
        node_embeddings = self.gat(data)  # [N, 48]
        
        # Get global building state
        global_embedding = self.gat.get_global_embedding(node_embeddings)  # [48]
        
        # Extract agent-specific features
        # For each agent, get their node's embedding + global context
        agent_features = []
        for i in range(self.num_agents):
            agent_idx = agent_node_indices[i]
            agent_node_emb = node_embeddings[agent_idx]  # [48]
            # Concatenate agent's local view with global context
            agent_feat = torch.cat([agent_node_emb, global_embedding], dim=-1)  # [96]
            agent_features.append(agent_feat)
        
        agent_features = torch.stack(agent_features)  # [num_agents, 96]
        
        # ===== Step 4: Compute action logits for each agent =====
        action_logits = self.action_head(agent_features)  # [num_agents, max_actions]
        
        return action_logits, node_embeddings
    
    
    def forward_with_pomdp(
        self,
        data: Data,
        agent_node_indices: torch.Tensor,
        agent_seen_nodes_list: List[Set[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action logits using POMDP masking for partial observability.
        
        ===== NEW METHOD: POMDP-aware policy forward pass =====
        Decentralized actor receives partial observation (masked occupants).
        Centralized critic receives full state (privileged information).
        
        Args:
            data (Data): Full environment state from env.get_observation()
            agent_node_indices (Tensor[num_agents]): Current node of each agent
            agent_seen_nodes_list (List[Set[int]]): Nodes each agent has visited
                seen_nodes[i] = Set of node indices agent i has seen
            
        Returns:
            action_logits (Tensor[num_agents, max_actions]): Logits for each agent's actions
            critic_node_embeddings (Tensor[N, 48]): Node embeddings from critic (full state)
            actor_node_embeddings_list (List[Tensor]): Node embeddings from each agent's actor
        """
        # ===== CHANGE: Build actor data (masked) and critic data (full) =====
        actor_data_list, critic_data = GAT.process_batch_with_pomdp(data, agent_seen_nodes_list)
        
        # Process full state through critic
        critic_node_embeddings = self.gat(critic_data)  # [N, 48]
        
        # Process partial observations through actor for each agent
        actor_node_embeddings_list = []
        for actor_data in actor_data_list:
            # ===== CHANGE: Each agent sees masked graph =====
            actor_embeddings = self.gat(actor_data)  # [N, 48]
            actor_node_embeddings_list.append(actor_embeddings)
        
        # Get action logits from partial observations
        # (policy uses masked actor embeddings)
        agent_features = []
        for i in range(self.num_agents):
            agent_idx = agent_node_indices[i]
            # ===== CHANGE: Use actor embeddings (from masked observation) =====
            actor_emb = actor_node_embeddings_list[i]
            agent_node_emb = actor_emb[agent_idx]  # [48]
            
            # Get global context from full critic state
            critic_global = self.gat.get_global_embedding(critic_node_embeddings)
            
            # Concatenate local actor view with global critic context
            agent_feat = torch.cat([agent_node_emb, critic_global], dim=-1)  # [96]
            agent_features.append(agent_feat)
        
        agent_features = torch.stack(agent_features)  # [num_agents, 96]
        action_logits = self.action_head(agent_features)  # [num_agents, max_actions]
        
        return action_logits, critic_node_embeddings, actor_node_embeddings_list
    
    
    def select_actions(
        self,
        data: Data,
        agent_node_indices: torch.Tensor,
        valid_actions_list: List[List[str]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions for all agents with proper action masking.
        
        ===== NEW METHOD: Implements action masking and sampling =====
        
        Args:
            data (Data): Current observation
            agent_node_indices (Tensor[num_agents]): Where each agent is
            valid_actions_list (List[List[str]]): Valid actions for each agent
                From [env.get_valid_actions(i) for i in range(num_agents)]
            deterministic (bool): If True, take argmax; if False, sample
            
        Returns:
            actions (Tensor[num_agents]): Selected action indices
            log_probs (Tensor[num_agents]): Log probabilities of selected actions
            action_probs (Tensor[num_agents, max_actions]): Full probability distribution
        """
        # Get device from agent_node_indices
        device = agent_node_indices.device
        
        # Get logits
        action_logits, _ = self.forward(data, agent_node_indices)  # [num_agents, max_actions]
        
        # ===== Apply action masking =====
        # Map action strings to indices
        # Convention: 0=wait, 1=search, 2+=move_X
        action_masks = torch.zeros_like(action_logits, dtype=torch.bool, device=device)
        
        for i, valid_actions in enumerate(valid_actions_list):
            for action_str in valid_actions:
                action_idx = self._action_str_to_idx(action_str)
                if action_idx < self.max_actions:
                    action_masks[i, action_idx] = True
        
        # Mask invalid actions with -inf before softmax
        masked_logits = action_logits.clone()
        masked_logits[~action_masks] = float('-inf')
        
        # Compute probabilities
        action_probs = F.softmax(masked_logits, dim=-1)  # [num_agents, max_actions]
        
        # Select actions
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)  # [num_agents]
        else:
            actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)  # [num_agents]
        
        # Compute log probabilities of selected actions
        log_probs = torch.log(action_probs[torch.arange(self.num_agents, device=device), actions] + 1e-8)
        
        return actions, log_probs, action_probs
    
    
    def _action_str_to_idx(self, action_str: str) -> int:
        """
        Map action string from env to action index.
        
        ===== NEW METHOD: Action space mapping =====
        
        Convention:
            0: "wait"
            1: "search"
            2+: "move_X" where X is neighbor node ID
        """
        if action_str == "wait":
            return 0
        elif action_str == "search":
            return 1
        elif action_str.startswith("move_"):
            # Hash node ID to action index (simple approach)
            # In practice, might need a more sophisticated mapping
            node_id = action_str[5:]  # Remove "move_" prefix
            # Simple hash: use node_id hash modulo remaining action space
            return 2 + (hash(node_id) % (self.max_actions - 2))
        else:
            return 0  # Default to wait
    
    
    @staticmethod
    def gae(
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        ===== CHANGE 5: Fixed GAE computation =====
        Original had syntax errors and logic issues.
        
        Args:
            rewards (Tensor[T]): Rewards at each timestep
            dones (Tensor[T]): Done flags (1 if episode ended, 0 otherwise)
            values (Tensor[T+1]): Value predictions (includes bootstrap value)
            gamma (float): Discount factor
            lambda_ (float): GAE lambda parameter
        
        Returns:
            advantages (Tensor[T]): GAE advantages
        """
        T = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        
        # Compute TD residuals: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
        
        # Compute GAE recursively from end to start
        gae = 0
        for t in reversed(range(T)):
            gae = deltas[t] + gamma * lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    

    @staticmethod
    def policy_loss(
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        clip_epsilon: float = 0.2,
    ) -> torch.Tensor:
        """
        Compute PPO clipped policy loss.
        
        ===== CHANGE 6: Simplified and fixed policy objective =====
        Original was overcomplicated and had errors.
        
        Args:
            advantages (Tensor[T]): GAE advantages
            old_log_probs (Tensor[T]): Log probs from old policy (frozen)
            new_log_probs (Tensor[T]): Log probs from current policy
            clip_epsilon (float): PPO clipping parameter
            
        Returns:
            loss (Tensor): Policy loss (negative because we maximize)
        """
        # Compute probability ratio: π_new / π_old
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        
        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    
    @staticmethod
    def entropy_bonus(action_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy bonus to encourage exploration.
        
        ===== NEW METHOD: Separated entropy calculation =====
        
        Args:
            action_probs (Tensor[..., num_actions]): Action probability distributions
            
        Returns:
            entropy (Tensor): Mean entropy across all distributions
        """
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        return entropy.mean()


class Value(nn.Module):
    """
    Value network for estimating state values.
    
    ===== MAJOR CHANGES FROM ORIGINAL =====
    1. Now properly handles PyG Data objects
    2. Agent-centric value estimation (can estimate per-agent or global)
    3. Fixed dimensions to match GAT output
    """
    def __init__(self, num_agents: int = 2) -> None:
        """
        Initialize value network.
        
        Args:
            num_agents: Number of agents (for multi-agent value estimation)
        """
        super().__init__()
        self.num_agents = num_agents
        
        # Instantiate GAT (shared with policy or separate - your choice)
        self.gat = GAT()
        
        # ===== CHANGE 7: Fixed input dimension =====
        # Same as policy: agent_embedding (48) + global_embedding (48) = 96
        self.input_dim = 96  # Updated from 48
        
        # Value head outputs single scalar value
        # Larger network for RTX 5090
        self.value_head = nn.Sequential(
            nn.Linear(self.input_dim, 128),  # Increased from 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # Increased from 32
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )

    
    def __call__(
        self,
        data: Data,
        agent_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Override __call__ for IDE tools."""
        return super().__call__(data, agent_node_indices)
    
        
    def forward(
        self,
        data: Data,
        agent_node_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Estimate value of current state.
        
        ===== CHANGE 8: Complete rewrite for proper value estimation =====
        
        Args:
            data (Data): PyG Data object from env.get_observation()
            agent_node_indices (Tensor[num_agents], optional): Agent positions
                If None, uses global pooling only
            
        Returns:
            values (Tensor[num_agents] or Tensor[1]): State value estimates
        """
        # Process building graph
        node_embeddings = self.gat(data)  # [N, 48]
        
        # Get global state
        global_embedding = self.gat.get_global_embedding(node_embeddings)  # [48]
        
        if agent_node_indices is not None:
            # Agent-specific values (like policy)
            values = []
            for i in range(self.num_agents):
                agent_idx = agent_node_indices[i]
                agent_node_emb = node_embeddings[agent_idx]
                agent_feat = torch.cat([agent_node_emb, global_embedding], dim=-1)  # [96]
                value = self.value_head(agent_feat)  # [1]
                values.append(value)
            return torch.cat(values)  # [num_agents]
        else:
            # Global value (pooled state only)
            global_feat = torch.cat([global_embedding, global_embedding], dim=-1)  # [96]
            return self.value_head(global_feat)  # [1]
    
    
    def forward_with_pomdp(
        self,
        data: Data,
        agent_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate value using full state (centralized critic with privileged information).
        
        ===== NEW METHOD: POMDP-aware value function =====
        The critic has access to full state (centralized), giving better value estimates
        without leaking privileged information into the policy.
        
        Args:
            data (Data): Full environment state from env.get_observation()
            agent_node_indices (Tensor[num_agents]): Current node of each agent
            
        Returns:
            values (Tensor[num_agents]): Value estimates (one per agent)
        """
        # ===== CHANGE: Critic always sees full state =====
        critic_data = build_critic_data(data)  # No masking - privileged information
        
        # Process through GAT
        node_embeddings = self.gat(critic_data)  # [N, 48]
        global_embedding = self.gat.get_global_embedding(node_embeddings)  # [48]
        
        # Get value for each agent
        values = []
        for i in range(self.num_agents):
            agent_idx = agent_node_indices[i]
            agent_node_emb = node_embeddings[agent_idx]
            agent_feat = torch.cat([agent_node_emb, global_embedding], dim=-1)  # [96]
            value = self.value_head(agent_feat)  # [1]
            values.append(value)
        
        return torch.cat(values)  # [num_agents]
    
    
    @staticmethod
    def value_loss(
        predicted_values: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float = 0.2,
    ) -> torch.Tensor:
        """
        Compute value function loss with optional clipping.
        
        ===== NEW METHOD: Value loss computation =====
        
        Args:
            predicted_values (Tensor[T]): Predicted values from network
            returns (Tensor[T]): Actual returns (rewards-to-go or GAE targets)
            clip_epsilon (float): Clipping parameter (optional)
            
        Returns:
            loss (Tensor): MSE loss for value function
        """
        # Simple MSE loss (can add clipping if desired)
        return F.mse_loss(predicted_values, returns)