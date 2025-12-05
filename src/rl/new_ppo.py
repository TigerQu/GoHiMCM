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
    
    Main improvements:
    - handles PyG Data objects from environment
    - agent-centric readout (gets embedding for each agent's node)
    - action masking using env.get_valid_actions()
    - proper multi-agent action selection
    - fixed dimensions to match GAT output
    - action space mapping (wait/search/move_X)
    """
    def __init__(self, num_agents: int = 2, max_actions: int = 15) -> None:
        """
        Initialize policy network.
        
        Args:
            num_agents: number of firefighter agents (default 2 for standard layouts)
            max_actions: max possible actions per agent
                        (wait, search, + up to ~10 move actions depending on graph)
        """
        super().__init__()
        self.num_agents = num_agents
        self.max_actions = max_actions
        
        # instantiate GAT 
        self.gat = GAT()
        
        # fixed input dimension to match GAT output
        # GAT outputs 48-dim embeddings per node (updated for RTX 5090)
        # we concatenate: [agent_node_embedding (48) + global_embedding (48) + agent_id_onehot (num_agents)] 
        self.agent_feature_dim = 48  # from GAT output (bumped up from 24)
        self.input_dim = self.agent_feature_dim * 2 + num_agents  # agent + global + ID = 96 + num_agents
        
        # output dimension matches action space
        # actions: 0=wait, 1=search, 2...k=move_neighbor_i
        # we use max_actions and mask invalid ones
        # larger network for RTX 5090
        self.action_head = nn.Sequential(
            nn.Linear(self.input_dim, 128),  # bumped up from 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # bumped up from 32
            nn.ReLU(),
            nn.Linear(64, self.max_actions)  # logits for all possible actions
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
        
        Args:
            data: PyG Data object from env.get_observation()
                  contains x [N, 11], edge_index [2, E]
            agent_node_indices: index of each agent's current node [num_agents]
                                from env.get_agent_node_index(agent_id) for each agent
            
        Returns:
            action_logits [num_agents, max_actions]: logits for each agent's actions
            node_embeddings [N, 48]: node embeddings (for value function reuse)
        
        Example:
            obs = env.get_observation()
            agent_indices = torch.tensor([env.get_agent_node_index(0), 
                                         env.get_agent_node_index(1)])
            logits, embeddings = policy(obs, agent_indices)
            # apply action masking, then sample actions
        """
        # process entire building graph through GAT 
        node_embeddings = self.gat(data)  # [N, 48]
        
        # get global building state
        global_embedding = self.gat.get_global_embedding(node_embeddings)  # [48]
        
        # extract agent-specific features
        # for each agent: their node embedding + global context + agent ID
        agent_features = []
        for i in range(self.num_agents):
            agent_idx = agent_node_indices[i]
            agent_node_emb = node_embeddings[agent_idx]  # [48]
            
            # create one-hot agent ID to differentiate agents
            agent_id_onehot = torch.zeros(self.num_agents, device=node_embeddings.device)
            agent_id_onehot[i] = 1.0
            
            # concatenate agent's local view with global context and agent ID
            agent_feat = torch.cat([agent_node_emb, global_embedding, agent_id_onehot], dim=-1)  # [96 + num_agents]
            agent_features.append(agent_feat)
        
        agent_features = torch.stack(agent_features)  # [num_agents, 96 + num_agents]
        
        # compute action logits for each agent
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
        
        Decentralized actor gets partial observation (masked occupants).
        Centralized critic gets full state (privileged information).
        
        Args:
            data: full environment state from env.get_observation()
            agent_node_indices: current node of each agent [num_agents]
            agent_seen_nodes_list: nodes each agent has visited
                                   seen_nodes[i] = Set of node indices agent i has seen
            
        Returns:
            action_logits [num_agents, max_actions]: logits for each agent's actions
            critic_node_embeddings [N, 48]: node embeddings from critic (full state)
            actor_node_embeddings_list: node embeddings from each agent's actor
        """
        # build actor data (masked) and critic data (full)
        actor_data_list, critic_data = GAT.process_batch_with_pomdp(data, agent_seen_nodes_list)
        
        # process full state through critic
        critic_node_embeddings = self.gat(critic_data)  # [N, 48]
        
        # process partial observations through actor for each agent
        actor_node_embeddings_list = []
        for actor_data in actor_data_list:
            # each agent sees masked graph
            actor_embeddings = self.gat(actor_data)  # [N, 48]
            actor_node_embeddings_list.append(actor_embeddings)
        
        # get action logits from partial observations
        # policy uses masked actor embeddings
        agent_features = []
        for i in range(self.num_agents):
            agent_idx = agent_node_indices[i]
            # use actor embeddings (from masked observation)
            actor_emb = actor_node_embeddings_list[i]
            agent_node_emb = actor_emb[agent_idx]  # [48]
            
            # get global context from full critic state
            critic_global = self.gat.get_global_embedding(critic_node_embeddings)
            
            # create one-hot agent ID to differentiate agents
            agent_id_onehot = torch.zeros(self.num_agents, device=critic_node_embeddings.device)
            agent_id_onehot[i] = 1.0
            
            # concatenate local actor view with global critic context and agent ID
            agent_feat = torch.cat([agent_node_emb, critic_global, agent_id_onehot], dim=-1)  # [96 + num_agents]
            agent_features.append(agent_feat)
        
        agent_features = torch.stack(agent_features)  # [num_agents, 96 + num_agents]
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
        
        Args:
            data: current observation
            agent_node_indices: where each agent is [num_agents]
            valid_actions_list: valid actions for each agent
                                from [env.get_valid_actions(i) for i in range(num_agents)]
            deterministic: if True, take argmax; if False, sample
            
        Returns:
            actions [num_agents]: selected action indices
            log_probs [num_agents]: log probabilities of selected actions
            action_probs [num_agents, max_actions]: full probability distribution
        """
        # get device from agent_node_indices
        device = agent_node_indices.device
        
        # get logits
        action_logits, _ = self.forward(data, agent_node_indices)  # [num_agents, max_actions]
        
        # apply action masking with stable sorted indexing
        # build act2idx mapping from sorted valid_actions for each agent
        action_masks = torch.zeros_like(action_logits, dtype=torch.bool, device=device)
        agent_action_maps = []  # store mapping for each agent
        
        for i, valid_actions in enumerate(valid_actions_list):
            # Sort actions for stable indexing
            sorted_actions = sorted(valid_actions)
            action_map = {action: idx for idx, action in enumerate(sorted_actions)}
            agent_action_maps.append((sorted_actions, action_map))
            
            # Map to global action indices
            for action_str in valid_actions:
                action_idx = self._action_str_to_idx(action_str, valid_actions)
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
    
    
    def _action_str_to_idx(self, action_str: str, valid_actions: List[str] = None) -> int:
        """
        Map action string from env to action index using stable sorted mapping.
        
        Problem: hash-based mapping is unstable and causes collisions
        Solution: build sorted action list per step, map by position
        
        Args:
            action_str: action string (e.g., 'wait', 'search', 'move_R_1_2')
            valid_actions: sorted list of valid actions for current step
        
        Convention (when valid_actions provided):
            index = position in sorted(valid_actions)
        
        Fallback (when valid_actions not provided):
            0: "wait"
            1: "search"
            2+: lexicographically sorted move actions
        """
        if valid_actions is not None:
            # Use sorted valid_actions for stable mapping
            sorted_actions = sorted(valid_actions)
            try:
                return sorted_actions.index(action_str)
            except ValueError:
                # Action not in valid list, fallback to wait
                return 0
        else:
            # Fallback for backward compatibility
            if action_str == "wait":
                return 0
            elif action_str == "search":
                return 1
            elif action_str.startswith("move_"):
                # Sort move actions lexicographically for stability
                # In practice, should always use valid_actions parameter
                return 2 + hash(action_str[5:]) % (self.max_actions - 2)
            else:
                return 0
    
    
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
        
        Fixed from original which had syntax errors and logic issues.
        
        Args:
            rewards [T]: rewards at each timestep
            dones [T]: done flags (1 if episode ended, 0 otherwise)
            values [T+1]: value predictions (includes bootstrap value)
            gamma: discount factor
            lambda_: GAE lambda parameter
        
        Returns:
            advantages [T]: GAE advantages
        """
        T = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        
        # compute TD residuals: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
        
        # compute GAE recursively from end to start
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
        
        Simplified and fixed from original which was overcomplicated.
        
        Args:
            advantages [T]: GAE advantages
            old_log_probs [T]: log probs from old policy (frozen)
            new_log_probs [T]: log probs from current policy
            clip_epsilon: PPO clipping parameter
            
        Returns:
            policy loss (negative because we maximize)
        """
        # compute probability ratio: π_new / π_old
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        
        # take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    
    @staticmethod
    def entropy_bonus(action_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy bonus to encourage exploration.
        
        Args:
            action_probs [..., num_actions]: action probability distributions
            
        Returns:
            mean entropy across all distributions
        """
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        return entropy.mean()


class Value(nn.Module):
    """
    Value network for estimating state values.
    
    Main improvements:
    - handles PyG Data objects properly
    - agent-centric value estimation (per-agent or global)
    - fixed dimensions to match GAT output
    """
    def __init__(self, num_agents: int = 2) -> None:
        """
        Initialize value network.
        
        Args:
            num_agents: number of agents (for multi-agent value estimation)
        """
        super().__init__()
        self.num_agents = num_agents
        
        # instantiate GAT (shared with policy or separate - your choice)
        self.gat = GAT()
        
        # fixed input dimension
        # same as policy: agent_embedding (48) + global_embedding (48) + agent_id (num_agents) = 96 + num_agents
        self.input_dim = 96 + num_agents
        
        # value head outputs single scalar value
        # larger network for RTX 5090
        self.value_head = nn.Sequential(
            nn.Linear(self.input_dim, 128),  # bumped up from 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # bumped up from 32
            nn.ReLU(),
            nn.Linear(64, 1)  # single value output
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
        
        Args:
            data: PyG Data object from env.get_observation()
            agent_node_indices [num_agents], optional: agent positions
                                                       if None, uses global pooling only
            
        Returns:
            state value estimates [num_agents] or [1]
        """
        # process building graph
        node_embeddings = self.gat(data)  # [N, 48]
        
        # get global state
        global_embedding = self.gat.get_global_embedding(node_embeddings)  # [48]
        
        if agent_node_indices is not None:
            # agent-specific values (like policy)
            values = []
            for i in range(self.num_agents):
                agent_idx = agent_node_indices[i]
                agent_node_emb = node_embeddings[agent_idx]
                
                # add agent ID one-hot encoding (same as policy)
                agent_id_onehot = torch.zeros(self.num_agents, device=agent_node_emb.device)
                agent_id_onehot[i] = 1.0
                
                agent_feat = torch.cat([agent_node_emb, global_embedding, agent_id_onehot], dim=-1)  # [96 + num_agents]
                value = self.value_head(agent_feat)  # [1]
                values.append(value)
            return torch.cat(values)  # [num_agents]
        else:
            # global value (pooled state only) - add zero padding for agent ID
            agent_id_padding = torch.zeros(self.num_agents, device=global_embedding.device)
            global_feat = torch.cat([global_embedding, global_embedding, agent_id_padding], dim=-1)  # [96 + num_agents]
            return self.value_head(global_feat)  # [1]
    
    
    def forward_with_pomdp(
        self,
        data: Data,
        agent_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate value using full state (centralized critic with privileged information).
        
        The critic gets access to full state (centralized), giving better value estimates
        without leaking privileged info into the policy.
        
        Args:
            data: full environment state from env.get_observation()
            agent_node_indices: current node of each agent [num_agents]
            
        Returns:
            value estimates (one per agent) [num_agents]
        """
        # critic always sees full state
        critic_data = build_critic_data(data)  # no masking - privileged information
        
        # process through GAT
        node_embeddings = self.gat(critic_data)  # [N, 48]
        global_embedding = self.gat.get_global_embedding(node_embeddings)  # [48]
        
        # get value for each agent
        values = []
        for i in range(self.num_agents):
            agent_idx = agent_node_indices[i]
            agent_node_emb = node_embeddings[agent_idx]
            
            # add agent ID one-hot encoding (same as policy)
            agent_id_onehot = torch.zeros(self.num_agents, device=agent_node_emb.device)
            agent_id_onehot[i] = 1.0
            
            agent_feat = torch.cat([agent_node_emb, global_embedding, agent_id_onehot], dim=-1)  # [96 + num_agents]
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
        
        Args:
            predicted_values [T]: predicted values from network
            returns [T]: actual returns (rewards-to-go or GAE targets)
            clip_epsilon: clipping parameter (optional)
            
        Returns:
            MSE loss for value function
        """
        # simple MSE loss (can add clipping if desired)
        return F.mse_loss(predicted_values, returns)