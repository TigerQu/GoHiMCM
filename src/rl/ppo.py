import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.gat import GAT
from utils.configs import CONFIGS


class Policy(nn.Module):
    '''
    Policy network; uses GAT to process observation space
    '''
    def __init__(self,) -> None:
        super().__init__()
        # Instanciate GAT and dense layers
        self.gat: GAT = GAT()
        # Define the dense layer
        self.dense: nn.Sequential = nn.Sequential(
            nn.Linear(24, 36),
            nn.ReLU(),
            nn.Linear(36, 11)
        )
        
    
    def __call__(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        # Override __call__ for IDE tools
        super().__call__(obs,)
    
        
    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Compute actions of the policy based on observations
        
        Args:
            obs (Tensor[B, N, 10]): Observations from the environment
            
        Returns:
            action (Tensor[B, _]): Action to execute in environment
        '''
        # GNN output
        out: torch.Tensor = self.gat(obs)
        # Aggregate GNN output along node dimension
        out = torch.mean(
            out, dim=-2,
        )
        # Dense output
        out = self.dense(out)
        
      
    @classmethod
    def gae(
        cls,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        value_pred: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Computes GAE, a low variance/low bias advantage estimation function
        
        Args:
            rewards (Tensor[B]): Float environment rewards at every step in the rollout
            dones (Tensor[B]): Boolean flags indicating if an episode has been terminated at every rollout
            value_pred (Tensor[B]): Predicted value derived from the value function
        
        Returns:
            advantage (Tensor): Low variance/low bias advantage estimations
        '''
        # Pre-compute the TD residuals
        # Distance between bootstrapped value and predicted value
        td_residuals: torch.Tensor = (rewards + value_pred[1:] * CONFIGS['discount_factor']) - value_pred[:-1]
        # Iteratively compute GAE
        advantages: torch.Tensor = torch.zeros_like(rewards)
        for t in reversed(td_residuals.size(-1) - 1):
            advantages[t] = td_residuals[..., t] + CONFIGS['discount_factor'] * CONFIGS['gae_decay'] * (1 - dones[..., t+1]) * advantages[..., t+1]
        
        return advantages
    

    @classmethod
    def policy_objective(
        cls,
        advantages: torch.Tensor,
        past_actions: torch.Tensor,
        current_actions: torch.Tensor,
        action_idx: int
        
    ) -> torch.Tensor:
        '''
        Computes the policy objective of PPO
        Combines clipped surrogate objective, value objective, and entropy bonus
        
        Args:
            advantages (Tensor[B]): Pre-computed GAEs
            past_actions (Tensor[B, 3]): Probability distribution output by the historic policy
            current_actions (Tensor[B, 3]): Probability distribution output by the current policy
            action_idx (int): Index of action taken
            
        Returns:
            objective (Tensor[B])
        '''
        # Policy ratio to measure how much the current policy has diverged from the frozen policy
        # Metric to penalize rapid change
        policy_ratio: torch.Tensor = torch.log(
            torch.exp(current_actions[..., action_idx]) / torch.exp(past_actions[..., action_idx])
        )
        # Vanilla policy objective (clipped surrogate)
        policy_objective: torch.Tensor = torch.min(
            advantages * policy_ratio,
            advantages * torch.clip(
                policy_ratio, 1 + CONFIGS['clipping_param'], 1 - CONFIGS['clipping_param'],
            ),
        )
        # Entropy bonus
        entropy_bonus: torch.Tensor = -torch.sum(
            current_actions * torch.log(current_actions), dim=-1,
        )
        # Return the full objective
        return policy_objective + CONFIGS['entropy_coef'] * entropy_bonus


class Value(nn.Module):
    '''
    Value network; uses GAT to process observation space
    '''
    def __init__(self,) -> None:
        super().__init__()
        # Instanciate GAT and dense layers
        self.gat: GAT = GAT()
        # Define the dense layer
        self.dense: nn.Sequential = nn.Sequential(
            nn.Linear(24, 36),
            nn.ReLU(),
            nn.Linear(36, 1)
        )

    
    def __call__(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        # Override __call__ for IDE tools
        super().__call__(obs,)
    
        
    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Compute actions of the policy based on observations
        Policy is asynchronous, so value function uses more observations than policy
        
        Args:
            obs (Tensor[B, N, 10]): Observations from the environment
            
        Returns:
            value (Tensor[B]): Predicted value of the policy's action given observations
        '''
        # GNN output
        out: torch.Tensor = self.gat(obs)
        # Aggregate GNN output along node dimension
        out = torch.mean(
            out, dim=-2,
        )
        # Dense output
        out = self.dense(out)