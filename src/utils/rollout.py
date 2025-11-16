import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class RolloutBuffer(Dataset):
    '''
    Buffer to store rollouts of (Obs, Action, Rew, Dones) for training/batching
    '''
    def __init__(
            self,
            obs: torch.Tensor,
            edgelist: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor
    ) -> None:
        self.obs = obs
        self.edgelist = edgelist
        # Discrete action space
        self.actions = actions
        # Scalar rewards
        self.rewards = rewards
        # Binary flags indicating whether an episode has concluded
        self.dones = dones
      
        
    def __len__(self,) -> int:
        return len(self.dones)
    
    
    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:
        return {
            'edges': self.edgelist[idx],
            'obs': self.obs[idx],
            'next_obs': self.obs[idx+1],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx]
        }