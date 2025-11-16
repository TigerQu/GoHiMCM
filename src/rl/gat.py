import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphNorm

from utils.configs import CONFIGS


class GAT(nn.Module):
    '''
    Graph attention network used for the policy and value functions
    '''
    def __init__(self,) -> None:
        super().__init__()
        # Define the GAT layers of the network
        self.gat1: GATConv = GATConv(
                in_channels=10,
                out_channels=32,
                heads=CONFIGS['gat']['heads'],
                dropout=CONFIGS['gat']['dropout'],
            )
        self.gat2: GATConv = GATConv(
                in_channels=32,
                out_channels=48,
                heads=CONFIGS['gat']['heads'],
                dropout=CONFIGS['gat']['dropout'],
            )
        self.gat3: GATConv = GATConv(
                in_channels=48,
                out_channels=24,
                heads=CONFIGS['gat']['heads'],
                dropout=CONFIGS['gat']['dropout'],
            )
        # Define helper layers
        self.elu: nn.ELU = nn.ELU(CONFIGS['gat']['elu_parameter']) # Adds nonlinearity
        self.norm1, self.norm2 = GraphNorm(10), GraphNorm(32)
        
        
    def __call__(
        self,
        adjacency_list: torch.Tensor,
        feature_vectors: torch.Tensor,
    ) -> torch.Tensor:
        # Override call for IDE tools
        super().__call__(
            adjacency_list,
            feature_vectors,
        )
        
        
    def forward(
        self,
        adjacency_list: torch.Tensor,
        feature_vectors: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Pass an input x (observations in context) through the GAT
        
        Args:
            adjacency_list (Tensor[B, E, 2]): Adjacency list of node connections (B, E, 2)
            feature_vectors (Tensor[B, N, 10]): Feature vectors for all nodes (B, N, F)
            
        Returns:
            out (Tensor[B, 24]): Output of the GAT
        '''
        # Reverse edge dimensions
        torch.transpose(
            adjacency_list, 0, 1,
        )
        # Layer 1
        out = self.gat1(feature_vectors, adjacency_list)
        out = self.elu(
            self.norm1(out)
        )
        # Layer 2
        out = self.gat2(out, adjacency_list)
        out = self.elu(
            self.norm2(out)
        )
        # Layer 3
        out = self.gat3(out)