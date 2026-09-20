from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class ResidualGATBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int, dropout: float, concat: bool = True) -> None:
        super().__init__()
        self.dropout = dropout
        self.concat = concat
        self.conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)
        self.out_channels = out_channels * heads if concat else out_channels
        self.norm = nn.LayerNorm(self.out_channels)
        self.residual = nn.Linear(in_channels, self.out_channels, bias=False) if in_channels != self.out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x, edge_index)
        x = x + residual
        x = self.norm(x)
        x = F.elu(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class GraphGATClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 192,
        heads: int = 8,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.block1 = ResidualGATBlock(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.block2 = ResidualGATBlock(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.block3 = ResidualGATBlock(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.block4 = ResidualGATBlock(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False)

        self.graph_norm = nn.LayerNorm(hidden_channels)
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        x = self.block4(x, edge_index)
        x = self.graph_norm(x)
        x = global_mean_pool(x, batch)
        x = self.pool_mlp(x).squeeze(-1)
        return x
