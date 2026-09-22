from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from .train import TrainConfig, train_and_evaluate


class ResidualGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.residual = nn.Linear(in_channels, out_channels, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index) + self.residual(x)
        x = F.elu(self.norm(x))
        return F.dropout(x, p=self.dropout, training=self.training)


class GraphGCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.4) -> None:
        super().__init__()
        self.block1 = ResidualGCNBlock(in_channels, hidden_channels, dropout)
        self.block2 = ResidualGCNBlock(hidden_channels, hidden_channels, dropout)
        self.block3 = ResidualGCNBlock(hidden_channels, hidden_channels, dropout)
        self.block4 = ResidualGCNBlock(hidden_channels, hidden_channels, dropout)
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
        x = global_mean_pool(self.graph_norm(x), batch)
        return self.pool_mlp(x).squeeze(-1)


def train_gcn(output_dir: Path, device: torch.device, epochs: int = 60, early_stopping_patience: int = 10) -> dict[str, object]:
    payload = torch.load(output_dir / "processed_graph.pt", map_location="cpu", weights_only=False)
    graph = payload["data"]
    model = GraphGCNClassifier(in_channels=int(graph.x.shape[1]))
    training_dir = output_dir / "gcn_training"
    logger = logging.getLogger(f"gcn-{output_dir.name}")
    metrics = train_and_evaluate(
        model=model,
        graph_data=graph,
        labels=graph.y,
        output_dir=training_dir,
        logger=logger,
        device=device,
        config=TrainConfig(
            epochs=epochs,
            batch_size=256 if device.type == "cuda" else 64,
            gradient_accumulation_steps=1,
            early_stopping_patience=early_stopping_patience,
        ),
    )
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "gcn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the per-dataset GCN baseline")
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("dataset_names", nargs="+")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    for dataset_name in args.dataset_names:
        print(json.dumps(train_gcn(args.outputs_dir / dataset_name, device), indent=2))
