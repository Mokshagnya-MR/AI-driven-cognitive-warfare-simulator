from __future__ import annotations

import copy
import json
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from .evaluate import compute_class_distribution, compute_metrics, find_optimal_threshold, log_eval_diagnostics


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 5e-4
    weight_decay: float = 5e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 12
    subgraph_hops: int = 1
    num_workers: int = 2
    use_amp: bool = True
    cache_subgraphs: bool = True
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    threshold_objective: str = "f1"


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma)
        return (focal * bce).mean()


class NodeSubgraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        full_graph: Data,
        node_indices: np.ndarray,
        labels: np.ndarray,
        hops: int = 1,
        cache_subgraphs: bool = True,
    ) -> None:
        self.full_graph = full_graph
        self.node_indices = node_indices.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.hops = hops
        self.cache_subgraphs = cache_subgraphs
        self._cached_samples: list[Data] | None = None

        if self.cache_subgraphs:
            self._cached_samples = [self._build_sample(i) for i in range(len(self.node_indices))]

    def _build_sample(self, idx: int) -> Data:
        center = int(self.node_indices[idx])
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            center,
            self.hops,
            self.full_graph.edge_index,
            relabel_nodes=True,
        )

        x = self.full_graph.x[subset]
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return Data(x=x, edge_index=sub_edge_index, y=y)

    def __len__(self) -> int:
        return len(self._cached_samples) if self._cached_samples is not None else len(self.node_indices)

    def __getitem__(self, idx: int) -> Data:
        if self._cached_samples is not None:
            return self._cached_samples[idx]
        return self._build_sample(idx)


def _mixed_precision_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _build_loss_fn(
    cfg: TrainConfig,
    train_labels: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, float]:
    pos = float(np.sum(train_labels == 1))
    neg = float(np.sum(train_labels == 0))
    pos_weight_val = 1.0 if pos <= 0 else max(1.0, neg / max(pos, 1.0))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    if cfg.use_focal_loss:
        return FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=cfg.focal_gamma), pos_weight_val
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), pos_weight_val


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with _mixed_precision_context(device, use_amp):
                logits = model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.view(-1)
                loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            losses.append(loss.item())
            all_labels.extend(labels.long().cpu().numpy().tolist())
            all_probs.extend(probs.float().cpu().numpy().tolist())

    labels_np = np.array(all_labels, dtype=np.int64)
    probs_np = np.array(all_probs, dtype=np.float32)
    avg_loss = float(np.mean(losses) if losses else 0.0)
    return avg_loss, labels_np, probs_np


def train_and_evaluate(
    model: nn.Module,
    graph_data: Data,
    labels: torch.Tensor,
    output_dir: str | Path,
    logger: logging.Logger,
    device: torch.device,
    backend_features: np.ndarray | None = None,
    backend_model_path: str | Path | None = None,
    config: TrainConfig | None = None,
) -> Dict[str, object]:
    cfg = config or TrainConfig()
    if device.type == "cuda" and config is None:
        cfg.batch_size = min(cfg.batch_size, 16)
        cfg.gradient_accumulation_steps = max(cfg.gradient_accumulation_steps, 2)
        if cfg.cache_subgraphs:
            cfg.num_workers = 0
        else:
            cfg.num_workers = min(max(cfg.num_workers, 2), min(4, (os.cpu_count() or 1)))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labeled_idx = torch.where(labels >= 0)[0].cpu().numpy()
    labeled_y = labels[labeled_idx].cpu().numpy().astype(np.int64)

    train_idx, tmp_idx, train_y, tmp_y = train_test_split(
        labeled_idx,
        labeled_y,
        test_size=0.2,
        random_state=42,
        stratify=labeled_y if len(np.unique(labeled_y)) > 1 else None,
    )
    val_idx, test_idx, val_y, test_y = train_test_split(
        tmp_idx,
        tmp_y,
        test_size=0.5,
        random_state=42,
        stratify=tmp_y if len(np.unique(tmp_y)) > 1 else None,
    )

    logger.info("Train class distribution: %s", compute_class_distribution(train_y))
    logger.info("Val class distribution: %s", compute_class_distribution(val_y))
    logger.info("Test class distribution: %s", compute_class_distribution(test_y))

    train_dataset = NodeSubgraphDataset(
        graph_data,
        train_idx,
        train_y,
        hops=cfg.subgraph_hops,
        cache_subgraphs=cfg.cache_subgraphs,
    )
    val_dataset = NodeSubgraphDataset(
        graph_data,
        val_idx,
        val_y,
        hops=cfg.subgraph_hops,
        cache_subgraphs=cfg.cache_subgraphs,
    )
    test_dataset = NodeSubgraphDataset(
        graph_data,
        test_idx,
        test_y,
        hops=cfg.subgraph_hops,
        cache_subgraphs=cfg.cache_subgraphs,
    )

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model.to(device)
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            logger.info("torch.compile unavailable for this model; continuing without it")

    criterion, pos_weight_value = _build_loss_fn(cfg, train_y, device)
    logger.info("Using weighted loss with pos_weight=%.4f", pos_weight_value)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    scaler = GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        accumulation_steps = max(1, int(cfg.gradient_accumulation_steps))

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with _mixed_precision_context(device, cfg.use_amp):
                logits = model(batch.x, batch.edge_index, batch.batch)
                y = batch.y.view(-1)
                loss = criterion(logits, y)
                loss_to_backprop = loss / accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
                if step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_to_backprop.backward()
                if step % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item())

        if len(train_loader) % accumulation_steps != 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = float(np.mean(train_losses) if train_losses else 0.0)

        val_loss, val_labels, val_probs = _evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            use_amp=cfg.use_amp,
        )
        val_threshold, val_metrics = find_optimal_threshold(
            y_true=val_labels,
            probabilities=val_probs,
            objective=cfg.threshold_objective if cfg.threshold_objective in {"f1", "recall"} else "f1",
        )
        scheduler.step(val_loss)

        logger.info(
            "Epoch %d | train_loss=%.4f val_loss=%.4f val_f1=%.4f val_recall=%.4f threshold=%.3f lr=%.6f",
            epoch,
            train_loss,
            val_loss,
            float(val_metrics["f1"]),
            float(val_metrics["recall"]),
            val_threshold,
            optimizer.param_groups[0]["lr"],
        )
        log_eval_diagnostics(logger, "Val", val_labels, val_probs, val_threshold)

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_threshold = float(val_threshold)
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_labels, test_probs = _evaluate_loader(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        use_amp=cfg.use_amp,
    )
    test_metrics = compute_metrics(test_labels, test_probs, threshold=best_threshold)
    log_eval_diagnostics(logger, "Test", test_labels, test_probs, best_threshold)

    backend_artifacts: dict[str, object] = {}
    if backend_features is not None and backend_model_path is not None:
        backend_array = np.asarray(backend_features, dtype=np.float32)
        if backend_array.shape[0] == len(labels):
            backend_train_x = backend_array[train_idx]
            backend_val_x = backend_array[val_idx]
            backend_test_x = backend_array[test_idx]

            backend_model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
            backend_model.fit(backend_train_x, train_y)

            backend_val_probs = backend_model.predict_proba(backend_val_x)[:, 1]
            backend_threshold, backend_val_metrics = find_optimal_threshold(
                y_true=val_y,
                probabilities=backend_val_probs,
                objective=cfg.threshold_objective if cfg.threshold_objective in {"f1", "recall"} else "f1",
            )
            backend_test_probs = backend_model.predict_proba(backend_test_x)[:, 1]
            backend_test_metrics = compute_metrics(test_y, backend_test_probs, threshold=backend_threshold)

            backend_path = Path(backend_model_path)
            backend_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(backend_model, backend_path)

            metadata_path = backend_path.with_name("model_metadata.json")
            backend_metadata = {
                "model_type": "logistic_regression_surrogate",
                "feature_names": ["velocity", "bot_ratio", "echo_density", "depth"],
                "threshold": float(backend_threshold),
                "validation": {
                    "accuracy": float(backend_val_metrics["accuracy"]),
                    "precision": float(backend_val_metrics["precision"]),
                    "recall": float(backend_val_metrics["recall"]),
                    "f1": float(backend_val_metrics["f1"]),
                },
                "test": {
                    "accuracy": float(backend_test_metrics["accuracy"]),
                    "precision": float(backend_test_metrics["precision"]),
                    "recall": float(backend_test_metrics["recall"]),
                    "f1": float(backend_test_metrics["f1"]),
                    "confusion_matrix": backend_test_metrics["confusion_matrix"],
                },
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(backend_metadata, f, indent=2)

            backend_artifacts = {
                "backend_model": str(backend_path),
                "backend_metadata": str(metadata_path),
                "backend_threshold": float(backend_threshold),
                "backend_test_f1": float(backend_test_metrics["f1"]),
            }

    state_dict_path = output_path / "model_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)

    export_source = getattr(model, "_orig_mod", model)
    export_model = copy.deepcopy(export_source).to("cpu")
    export_model.eval()
    scripted_model = torch.jit.script(export_model)
    scripted_path = output_path / "model_torchscript.pt"
    scripted_model.save(scripted_path)

    metrics_payload = {
        "test_loss": test_loss,
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "threshold_used": float(best_threshold),
        "best_val_f1": float(best_val_f1),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "artifacts": {
            "state_dict": str(state_dict_path.name),
            "torchscript": str(scripted_path.name),
            **backend_artifacts,
        },
    }

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload
