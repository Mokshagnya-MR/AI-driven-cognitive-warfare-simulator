from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Sequence

import json
import os
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from app.core.config import BACKEND_MODEL_PATH, FEATURE_NAMES, MODEL_METADATA_PATH, MODEL_PATH, MODEL_STATE_DICT_PATH, PROCESSED_GRAPH_PATH
from app.core.logger import get_logger


logger = get_logger(__name__)
ALLOW_TORCHSCRIPT_FALLBACK = os.getenv("ALLOW_TORCHSCRIPT_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}


def _short_exception_message(exc: Exception, max_len: int = 220) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    first_line = text.splitlines()[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 3] + "..."


class FallbackDetectionModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        probability = self.predict_proba(features)[:, 1]
        return (probability >= 0.5).astype(int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = features.sum(axis=1)
        probability = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probability, probability])


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
    def __init__(self, in_channels: int, hidden_channels: int = 192, heads: int = 8, dropout: float = 0.4) -> None:
        super().__init__()
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


class TorchScriptDetectionModel:
    def __init__(self, model: torch.jit.ScriptModule, device: torch.device, feature_dim: int, feature_template: np.ndarray) -> None:
        self.model = model
        self.device = device
        self.feature_dim = feature_dim
        self.feature_template = feature_template.astype(np.float32, copy=True)

    def _prepare_feature_tensor(self, features: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(features):
            array = features.detach().cpu().numpy()
        else:
            array = np.asarray(features, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim != 2:
            array = array.reshape(array.shape[0], -1)

        if array.shape[1] == self.feature_dim:
            prepared = array.astype(np.float32, copy=False)
        else:
            prepared = np.repeat(self.feature_template.reshape(1, -1), repeats=array.shape[0], axis=0)
            insert_start = 256 if self.feature_dim >= 260 else 0
            insert_end = min(self.feature_dim, insert_start + array.shape[1])
            prepared[:, insert_start:insert_end] = array[:, : insert_end - insert_start]

        return torch.tensor(prepared, dtype=torch.float32, device=self.device)

    def _prepare_graph_inputs(self, feature_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_count = int(feature_tensor.shape[0])
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        batch = torch.arange(node_count, dtype=torch.long, device=self.device)
        return feature_tensor, edge_index, batch

    def predict_positive_probabilities(self, features: Sequence[float] | np.ndarray | torch.Tensor) -> np.ndarray:
        feature_tensor = self._prepare_feature_tensor(features)

        with torch.no_grad():
            try:
                logits = self.model(feature_tensor)
            except Exception:
                graph_x, edge_index, batch = self._prepare_graph_inputs(feature_tensor)
                logits = self.model(graph_x, edge_index, batch)

            probabilities = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32, device=self.device)).detach().cpu().numpy()

        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        return probabilities

    def predict(self, features: Sequence[float] | np.ndarray | torch.Tensor, threshold: float) -> Dict[str, Any]:
        positive_probability = float(self.predict_positive_probabilities(features)[0])
        prediction = int(positive_probability > threshold)
        confidence = float(max(positive_probability, 1.0 - positive_probability))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "threshold_used": float(threshold),
        }


class TrainedGNNDetectionModel:
    def __init__(self, model: GraphGATClassifier, device: torch.device, feature_dim: int, feature_template: np.ndarray) -> None:
        self.model = model
        self.device = device
        self.feature_dim = feature_dim
        self.feature_template = feature_template.astype(np.float32, copy=True)

    def _prepare_feature_tensor(self, features: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(features):
            array = features.detach().cpu().numpy()
        else:
            array = np.asarray(features, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim != 2:
            array = array.reshape(array.shape[0], -1)

        if array.shape[1] == self.feature_dim:
            prepared = array.astype(np.float32, copy=False)
        else:
            prepared = np.repeat(self.feature_template.reshape(1, -1), repeats=array.shape[0], axis=0)
            copy_width = min(array.shape[1], 4, self.feature_dim)
            prepared[:, :copy_width] = array[:, :copy_width]

        return torch.tensor(prepared, dtype=torch.float32, device=self.device)

    def _predict_single_probability(self, feature_row: torch.Tensor) -> float:
        x = feature_row.unsqueeze(0)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        batch = torch.zeros((1,), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x, edge_index, batch)
            probability = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32, device=self.device)).item()
        return float(probability)

    def _tabular_adapter_probability(self, feature_row: torch.Tensor) -> float:
        row = feature_row.detach().cpu().numpy().astype(np.float32)
        velocity = float(row[0])
        bot_ratio = float(row[1])
        echo_density = float(row[2])
        depth = float(row[3])

        velocity_term = np.clip(velocity / 14.0, 0.0, 1.5)
        bot_term = np.clip(bot_ratio / 0.25, 0.0, 1.5)
        echo_term = np.clip(echo_density / 0.55, 0.0, 1.5)
        depth_term = np.clip(depth / 8.0, 0.0, 1.5)

        score = (
            1.35 * velocity_term
            + 2.2 * bot_term
            + 1.65 * echo_term
            + 0.85 * depth_term
            - 2.1
        )
        return float(1.0 / (1.0 + np.exp(-score)))

    def predict_positive_probabilities(self, features: Sequence[float] | np.ndarray | torch.Tensor) -> np.ndarray:
        feature_tensor = self._prepare_feature_tensor(features)
        probabilities = []
        for row in feature_tensor:
            gnn_probability = self._predict_single_probability(row)
            adapter_probability = self._tabular_adapter_probability(row)

            if abs(gnn_probability - 0.5) < 0.08:
                blended_probability = adapter_probability
            else:
                blended_probability = float(0.35 * gnn_probability + 0.65 * adapter_probability)

            probabilities.append(blended_probability)

        return np.asarray(probabilities, dtype=np.float32)

    def predict(self, features: Sequence[float] | np.ndarray | torch.Tensor, threshold: float) -> Dict[str, Any]:
        positive_probability = float(self.predict_positive_probabilities(features)[0])
        prediction = int(positive_probability > threshold)
        confidence = float(max(positive_probability, 1.0 - positive_probability))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "threshold_used": float(threshold),
        }


class ModelService:
    _instance: Optional["ModelService"] = None
    _lock = Lock()

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.metadata_path = MODEL_METADATA_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names = list(FEATURE_NAMES)
        self.feature_dim = len(self.feature_names)
        self.feature_template = np.zeros((self.feature_dim,), dtype=np.float32)
        self._load_feature_template()
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        self.threshold = self._load_threshold(self.metadata)
        self.model_info = self._extract_model_info(self.metadata)
        self.is_fallback = isinstance(self.model, FallbackDetectionModel)
        self.model_kind = type(self.model).__name__
        logger.info("Model loaded: %s | threshold=%.6f", self.model_kind, self.threshold)

    def _load_feature_template(self) -> None:
        try:
            processed_graph_path = PROCESSED_GRAPH_PATH
            if not processed_graph_path.exists():
                return
            payload = torch.load(str(processed_graph_path), map_location="cpu", weights_only=False)
            graph = payload.get("data") if isinstance(payload, dict) else None
            if graph is None or not hasattr(graph, "x"):
                return
            self.feature_dim = int(graph.x.shape[1])
            self.feature_template = graph.x.detach().float().mean(dim=0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("Unable to derive feature template from processed graph: %s", exc)
            self.feature_dim = max(self.feature_dim, len(self.feature_names))
            self.feature_template = np.zeros((self.feature_dim,), dtype=np.float32)

    def _load_model(self) -> Any:
        trained_gnn_model = self._load_trained_gnn_model()
        if trained_gnn_model is not None:
            logger.info("Using trained GNN state dict for inference")
            return trained_gnn_model

        if not ALLOW_TORCHSCRIPT_FALLBACK:
            logger.warning(
                "State dict loading failed and TorchScript fallback is disabled; using trained fallback model"
            )
            return self._load_trained_fallback_model(skip_gnn_attempt=True)

        if not self.model_path.exists():
            logger.warning("TorchScript model missing at %s, using fallback model", self.model_path)
            return self._load_trained_fallback_model(skip_gnn_attempt=True)

        try:
            logger.info("Loading model from %s", self.model_path)
            scripted_model = torch.jit.load(str(self.model_path), map_location=self.device)
            scripted_model.eval()
            return TorchScriptDetectionModel(
                model=scripted_model,
                device=self.device,
                feature_dim=self.feature_dim,
                feature_template=self.feature_template,
            )
        except Exception as exc:
            logger.exception(
                "Failed to load TorchScript model, using trained fallback model. reason=%s",
                _short_exception_message(exc),
            )
            return self._load_trained_fallback_model(skip_gnn_attempt=True)

    def _infer_in_channels_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
        candidate_keys = ("block1.conv.lin.weight", "block1.residual.weight")
        for key in candidate_keys:
            tensor = state_dict.get(key)
            if torch.is_tensor(tensor) and tensor.ndim == 2:
                return int(tensor.shape[1])
        return None

    def _feature_template_for_dim(self, target_dim: int) -> np.ndarray:
        base = np.asarray(self.feature_template, dtype=np.float32).reshape(-1)
        if target_dim <= 0:
            return base.copy()
        if base.shape[0] == target_dim:
            return base.copy()
        expanded = np.zeros((target_dim,), dtype=np.float32)
        copy_width = min(base.shape[0], target_dim, len(self.feature_names))
        if copy_width > 0:
            expanded[:copy_width] = base[:copy_width]
        return expanded

    def _load_trained_gnn_model(self) -> Any:
        if not MODEL_STATE_DICT_PATH.exists():
            logger.warning("Trained GNN state dict missing at %s", MODEL_STATE_DICT_PATH)
            return None

        try:
            logger.info("Loading trained GNN state dict from %s", MODEL_STATE_DICT_PATH)
            state_dict = torch.load(str(MODEL_STATE_DICT_PATH), map_location=self.device)
            inferred_in_channels = self._infer_in_channels_from_state_dict(state_dict) or self.feature_dim
            if inferred_in_channels != self.feature_dim:
                logger.warning(
                    "Checkpoint input feature dim (%s) differs from configured dim (%s); adapting runtime loader",
                    inferred_in_channels,
                    self.feature_dim,
                )

            model = GraphGATClassifier(in_channels=inferred_in_channels, hidden_channels=192, heads=8, dropout=0.4)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return TrainedGNNDetectionModel(
                model=model,
                device=self.device,
                feature_dim=inferred_in_channels,
                feature_template=self._feature_template_for_dim(inferred_in_channels),
            )
        except RuntimeError as exc:
            logger.exception(
                "Strict state dict load failed; attempting strict=False diagnostic load. reason=%s",
                _short_exception_message(exc),
            )
            try:
                state_dict = torch.load(str(MODEL_STATE_DICT_PATH), map_location=self.device)
                inferred_in_channels = self._infer_in_channels_from_state_dict(state_dict) or self.feature_dim
                model = GraphGATClassifier(in_channels=inferred_in_channels, hidden_channels=192, heads=8, dropout=0.4)
                incompatibility = model.load_state_dict(state_dict, strict=False)
                missing_keys = list(incompatibility.missing_keys)
                unexpected_keys = list(incompatibility.unexpected_keys)
                logger.warning(
                    "Loaded GNN state dict with strict=False (diagnostic mode). missing_keys=%s unexpected_keys=%s",
                    missing_keys,
                    unexpected_keys,
                )
                model.to(self.device)
                model.eval()
                return TrainedGNNDetectionModel(
                    model=model,
                    device=self.device,
                    feature_dim=inferred_in_channels,
                    feature_template=self._feature_template_for_dim(inferred_in_channels),
                )
            except Exception as diag_exc:
                logger.exception(
                    "strict=False diagnostic load also failed for trained GNN state dict: %s",
                    _short_exception_message(diag_exc),
                )
                return None
        except Exception as exc:
            logger.exception("Failed to load trained GNN state dict: %s", _short_exception_message(exc))
            return None

    def _load_trained_fallback_model(self, skip_gnn_attempt: bool = False) -> Any:
        if not skip_gnn_attempt:
            trained_gnn_model = self._load_trained_gnn_model()
            if trained_gnn_model is not None:
                return trained_gnn_model

        if BACKEND_MODEL_PATH.exists():
            try:
                logger.info("Loading trained fallback model from %s", BACKEND_MODEL_PATH)
                return joblib.load(BACKEND_MODEL_PATH)
            except Exception as exc:
                logger.warning("Failed to load trained fallback model, using dummy model: %s", _short_exception_message(exc))
        else:
            logger.warning("Trained fallback model missing at %s, using dummy model", BACKEND_MODEL_PATH)
        return FallbackDetectionModel()

    def _load_metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            logger.warning("Model metadata missing at %s, using default threshold 0.5", self.metadata_path)
            return {}

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                logger.warning("Model metadata has unexpected format, using defaults")
                return {}
            return metadata
        except Exception as exc:
            logger.warning("Failed to load model metadata, using default threshold: %s", exc)
            return {}

    def _load_threshold(self, metadata: Dict[str, Any]) -> float:
        threshold = metadata.get("threshold_used", metadata.get("threshold", 0.5))
        return float(np.clip(threshold, 0.0, 1.0))

    def _extract_model_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_type": metadata.get("model_type"),
            "feature_names": metadata.get("feature_names", []),
            "validation": metadata.get("validation", {}),
            "test": metadata.get("test", {}),
        }

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, features: Sequence[float] | np.ndarray | torch.Tensor) -> Dict[str, Any]:
        positive_probability, confidence = self._predict_probability_and_confidence(features)
        prediction = int(positive_probability > self.threshold)
        logger.info(
            "Inference complete | probability=%.6f prediction=%s threshold=%.6f",
            positive_probability,
            prediction,
            self.threshold,
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "threshold_used": self.threshold,
        }

    def predict_positive_probabilities(self, features: Sequence[float] | np.ndarray | torch.Tensor) -> np.ndarray:
        if torch.is_tensor(features):
            array = features.detach().cpu().numpy().astype(np.float32)
        else:
            array = np.asarray(features, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if isinstance(self.model, (TorchScriptDetectionModel, TrainedGNNDetectionModel)):
            probabilities = self.model.predict_positive_probabilities(array)
            return np.asarray(probabilities, dtype=float)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(array)
            return np.asarray(probabilities[:, 1], dtype=float)
        if hasattr(self.model, "decision_function"):
            scores = np.ravel(self.model.decision_function(array)).astype(float)
            return 1.0 / (1.0 + np.exp(-scores))

        return np.full(shape=(array.shape[0],), fill_value=0.5, dtype=float)

    def _predict_probability_and_confidence(self, features: Sequence[float] | np.ndarray | torch.Tensor) -> tuple[float, float]:
        probabilities = self.predict_positive_probabilities(features)
        positive_probability = float(probabilities[0])

        if isinstance(self.model, (TorchScriptDetectionModel, TrainedGNNDetectionModel)):
            confidence = max(positive_probability, 1.0 - positive_probability)
        elif hasattr(self.model, "predict_proba"):
            if torch.is_tensor(features):
                array = features.detach().cpu().numpy().astype(np.float32)
            else:
                array = np.asarray(features, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            row = np.asarray(self.model.predict_proba(array)[0], dtype=float)
            confidence = float(np.max(row))
        else:
            confidence = max(positive_probability, 1.0 - positive_probability)

        return positive_probability, confidence


def get_model_service() -> ModelService:
    return ModelService.get_instance()


def predict(features: Sequence[float]) -> Dict[str, Any]:
    return get_model_service().predict(features)