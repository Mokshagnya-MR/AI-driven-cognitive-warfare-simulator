from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from app.core.config import FEATURE_NAMES


def metrics_to_feature_vector(graph_stats: Mapping[str, Any], propagation_metrics: Mapping[str, Any]) -> list[float]:
    return [
        float(propagation_metrics.get("velocity", 0.0)),
        float(graph_stats.get("bot_ratio", 0.0)),
        float(propagation_metrics.get("echo_chamber_density", 0.0)),
        float(propagation_metrics.get("depth", 0.0)),
    ]


def normalize_feature_vector(features: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.astype(np.float32, copy=False)


def get_feature_names() -> list[str]:
    return list(FEATURE_NAMES)