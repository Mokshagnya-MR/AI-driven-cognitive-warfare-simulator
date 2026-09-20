from __future__ import annotations

from typing import Any

from .train import TrainConfig, train_and_evaluate as _train_and_evaluate


def train_and_evaluate(*args: Any, **kwargs: Any):
	return _train_and_evaluate(*args, **kwargs)


__all__ = ["TrainConfig", "train_and_evaluate"]
