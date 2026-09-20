from __future__ import annotations

import logging
from typing import Dict, Literal

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_class_distribution(labels: np.ndarray) -> Dict[str, int]:
    if labels.size == 0:
        return {"negative": 0, "positive": 0}
    return {
        "negative": int(np.sum(labels == 0)),
        "positive": int(np.sum(labels == 1)),
    }


def prediction_distribution(probabilities: np.ndarray, threshold: float) -> Dict[str, int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "pred_negative": int(np.sum(predictions == 0)),
        "pred_positive": int(np.sum(predictions == 1)),
    }


def probability_histogram(probabilities: np.ndarray, bins: int = 10) -> Dict[str, list[float] | list[int]]:
    hist, edges = np.histogram(probabilities, bins=bins, range=(0.0, 1.0))
    return {"bin_edges": edges.astype(float).tolist(), "counts": hist.astype(int).tolist()}


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, float | list[list[int]]]:
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    predictions = (probabilities >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": cm,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    objective: Literal["f1", "recall"] = "f1",
    search_min: float = 0.05,
    search_max: float = 0.95,
    steps: int = 181,
) -> tuple[float, Dict[str, float | list[list[int]]]]:
    if y_true.size == 0:
        metrics = compute_metrics(y_true, probabilities, threshold=0.5)
        return 0.5, metrics

    thresholds = np.linspace(search_min, search_max, steps)
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, probabilities, threshold=best_threshold)

    for threshold in thresholds:
        metrics = compute_metrics(y_true, probabilities, threshold=float(threshold))
        if objective == "recall":
            score = float(metrics["recall"])
            tie_break = float(metrics["f1"])
            best_score = float(best_metrics["recall"])
            best_tie = float(best_metrics["f1"])
        else:
            score = float(metrics["f1"])
            tie_break = float(metrics["recall"])
            best_score = float(best_metrics["f1"])
            best_tie = float(best_metrics["recall"])

        if score > best_score or (score == best_score and tie_break > best_tie):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def log_eval_diagnostics(
    logger: logging.Logger,
    split_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    logger.info("%s class distribution: %s", split_name, compute_class_distribution(y_true))
    logger.info("%s prediction distribution @ threshold=%.3f: %s", split_name, threshold, prediction_distribution(probabilities, threshold))
    logger.info("%s probability histogram: %s", split_name, probability_histogram(probabilities))
