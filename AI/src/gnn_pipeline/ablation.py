from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


FEATURE_NAMES = ["velocity", "bot_ratio", "echo_density", "depth"]


def run_ablation(output_dir: Path, backend_dir: Path) -> dict[str, object]:
    sys.path.insert(0, str(backend_dir))
    from app.services.model_service import get_model_service
    from app.services.xai_service import get_shap_attributions

    features = np.load(output_dir / "backend_features.npy")
    labels = np.load(output_dir / "labels.npy").astype(np.int64)
    train_indices = np.load(output_dir / "splits" / "train_idx.npy")
    test_indices = np.load(output_dir / "splits" / "test_idx.npy")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics["threshold_used"])

    train_features = features[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    baseline_probabilities = get_model_service().predict_positive_probabilities(test_features)
    baseline_f1 = float(f1_score(test_labels, baseline_probabilities >= threshold, zero_division=0))
    means = train_features.mean(axis=0)

    ablations = []
    for index, feature_name in enumerate(FEATURE_NAMES):
        ablated = test_features.copy()
        ablated[:, index] = means[index]
        probabilities = get_model_service().predict_positive_probabilities(ablated)
        f1 = float(f1_score(test_labels, probabilities >= threshold, zero_division=0))
        ablations.append({"feature": feature_name, "f1": f1, "delta_vs_full": f1 - baseline_f1})

    sample = test_features[: min(20, len(test_features))]
    shap_values = np.asarray([get_shap_attributions(row.tolist())["shap_values"] for row in sample], dtype=float)
    shap_importance = np.abs(shap_values).mean(axis=0)
    result = {
        "dataset": output_dir.name,
        "serving_model": "FakeNewsNet GAT",
        "threshold": threshold,
        "full_feature_f1": baseline_f1,
        "ablations": ablations,
        "shap_mean_abs_importance": dict(zip(FEATURE_NAMES, shap_importance.tolist())),
        "shap_sample_size": int(len(sample)),
    }
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run serving-model feature ablation")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--backend-dir", type=Path, default=Path(__file__).resolve().parents[3] / "backend")
    args = parser.parse_args()
    print(json.dumps(run_ablation(args.output_dir, args.backend_dir), indent=2))
