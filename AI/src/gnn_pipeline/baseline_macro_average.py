from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_gcn_macro_average(outputs_dir: Path, dataset_names: list[str]) -> dict[str, object]:
    datasets = []
    for name in dataset_names:
        metrics = json.loads((outputs_dir / name / "gcn_training" / "metrics.json").read_text(encoding="utf-8"))
        datasets.append({"dataset": name, "accuracy": metrics["accuracy"], "f1": metrics["f1"]})

    result = {
        "accuracy": sum(d["accuracy"] for d in datasets) / len(datasets),
        "f1": sum(d["f1"] for d in datasets) / len(datasets),
        "datasets": datasets,
    }
    (outputs_dir / "gcn_macro_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compute_lr_macro_average(outputs_dir: Path, dataset_names: list[str]) -> dict[str, object]:
    datasets = []
    for name in dataset_names:
        metrics = json.loads((outputs_dir / name / "metrics.json").read_text(encoding="utf-8"))
        f1 = metrics["artifacts"]["backend_test_f1"]
        datasets.append({"dataset": name, "f1": f1, "source": "per-dataset train_and_evaluate LR surrogate"})

    result = {
        "datasets": datasets,
        "averaging": "unweighted mean across datasets",
        "macro_average": {"f1": sum(d["f1"] for d in datasets) / len(datasets)},
    }
    (outputs_dir / "lr_macro_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute unweighted macro-average metrics for the GCN and LR baselines")
    parser.add_argument("--outputs-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs")
    parser.add_argument("dataset_names", nargs="+", help="Dataset output directory names")
    args = parser.parse_args()

    print(json.dumps({
        "gcn": compute_gcn_macro_average(args.outputs_dir, args.dataset_names),
        "lr": compute_lr_macro_average(args.outputs_dir, args.dataset_names),
    }, indent=2))
