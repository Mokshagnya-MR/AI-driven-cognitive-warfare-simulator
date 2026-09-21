from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def compute_macro_average(outputs_dir: Path, dataset_names: list[str]) -> dict[str, object]:
    reports = {}
    for dataset_name in dataset_names:
        report_path = outputs_dir / dataset_name / "eval" / "eval_metrics.json"
        reports[dataset_name] = json.loads(report_path.read_text(encoding="utf-8"))

    macro = {
        metric: sum(float(reports[name].get(metric, 0.0)) for name in dataset_names) / len(dataset_names)
        for metric in METRICS
    }
    result = {"datasets": reports, "macro_average": macro, "averaging": "unweighted mean across datasets"}
    (outputs_dir / "macro_eval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute unweighted per-dataset metric averages")
    parser.add_argument("--outputs-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs")
    parser.add_argument("dataset_names", nargs="+", help="Dataset output directory names")
    args = parser.parse_args()
    print(json.dumps(compute_macro_average(args.outputs_dir, args.dataset_names), indent=2))
