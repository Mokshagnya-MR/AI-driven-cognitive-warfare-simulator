from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from torch_geometric.loader import DataLoader

from .evaluate import compute_extended_metrics
from .model import GraphGATClassifier
from .train import NodeSubgraphDataset


def run_report(output_dir: Path, dataset_name: str) -> dict[str, object]:
    graph_payload = torch.load(output_dir / "processed_graph.pt", map_location="cpu", weights_only=False)
    graph = graph_payload["data"]
    test_indices = np.load(output_dir / "splits" / "test_idx.npy")
    labels = graph.y.detach().cpu().numpy().astype(np.int64)
    test_labels = labels[test_indices]

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics_payload["threshold_used"])
    state_dict = torch.load(output_dir / "model_state_dict.pt", map_location="cpu", weights_only=True)
    model = GraphGATClassifier(in_channels=int(graph.x.shape[1]), hidden_channels=64, heads=4, dropout=0.4)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = NodeSubgraphDataset(graph, test_indices, test_labels, hops=1, cache_subgraphs=False)
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
            logits = model(batch.x, batch.edge_index, batch.batch)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    test_probabilities = np.asarray(probabilities, dtype=np.float32)
    report_metrics = compute_extended_metrics(test_labels, test_probabilities, threshold)
    report_metrics["split"] = "test"
    report_metrics["dataset"] = dataset_name
    report_metrics["threshold_source"] = "validation split, loaded from metrics.json"

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval_metrics.json").write_text(json.dumps(report_metrics, indent=2), encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(test_labels, test_probabilities >= threshold)
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(test_labels, test_probabilities)
    plt.tight_layout()
    plt.savefig(eval_dir / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(test_labels, test_probabilities)
    plt.tight_layout()
    plt.savefig(eval_dir / "precision_recall_curve.png", dpi=160)
    plt.close()
    return report_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the frozen test split without changing runtime metrics")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs")
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    print(json.dumps(run_report(args.output_dir, args.dataset_name), indent=2))
