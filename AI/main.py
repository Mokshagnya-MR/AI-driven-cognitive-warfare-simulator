from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.gnn_pipeline import (
    build_backend_tabular_features,
    GraphGATClassifier,
    build_graph,
    build_node_features,
    discover_datasets,
    load_fakenewsnet,
    load_liar,
    load_pheme,
    preprocess_news_dataframe,
    train_and_evaluate,
)
from src.gnn_pipeline.train import TrainConfig


DEFAULT_PIPELINE_CONFIG = {
    "seed": 42,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}


def load_pipeline_config(project_root: Path, config_path: Path | None = None) -> dict[str, float | int]:
    """Read AI/config.yaml (seed, split ratios) instead of relying on values hardcoded in train.py."""
    path = config_path or (project_root / "config.yaml")
    config = dict(DEFAULT_PIPELINE_CONFIG)
    if not path.exists():
        return config

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "seed" in raw:
        config["seed"] = int(raw["seed"])

    split = raw.get("split", {})
    if isinstance(split, dict):
        config["train_ratio"] = float(split.get("train", config["train_ratio"]))
        config["val_ratio"] = float(split.get("validation", config["val_ratio"]))
        config["test_ratio"] = float(split.get("test", config["test_ratio"]))

    ratio_sum = config["train_ratio"] + config["val_ratio"] + config["test_ratio"]
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"config.yaml split ratios must sum to 1.0, got {ratio_sum}")

    return config


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _configure_torch_for_device(device: torch.device) -> None:
    if device.type != "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("misinfo-gnn")


def _assemble_user_feature_matrix(
    node_to_idx: dict[str, int],
    user_feature_dict: dict[str, np.ndarray | sparse.spmatrix],
    max_output_dim: int = 128,
) -> np.ndarray | None:
    if not user_feature_dict:
        return None

    feature_items: list[tuple[int, np.ndarray | sparse.spmatrix]] = []
    for node_id, feat in user_feature_dict.items():
        idx = node_to_idx.get(node_id)
        if idx is not None:
            feature_items.append((idx, feat))

    if not feature_items:
        return None

    first_feat = feature_items[0][1]
    if sparse.issparse(first_feat):
        sparse_rows = []
        node_indices = []
        for idx, feat in feature_items:
            if sparse.issparse(feat) and feat.shape[0] == 1:
                sparse_rows.append(feat.tocsr())
                node_indices.append(idx)

        if not sparse_rows:
            return None

        feature_matrix = sparse.vstack(sparse_rows, format="csr")
        feat_dim = feature_matrix.shape[1]
        n_samples = feature_matrix.shape[0]
        if feat_dim <= 1 or n_samples <= 1:
            reduced = feature_matrix.toarray().astype(np.float32)
        else:
            output_dim = min(max_output_dim, feat_dim, n_samples)
            if output_dim >= min(feat_dim, n_samples):
                output_dim = max(1, min(feat_dim, n_samples) - 1)
            if output_dim <= 0:
                reduced = feature_matrix.toarray().astype(np.float32)
            else:
                reducer = TruncatedSVD(n_components=output_dim, random_state=42)
                reduced = reducer.fit_transform(feature_matrix).astype(np.float32)

        mat = np.zeros((len(node_to_idx), reduced.shape[1]), dtype=np.float32)
        for row_idx, node_idx in enumerate(node_indices):
            mat[node_idx] = reduced[row_idx]
        return mat

    dense_rows = []
    node_indices = []
    feat_dim = None
    for idx, feat in feature_items:
        arr = np.asarray(feat, dtype=np.float32).ravel()
        if feat_dim is None:
            feat_dim = len(arr)
        if len(arr) == feat_dim:
            dense_rows.append(arr)
            node_indices.append(idx)

    if not dense_rows or feat_dim is None:
        return None

    if feat_dim > max_output_dim:
        feature_matrix = np.stack(dense_rows, axis=0)
        reducer = TruncatedSVD(n_components=max(1, min(max_output_dim, feature_matrix.shape[0], feature_matrix.shape[1]) - 1), random_state=42)
        reduced = reducer.fit_transform(feature_matrix).astype(np.float32)
        mat = np.zeros((len(node_to_idx), reduced.shape[1]), dtype=np.float32)
        for row_idx, node_idx in enumerate(node_indices):
            mat[node_idx] = reduced[row_idx]
        return mat

    mat = np.zeros((len(node_to_idx), feat_dim), dtype=np.float32)
    for row_idx, node_idx in enumerate(node_indices):
        mat[node_idx] = dense_rows[row_idx]

    return mat



def _resolve_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_artifact_manifest(output_dir: Path, registry: dict[str, object], metrics: dict[str, object], seed: int = 42) -> None:
    """Record enough provenance to reject artifacts from an unrelated dataset run."""
    registry_json = json.dumps(registry, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema_version": 1,
        "dataset_registry_sha256": hashlib.sha256(registry_json.encode("utf-8")).hexdigest(),
        "feature_schema": ["velocity", "bot_ratio", "echo_density", "depth"],
        "split_seed": seed,
        "artifacts": metrics.get("artifacts", {}),
    }
    with open(output_dir / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_pipeline_for_dataset(
    dataset_name: str,
    dataset_root: Path,
    output_dir: Path,
    use_cuda: bool = False,
    embedding_backend: str = "sentence-transformer",
    text_feature_dim: int = 256,
    epochs: int = 60,
    early_stopping_patience: int = 10,
) -> dict[str, object]:
    """Train one isolated graph so dataset structure is not mixed with others."""
    logger = setup_logging()
    project_root = Path(__file__).resolve().parent
    pipeline_config = load_pipeline_config(project_root)
    set_seed(int(pipeline_config["seed"]))
    device = _resolve_device(use_cuda=use_cuda)
    _configure_torch_for_device(device)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    logger.info("Dataset %s | device: %s | pipeline_config: %s", dataset_name, device, pipeline_config)

    registry = discover_datasets(dataset_root)
    registry_payload = registry.to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_registry.json").write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    if dataset_name == "fakenewsnet":
        news_df, user_news_edges, user_user_edges, user_features = load_fakenewsnet(registry, logger)
    elif dataset_name == "liar":
        news_df, liar_speaker_history_features = load_liar(registry, logger)
        user_news_edges, user_user_edges, user_features = [], [], liar_speaker_history_features
    elif dataset_name == "pheme":
        news_df, user_news_edges = load_pheme(registry, logger)
        user_user_edges, user_features = [], {}
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    news_df = preprocess_news_dataframe(news_df)
    graph, node_to_idx, texts, timestamps, labels, node_types = build_graph(
        news_df=news_df,
        user_news_edges=user_news_edges,
        user_user_edges=user_user_edges,
    )
    user_feat_mat = _assemble_user_feature_matrix(node_to_idx, user_features)
    backend_features = build_backend_tabular_features(graph.edge_index, timestamps, node_types)
    np.save(output_dir / "backend_features.npy", backend_features)
    np.save(output_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    graph.x = build_node_features(
        edge_index=graph.edge_index,
        texts=texts,
        timestamps=timestamps,
        node_types=node_types,
        user_feature_matrix=user_feat_mat,
        device=device,
        logger=logger,
        embedding_backend=embedding_backend,
        text_feature_dim=text_feature_dim,
    )
    graph.y = torch.tensor(labels, dtype=torch.float32)
    torch.save({"data": graph, "node_to_idx": node_to_idx, "node_types": node_types}, output_dir / "processed_graph.pt")

    model = GraphGATClassifier(in_channels=graph.x.size(1), hidden_channels=64, heads=4, dropout=0.4)
    metrics = train_and_evaluate(
        model=model,
        graph_data=graph,
        labels=graph.y,
        output_dir=output_dir,
        logger=logger,
        device=device,
        backend_features=backend_features,
        backend_model_path=project_root.parent / "backend" / "model" / f"{dataset_name}_model.pkl",
        config=TrainConfig(
            epochs=epochs,
            batch_size=256 if device.type == "cuda" else 64,
            gradient_accumulation_steps=1,
            early_stopping_patience=early_stopping_patience,
            seed=int(pipeline_config["seed"]),
            train_ratio=float(pipeline_config["train_ratio"]),
            val_ratio=float(pipeline_config["val_ratio"]),
            test_ratio=float(pipeline_config["test_ratio"]),
        ),
    )
    _write_artifact_manifest(output_dir, registry_payload, metrics, seed=int(pipeline_config["seed"]))
    return metrics


def run_pipeline(
    dataset_root: Path,
    output_dir: Path,
    use_cuda: bool = False,
    embedding_backend: str = "sentence-transformer",
    text_feature_dim: int = 256,
) -> None:
    logger = setup_logging()
    set_seed(42)

    device = _resolve_device(use_cuda=use_cuda)
    _configure_torch_for_device(device)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    logger.info("Using device: %s", device)

    logger.info("Step 1: Data discovery")
    registry = discover_datasets(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_payload = registry.to_dict()
    with open(output_dir / "dataset_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_payload, f, indent=2)

    logger.info("Step 2: Data loading")
    fakenews_df, fakenews_user_news_edges, fakenews_user_user_edges, user_feature_dict = load_fakenewsnet(registry, logger)
    liar_df, _liar_speaker_history_features = load_liar(registry, logger)
    pheme_df, pheme_user_news_edges = load_pheme(registry, logger)

    all_df = pd.concat([fakenews_df, liar_df, pheme_df], ignore_index=True)

    logger.info("Step 3: Preprocessing")
    all_df = preprocess_news_dataframe(all_df)

    logger.info("Step 4: Graph building")
    all_user_news_edges = fakenews_user_news_edges + pheme_user_news_edges
    graph, node_to_idx, texts, timestamps, labels, node_types = build_graph(
        news_df=all_df,
        user_news_edges=all_user_news_edges,
        user_user_edges=fakenews_user_user_edges,
    )

    user_feat_mat = _assemble_user_feature_matrix(node_to_idx=node_to_idx, user_feature_dict=user_feature_dict)
    backend_features = build_backend_tabular_features(
        edge_index=graph.edge_index,
        timestamps=timestamps,
        node_types=node_types,
    )

    logger.info("Step 5: Feature engineering")
    x = build_node_features(
        edge_index=graph.edge_index,
        texts=texts,
        timestamps=timestamps,
        node_types=node_types,
        user_feature_matrix=user_feat_mat,
        device=device,
        logger=logger,
        embedding_backend=embedding_backend,
        text_feature_dim=text_feature_dim,
    )

    y = torch.tensor(labels, dtype=torch.float32)
    graph.x = x
    graph.y = y

    logger.info("Saving processed graph")
    torch.save(
        {
            "data": graph,
            "node_to_idx": node_to_idx,
            "node_types": node_types,
        },
        output_dir / "processed_graph.pt",
    )

    logger.info("Step 6-8: Model, training, and evaluation")
    training_config = TrainConfig(
        epochs=6,
        batch_size=256 if device.type == "cuda" else 64,
        gradient_accumulation_steps=1,
        early_stopping_patience=2,
    )
    model = GraphGATClassifier(
        in_channels=graph.x.size(1),
        hidden_channels=64,
        heads=4,
        dropout=0.4,
    )
    metrics = train_and_evaluate(
        model=model,
        graph_data=graph,
        labels=graph.y,
        output_dir=output_dir,
        logger=logger,
        device=device,
        backend_features=backend_features,
        backend_model_path=project_root.parent / "backend" / "model" / "model.pkl",
        config=training_config,
    )

    _write_artifact_manifest(output_dir, registry_payload, metrics)

    logger.info("Final metrics: %s", metrics)
    logger.info("Pipeline complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Misinformation GNN pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["combined", "per-dataset"],
        help="'combined' runs the legacy merged-graph pipeline; 'per-dataset' isolates one dataset's graph",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        choices=["fakenewsnet", "liar", "pheme"],
        help="Required when --mode per-dataset",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA explicitly")
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="sentence-transformer",
        choices=["hashing", "transformer", "sentence-transformer"],
        help="Text feature backend",
    )
    parser.add_argument("--text-feature-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60, help="Only used in --mode per-dataset")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Only used in --mode per-dataset")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.mode == "per-dataset":
        if args.dataset_name is None:
            parser.error("--dataset-name is required when --mode per-dataset")
        metrics = run_pipeline_for_dataset(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root or (project_root / "dataset"),
            output_dir=(args.output_dir or (project_root / "outputs")) / args.dataset_name,
            use_cuda=args.use_cuda,
            embedding_backend=args.embedding_backend,
            text_feature_dim=args.text_feature_dim,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
        )
        print(json.dumps(metrics, indent=2))
    else:
        run_pipeline(
            dataset_root=args.dataset_root or (project_root / "dataset"),
            output_dir=args.output_dir or (project_root / "outputs"),
            use_cuda=args.use_cuda,
            embedding_backend=args.embedding_backend,
            text_feature_dim=args.text_feature_dim,
        )
