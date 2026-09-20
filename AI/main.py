from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    with open(output_dir / "dataset_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry.to_dict(), f, indent=2)

    logger.info("Step 2: Data loading")
    fakenews_df, fakenews_user_news_edges, fakenews_user_user_edges, user_feature_dict = load_fakenewsnet(registry, logger)
    liar_df = load_liar(registry, logger)
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
    model = GraphGATClassifier(in_channels=graph.x.size(1), hidden_channels=192, heads=8, dropout=0.4)
    metrics = train_and_evaluate(
        model=model,
        graph_data=graph,
        labels=graph.y,
        output_dir=output_dir,
        logger=logger,
        device=device,
        backend_features=backend_features,
        backend_model_path=project_root.parent / "backend" / "model" / "model.pkl",
    )

    logger.info("Final metrics: %s", metrics)
    logger.info("Pipeline complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Misinformation GNN pipeline")
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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    run_pipeline(
        dataset_root=args.dataset_root or (project_root / "dataset"),
        output_dir=args.output_dir or (project_root / "outputs"),
        use_cuda=args.use_cuda,
        embedding_backend=args.embedding_backend,
        text_feature_dim=args.text_feature_dim,
    )
