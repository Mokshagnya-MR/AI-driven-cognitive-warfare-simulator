from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import issparse

from .discovery import DatasetRegistry


LIAR_LABEL_MAP = {
    "true": 0,
    "mostly-true": 0,
    "half-true": 1,
    "barely-true": 1,
    "false": 1,
    "pants-fire": 1,
}

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]


def _normalize_token(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("-webpage", "")
    text = text.replace(".json", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def _as_path_list(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _infer_source_from_content_csv(path: str) -> str:
    name = Path(path).name
    if "_" in name:
        return name.split("_", 1)[0]
    return Path(path).stem


def _build_token_aliases(source: str, raw_id: str) -> Iterable[str]:
    base = _normalize_token(raw_id)
    yield base
    yield _normalize_token(f"{source}_{raw_id}")
    yield _normalize_token(f"{source}_{base}")


def _extract_mat_features(path: str, logger: logging.Logger) -> np.ndarray | sparse.spmatrix | None:
    try:
        mat = loadmat(path)
    except Exception as exc:
        logger.warning("Unable to read mat file %s: %s", path, exc)
        return None

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if issparse(value):
            if value.ndim == 2 and value.shape[0] > 0:
                return value.tocsr().astype(np.float32)
        if isinstance(value, np.ndarray):
            if value.dtype == object and value.size > 0:
                flat = value.ravel()
                for item in flat:
                    if issparse(item):
                        if item.ndim == 2 and item.shape[0] > 0:
                            return item.tocsr().astype(np.float32)
                    elif isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[0] > 0:
                        if np.issubdtype(item.dtype, np.number):
                            return item.astype(np.float32)
            if value.ndim == 2 and value.shape[0] > 0 and np.issubdtype(value.dtype, np.number):
                return value.astype(np.float32)
    logger.warning("No numeric 2D matrix found in %s", path)
    return None


def _compress_sparse_row(row: np.ndarray | "scipy.sparse.spmatrix", output_dim: int = 64) -> np.ndarray:
    if hasattr(row, "tocoo"):
        coo = row.tocoo()
        vec = np.zeros(output_dim, dtype=np.float32)
        if coo.nnz == 0:
            return vec
        cols = coo.col.astype(np.int64, copy=False)
        vals = coo.data.astype(np.float32, copy=False)
        bins = np.mod(cols * 1315423911, output_dim)
        np.add.at(vec, bins, vals)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    arr = np.asarray(row, dtype=np.float32).ravel()
    if arr.size <= output_dim:
        out = np.zeros(output_dim, dtype=np.float32)
        out[: arr.size] = arr
        return out
    out = np.zeros(output_dim, dtype=np.float32)
    step = int(np.ceil(arr.size / output_dim))
    for i in range(output_dim):
        start = i * step
        if start >= arr.size:
            break
        out[i] = float(arr[start : min(start + step, arr.size)].sum())
    norm = np.linalg.norm(out)
    if norm > 0:
        out /= norm
    return out


def load_fakenewsnet(
    registry: DatasetRegistry,
    logger: logging.Logger,
    chunksize: int = 5000,
) -> tuple[pd.DataFrame, list[tuple[str, str]], list[tuple[str, str]], dict[str, np.ndarray]]:
    rows: List[dict] = []
    user_news_edges: List[Tuple[str, str]] = []
    user_user_edges: List[Tuple[str, str]] = []
    user_features: Dict[str, np.ndarray] = {}

    content_files = [(p, 1) for p in registry.fakenews_fake_csvs] + [(p, 0) for p in registry.fakenews_real_csvs]
    token_to_news_id: Dict[str, str] = {}

    for csv_path, label in content_files:
        source = _infer_source_from_content_csv(csv_path)
        logger.info("Loading FakeNewsNet content: %s", csv_path)
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunk = chunk[[c for c in ["id", "title", "text", "publish_date"] if c in chunk.columns]].copy()
            if "id" not in chunk.columns:
                continue

            title_series = chunk["title"] if "title" in chunk.columns else pd.Series([""] * len(chunk))
            text_series = chunk["text"] if "text" in chunk.columns else pd.Series([""] * len(chunk))
            timestamp_series = chunk["publish_date"] if "publish_date" in chunk.columns else pd.Series([None] * len(chunk))

            for row_id, title, text, ts in zip(chunk["id"].astype(str), title_series, text_series, timestamp_series):
                global_id = f"fakenews:{source}:{row_id}"
                rows.append(
                    {
                        "id": global_id,
                        "parent_id": None,
                        "text": f"{title or ''} {text or ''}".strip(),
                        "label": int(label),
                        "timestamp": ts,
                        "dataset": "fakenewsnet",
                        "source": source,
                    }
                )
                for alias in _build_token_aliases(source, row_id):
                    token_to_news_id[alias] = global_id

    source_to_news_list: Dict[str, List[str]] = {}
    for source, news_files in registry.news_files.items():
        news_lines: List[str] = []
        for news_file in _as_path_list(news_files):
            news_lines.extend(_read_lines(news_file))
        news_lines = list(dict.fromkeys(news_lines))
        source_to_news_list[source] = news_lines
        for item in news_lines:
            tok = _normalize_token(item)
            if tok not in token_to_news_id:
                guessed_label = 1 if "fake" in tok else 0
                synth_id = f"fakenews:{source}:{item}"
                token_to_news_id[tok] = synth_id
                rows.append(
                    {
                        "id": synth_id,
                        "parent_id": None,
                        "text": "",
                        "label": guessed_label,
                        "timestamp": None,
                        "dataset": "fakenewsnet",
                        "source": source,
                    }
                )

    source_to_user_list: Dict[str, List[str]] = {}
    for source, user_files in registry.user_files.items():
        user_lines: List[str] = []
        for user_file in _as_path_list(user_files):
            user_lines.extend(_read_lines(user_file))
        source_to_user_list[source] = list(dict.fromkeys(user_lines))

    for source, edge_files in registry.news_user_files.items():
        news_names = source_to_news_list.get(source, [])
        user_names = source_to_user_list.get(source, [])
        for edge_file in _as_path_list(edge_files):
            logger.info("Loading FakeNewsNet news-user edges: %s", edge_file)
            if not news_names or not user_names:
                logger.warning("Missing news/user lists for source %s; skipping %s", source, edge_file)
                continue

            edge_df = pd.read_csv(edge_file, sep=r"\s+", header=None, engine="python")
            if edge_df.shape[1] < 2:
                continue

            for _, row in edge_df.iterrows():
                user_idx = int(row.iloc[0])
                news_idx = int(row.iloc[1])
                if user_idx <= 0 or user_idx > len(user_names) or news_idx <= 0 or news_idx > len(news_names):
                    continue
                user_global = f"user:{source}:{user_names[user_idx - 1]}"
                news_tok = _normalize_token(news_names[news_idx - 1])
                news_global = token_to_news_id.get(news_tok)
                if news_global:
                    user_news_edges.append((user_global, news_global))

    for source, edge_files in registry.user_user_files.items():
        user_names = source_to_user_list.get(source, [])
        for edge_file in _as_path_list(edge_files):
            logger.info("Loading FakeNewsNet user-user edges: %s", edge_file)
            if not user_names:
                logger.warning("Missing user list for source %s; skipping %s", source, edge_file)
                continue

            edge_df = pd.read_csv(edge_file, sep=r"\s+", header=None, engine="python")
            if edge_df.shape[1] < 2:
                continue

            for _, row in edge_df.iterrows():
                src_idx = int(row.iloc[0])
                dst_idx = int(row.iloc[1])
                if src_idx <= 0 or src_idx > len(user_names) or dst_idx <= 0 or dst_idx > len(user_names):
                    continue
                src_user = f"user:{source}:{user_names[src_idx - 1]}"
                dst_user = f"user:{source}:{user_names[dst_idx - 1]}"
                user_user_edges.append((src_user, dst_user))

    for source, mat_paths in registry.user_feature_mats.items():
        user_names = source_to_user_list.get(source, [])
        if not user_names:
            continue
        for mat_path in _as_path_list(mat_paths):
            matrix = _extract_mat_features(mat_path, logger)
            if matrix is None:
                continue
            max_rows = min(matrix.shape[0], len(user_names))
            for idx in range(max_rows):
                user_global = f"user:{source}:{user_names[idx]}"
                user_features[user_global] = _compress_sparse_row(matrix[idx])

    df = pd.DataFrame(rows)
    return df, user_news_edges, user_user_edges, user_features


def load_liar(
    registry: DatasetRegistry,
    logger: logging.Logger,
    chunksize: int = 10000,
) -> pd.DataFrame:
    rows: List[dict] = []

    for split in ["train", "valid", "test"]:
        paths = _as_path_list(registry.liar_tsvs.get(split))
        if not paths:
            continue

        for path in paths:
            logger.info("Loading LIAR %s: %s", split, path)
            for chunk in pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS, chunksize=chunksize):
                chunk = chunk.dropna(subset=["statement", "label", "id"])
                mapped = chunk["label"].astype(str).str.strip().str.lower().map(LIAR_LABEL_MAP)
                chunk = chunk[mapped.notna()].copy()
                chunk["binary_label"] = mapped[mapped.notna()].astype(int)

                source_name = Path(path).parent.name.replace(" ", "_")
                for row_id, text, label in zip(
                    chunk["id"].astype(str),
                    chunk["statement"].astype(str),
                    chunk["binary_label"],
                ):
                    rows.append(
                        {
                            "id": f"liar:{split}:{source_name}:{row_id.replace('.json', '')}",
                            "parent_id": None,
                            "text": text,
                            "label": int(label),
                            "timestamp": None,
                            "dataset": "liar",
                            "source": source_name,
                        }
                    )

    return pd.DataFrame(rows)


def load_pheme(
    registry: DatasetRegistry,
    logger: logging.Logger,
    chunksize: int = 10000,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    rows: List[dict] = []
    user_news_edges: List[Tuple[str, str]] = []

    for csv_path in registry.pheme_candidates:
        logger.info("Inspecting PHEME candidate: %s", csv_path)
        try:
            preview = pd.read_csv(csv_path, nrows=3)
        except Exception as exc:
            logger.warning("Skipping unreadable csv %s: %s", csv_path, exc)
            continue

        columns = {c.lower() for c in preview.columns}
        if "text" not in columns or "is_rumor" not in columns:
            continue

        topic_last_id: Dict[str, str] = {}
        stem = Path(csv_path).stem
        offset = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunk_cols = {c.lower(): c for c in chunk.columns}
            text_col = chunk_cols.get("text")
            label_col = chunk_cols.get("is_rumor")
            user_col = chunk_cols.get("user.handle") or chunk_cols.get("user")
            topic_col = chunk_cols.get("topic")
            ts_col = chunk_cols.get("timestamp") or chunk_cols.get("created_at")

            if text_col is None or label_col is None:
                continue

            chunk = chunk.dropna(subset=[text_col, label_col])
            for _, row in chunk.iterrows():
                text = str(row[text_col])
                label = int(row[label_col])
                topic = str(row[topic_col]) if topic_col else "unknown"
                ts = row[ts_col] if ts_col else None

                node_id = f"pheme:{stem}:{offset}"
                parent_id = topic_last_id.get(topic)
                topic_last_id[topic] = node_id

                rows.append(
                    {
                        "id": node_id,
                        "parent_id": parent_id,
                        "text": text,
                        "label": label,
                        "timestamp": ts,
                        "dataset": "pheme",
                        "source": stem,
                        "topic": topic,
                    }
                )

                if user_col and pd.notna(row[user_col]):
                    user_handle = str(row[user_col]).strip()
                    if user_handle:
                        user_id = f"user:pheme:{user_handle}"
                        user_news_edges.append((user_id, node_id))

                offset += 1

    return pd.DataFrame(rows), user_news_edges
