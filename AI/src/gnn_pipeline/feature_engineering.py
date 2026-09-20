from __future__ import annotations

import logging
from collections import deque
from typing import List

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _reduce_dense_dimensions(arr: np.ndarray, output_dim: int) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] <= output_dim:
        return arr.astype(np.float32)

    max_components = min(output_dim, arr.shape[0] - 1, arr.shape[1] - 1)
    if max_components < 2:
        return arr.astype(np.float32)

    reducer = TruncatedSVD(n_components=max_components, random_state=42)
    return reducer.fit_transform(arr).astype(np.float32)


def build_text_embeddings(
    texts: List[str],
    device: torch.device,
    logger: logging.Logger,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    logger.info("Loading transformer model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding nodes"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            model_out = model(**encoded)
            pooled = _mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
            outputs.append(pooled.detach().cpu())

    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def build_sentence_transformer_embeddings(
    texts: List[str],
    device: torch.device,
    logger: logging.Logger,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    output_dim: int = 256,
) -> np.ndarray:
    if SentenceTransformer is None:
        logger.warning("sentence-transformers not installed; falling back to hashing embeddings")
        return build_text_hash_embeddings(texts=texts, n_features=output_dim)

    logger.info("Loading sentence-transformer model: %s", model_name)
    st_model = SentenceTransformer(model_name, device=str(device))
    emb = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    emb = _reduce_dense_dimensions(emb, output_dim)
    return emb


def build_text_hash_embeddings(
    texts: List[str],
    n_features: int = 256,
) -> np.ndarray:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    x = vectorizer.transform(texts)
    return x.astype(np.float32).toarray()


def _build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for s, d in zip(src.tolist(), dst.tolist()):
        if s == d:
            continue
        neighbors[s].add(d)
        neighbors[d].add(s)
    return neighbors


def _build_structural_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    timestamps: List[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    out_deg = np.bincount(src, minlength=num_nodes)
    in_deg = np.bincount(dst, minlength=num_nodes)
    degree = (out_deg + in_deg).astype(np.float32)

    degree_norm = degree.copy()
    if degree_norm.max() > 0:
        degree_norm = degree_norm / degree_norm.max()

    denom = max(1, num_nodes - 1)
    degree_centrality = (degree / float(denom)).astype(np.float32)
    degree_centrality = np.clip(degree_centrality, 0.0, 1.0)

    neighbors = _build_adjacency(edge_index, num_nodes)

    clustering = np.zeros((num_nodes,), dtype=np.float32)
    max_neighbors = 48
    for i in range(num_nodes):
        nbrs = list(neighbors[i])
        k = len(nbrs)
        if k < 2:
            continue
        if k > max_neighbors:
            nbrs = nbrs[:max_neighbors]
            k = len(nbrs)
        links = 0
        for u_idx in range(k):
            u = nbrs[u_idx]
            u_neighbors = neighbors[u]
            for v_idx in range(u_idx + 1, k):
                if nbrs[v_idx] in u_neighbors:
                    links += 1
        denom_pairs = k * (k - 1) / 2.0
        clustering[i] = float(links / denom_pairs) if denom_pairs > 0 else 0.0

    arr = np.array(timestamps, dtype=np.float64)
    valid_mask = np.isfinite(arr)
    propagation_depth = np.zeros((num_nodes,), dtype=np.float32)
    if valid_mask.any():
        min_t = np.nanmin(arr)
        roots = np.where(arr == min_t)[0].tolist()
        if roots:
            dist = np.full((num_nodes,), -1, dtype=np.int32)
            q = deque(roots)
            for r in roots:
                dist[r] = 0
            while q:
                cur = q.popleft()
                for nxt in neighbors[cur]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            known = dist >= 0
            if known.any() and dist[known].max() > 0:
                propagation_depth[known] = (dist[known] / dist[known].max()).astype(np.float32)

    return degree_norm, degree_centrality, clustering, propagation_depth


def _build_time_features(timestamps: List[float]) -> np.ndarray:
    arr = np.array(timestamps, dtype=np.float64)
    valid_mask = np.isfinite(arr)
    out = np.zeros((len(timestamps), 3), dtype=np.float32)
    if not valid_mask.any():
        return out

    valid_vals = arr[valid_mask]
    min_t = valid_vals.min()
    max_t = valid_vals.max()
    denom = (max_t - min_t) if (max_t - min_t) > 0 else 1.0

    out[valid_mask, 0] = ((arr[valid_mask] - min_t) / denom).astype(np.float32)
    out[valid_mask, 1] = 1.0

    sorted_valid = np.where(valid_mask)[0][np.argsort(arr[valid_mask])]
    if len(sorted_valid) > 1:
        delta = np.diff(arr[sorted_valid], prepend=arr[sorted_valid][0])
        delta = np.clip(delta, a_min=0.0, a_max=None)
        max_delta = delta.max() if delta.max() > 0 else 1.0
        out[sorted_valid, 2] = (delta / max_delta).astype(np.float32)

    return out


def _build_bot_probability(degree: np.ndarray, node_types: List[int]) -> np.ndarray:
    out = np.zeros((len(node_types),), dtype=np.float32)
    user_idx = np.array([i for i, t in enumerate(node_types) if t == 1], dtype=np.int64)
    if len(user_idx) == 0:
        return out

    deg_users = degree[user_idx]
    mean = deg_users.mean()
    std = deg_users.std() if deg_users.std() > 0 else 1.0
    z = (deg_users - mean) / std
    probs = 1.0 / (1.0 + np.exp(-z))
    out[user_idx] = probs.astype(np.float32)
    return out


def build_backend_tabular_features(
    edge_index: torch.Tensor,
    timestamps: List[float],
    node_types: List[int],
) -> np.ndarray:
    degree, _, clustering, propagation_depth = _build_structural_features(
        edge_index=edge_index,
        num_nodes=len(timestamps),
        timestamps=timestamps,
    )
    time_feats = _build_time_features(timestamps)
    bot_prob = _build_bot_probability(degree=degree, node_types=node_types)
    velocity = time_feats[:, 0].astype(np.float32)
    return np.stack([velocity, bot_prob, clustering, propagation_depth], axis=1).astype(np.float32)


def build_node_features(
    edge_index: torch.Tensor,
    texts: List[str],
    timestamps: List[float],
    node_types: List[int],
    user_feature_matrix: np.ndarray | None,
    device: torch.device,
    logger: logging.Logger,
    embedding_backend: str = "hashing",
    text_feature_dim: int = 256,
) -> torch.Tensor:
    backend = embedding_backend.strip().lower()
    if backend == "transformer":
        emb = build_text_embeddings(texts=texts, device=device, logger=logger)
        emb = _reduce_dense_dimensions(emb, text_feature_dim)
    elif backend in {"sentence-transformer", "sentence_transformer", "sbert"}:
        emb = build_sentence_transformer_embeddings(
            texts=texts,
            device=device,
            logger=logger,
            output_dim=text_feature_dim,
        )
    else:
        logger.info("Using hashing text features with %d dimensions", text_feature_dim)
        emb = build_text_hash_embeddings(texts=texts, n_features=text_feature_dim)

    degree, degree_centrality, clustering, propagation_depth = _build_structural_features(
        edge_index=edge_index,
        num_nodes=len(texts),
        timestamps=timestamps,
    )
    time_feats = _build_time_features(timestamps)
    bot_prob = _build_bot_probability(degree=degree, node_types=node_types).reshape(-1, 1)

    struct = np.stack([degree, degree_centrality, clustering, propagation_depth], axis=1).astype(np.float32)

    parts = [emb.astype(np.float32), struct, time_feats, bot_prob]
    if user_feature_matrix is not None and user_feature_matrix.shape[0] == len(texts):
        parts.append(user_feature_matrix.astype(np.float32))

    x = np.concatenate(parts, axis=1)
    return torch.tensor(x, dtype=torch.float32)
