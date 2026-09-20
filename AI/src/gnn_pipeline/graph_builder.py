from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph(
    news_df: pd.DataFrame,
    user_news_edges: List[Tuple[str, str]],
    user_user_edges: List[Tuple[str, str]],
) -> tuple[Data, dict[str, int], list[str], list[float], list[float], list[int]]:
    news_ids = news_df["id"].astype(str).tolist()
    user_ids = sorted(
        {
            uid
            for uid, _ in user_news_edges
        }
        | {u for u, _ in user_user_edges}
        | {v for _, v in user_user_edges}
    )

    node_ids = news_ids + user_ids
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    text_map = dict(zip(news_df["id"].astype(str), news_df["text"].astype(str)))
    label_map = dict(zip(news_df["id"].astype(str), news_df["label"].astype(float)))

    timestamp_map = {}
    for nid, ts in zip(news_df["id"].astype(str), news_df["timestamp"]):
        try:
            timestamp_map[nid] = pd.to_datetime(ts).timestamp() if pd.notna(ts) else np.nan
        except Exception:
            timestamp_map[nid] = np.nan

    texts = [text_map.get(nid, "") for nid in node_ids]
    labels = [label_map.get(nid, -1.0) for nid in node_ids]
    timestamps = [timestamp_map.get(nid, np.nan) for nid in node_ids]
    node_types = [0 if nid in text_map else 1 for nid in node_ids]

    edges: List[Tuple[int, int]] = []

    for src_user, dst_news in user_news_edges:
        if src_user in node_to_idx and dst_news in node_to_idx:
            s = node_to_idx[src_user]
            d = node_to_idx[dst_news]
            edges.append((s, d))
            edges.append((d, s))

    for src_user, dst_user in user_user_edges:
        if src_user in node_to_idx and dst_user in node_to_idx:
            s = node_to_idx[src_user]
            d = node_to_idx[dst_user]
            edges.append((s, d))
            edges.append((d, s))

    for _, row in news_df.iterrows():
        parent = row.get("parent_id")
        child = row.get("id")
        if pd.notna(parent) and str(parent) in node_to_idx and str(child) in node_to_idx:
            p = node_to_idx[str(parent)]
            c = node_to_idx[str(child)]
            edges.append((p, c))
            edges.append((c, p))

    if not edges:
        edges = [(i, i) for i in range(len(node_ids))]

    edges = list(set(edges))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(
        edge_index=edge_index,
        num_nodes=len(node_ids),
    )

    return data, node_to_idx, texts, timestamps, labels, node_types
