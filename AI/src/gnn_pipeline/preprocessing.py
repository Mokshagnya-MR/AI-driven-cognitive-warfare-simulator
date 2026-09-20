from __future__ import annotations

import re
import string

import pandas as pd


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


REQUIRED_COLUMNS = ["id", "parent_id", "text", "label", "timestamp"]


def clean_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = _URL_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = _WS_RE.sub(" ", text).strip()
    return text


def preprocess_news_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out["text"] = out["text"].map(clean_text)
    out = out[out["text"].notna()]
    out = out[out["text"].str.len() > 0]

    out["id"] = out["id"].astype(str)
    out["parent_id"] = out["parent_id"].where(out["parent_id"].notna(), None)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    out = out.drop_duplicates(subset=["id"])
    out = out.reset_index(drop=True)
    return out[REQUIRED_COLUMNS + [c for c in out.columns if c not in REQUIRED_COLUMNS]]
