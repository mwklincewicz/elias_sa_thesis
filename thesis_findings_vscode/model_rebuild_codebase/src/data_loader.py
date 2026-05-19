from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from config import CLEANED_CSV_PATH, RAW_CSV_PATH


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def detect_columns(df: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    text_col = "Answer" if "Answer" in df.columns else df.columns[0]

    emotion_cols = [c for c in df.columns if c.startswith("Answer.f1.") and c.endswith(".raw")]
    topic_cols = [c for c in df.columns if c.startswith("Answer.t1.") and c.endswith(".raw")]

    if not emotion_cols:
        emotion_cols = [c for c in df.columns if ".f1." in c]
    if not topic_cols:
        topic_cols = [c for c in df.columns if ".t1." in c]

    return text_col, emotion_cols, topic_cols


def display_name(col: str) -> str:
    x = col
    x = x.replace("Answer.f1.", "")
    x = x.replace("Answer.t1.", "")
    x = x.replace(".raw", "")
    return x.replace("_", " ").title()


def _coerce_binary_columns(df: pd.DataFrame, columns: list[str]) -> None:
    if not columns:
        return
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)


def refresh_derived_columns(
    df: pd.DataFrame,
    text_col: str,
    emotion_cols: list[str],
    topic_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    _coerce_binary_columns(df, emotion_cols)
    _coerce_binary_columns(df, topic_cols)

    df["char_count"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()
    df["emotion_count"] = df[emotion_cols].sum(axis=1)
    df["topic_count"] = df[topic_cols].sum(axis=1)
    return df


def load_lemotif_data(csv_path: str | Path) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    df = pd.read_csv(csv_path)
    text_col, emotion_cols, topic_cols = detect_columns(df)
    if not emotion_cols:
        raise ValueError("Could not detect emotion columns in the Lemotif dataset.")
    if not topic_cols:
        raise ValueError("Could not detect topic columns in the Lemotif dataset.")

    df = refresh_derived_columns(df, text_col, emotion_cols, topic_cols)
    return df, text_col, emotion_cols, topic_cols


def resolve_dataset_path(prefer_cleaned: bool = True) -> Path:
    if prefer_cleaned and CLEANED_CSV_PATH.exists():
        return CLEANED_CSV_PATH
    return RAW_CSV_PATH


def load_analysis_data(prefer_cleaned: bool = True) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    return load_lemotif_data(resolve_dataset_path(prefer_cleaned=prefer_cleaned))
