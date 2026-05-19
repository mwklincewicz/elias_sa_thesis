from __future__ import annotations

from pathlib import Path
import pandas as pd

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

def load_lemotif_data(csv_path: str | Path):
    df = pd.read_csv(csv_path)
    text_col, emotion_cols, topic_cols = detect_columns(df)
    df[text_col] = df[text_col].astype(str)
    return df, text_col, emotion_cols, topic_cols
