from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import seaborn as sns

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import GROUPED_EDA_OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from data_loader import load_analysis_data, load_lemotif_data

sns.set_theme(style="whitegrid")

OUTPUT_DIR = Path(os.getenv("LEMOTIF_GROUPED_ANALYSIS_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_DEFINITIONS = [
    {
        "key": "positive_contentment",
        "display": "Positive Contentment",
        "kind": "positive",
        "members": ["calm", "satisfied"],
    },
    {
        "key": "positive_activation",
        "display": "Positive Activation",
        "kind": "positive",
        "members": ["excited", "happy", "proud"],
    },
    {
        "key": "distress_activation",
        "display": "Distress Activation",
        "kind": "negative",
        "members": ["afraid", "angry", "anxious", "disgusted", "frustrated"],
    },
    {
        "key": "low_mood",
        "display": "Low Mood",
        "kind": "negative",
        "members": ["bored", "sad"],
    },
    {
        "key": "self_conscious_negative",
        "display": "Self-Conscious Negative",
        "kind": "negative",
        "members": ["ashamed", "awkward", "jealous"],
    },
    {
        "key": "reflective_ambiguous",
        "display": "Reflective / Ambiguous",
        "kind": "neutral",
        "members": ["confused", "nostalgic", "surprised"],
    },
]

GROUP_LOOKUP = {group["key"]: group for group in GROUP_DEFINITIONS}
GROUP_ORDER = [group["key"] for group in GROUP_DEFINITIONS]
GROUP_DISPLAY_ORDER = [group["display"] for group in GROUP_DEFINITIONS]
POSITIVE_GROUP_KEYS = [group["key"] for group in GROUP_DEFINITIONS if group["kind"] == "positive"]
NEGATIVE_GROUP_KEYS = [group["key"] for group in GROUP_DEFINITIONS if group["kind"] == "negative"]


def load_dataset() -> tuple[pd.DataFrame, str, list[str], list[str]]:
    custom_dataset = os.getenv("LEMOTIF_GROUPED_ANALYSIS_DATASET_PATH") or os.getenv(
        "LEMOTIF_ANALYSIS_DATASET_PATH"
    )
    if custom_dataset:
        return load_lemotif_data(Path(custom_dataset))
    return load_analysis_data(prefer_cleaned=True)


def emotion_key_from_column(column: str) -> str:
    return column.replace("Answer.f1.", "").replace(".raw", "")


def topic_key_from_column(column: str) -> str:
    return column.replace("Answer.t1.", "").replace(".raw", "")


def topic_display_name(column: str) -> str:
    return topic_key_from_column(column).replace("_", " ").title()


def resolve_group_columns(emotion_cols: list[str]) -> dict[str, list[str]]:
    emotion_lookup = {emotion_key_from_column(column): column for column in emotion_cols}
    resolved: dict[str, list[str]] = {}

    for group in GROUP_DEFINITIONS:
        columns = [emotion_lookup[member] for member in group["members"] if member in emotion_lookup]
        if not columns:
            raise ValueError(f"Could not resolve any emotion columns for group '{group['key']}'.")
        resolved[group["key"]] = columns

    return resolved


def build_group_frame(df: pd.DataFrame, emotion_cols: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    group_columns = resolve_group_columns(emotion_cols)
    frame = pd.DataFrame(index=df.index)

    for group in GROUP_DEFINITIONS:
        key = group["key"]
        columns = group_columns[key]
        frame[f"{key}_count"] = df[columns].sum(axis=1).astype(int)
        frame[key] = (frame[f"{key}_count"] > 0).astype(int)

    positive_label_columns = [
        column for key in POSITIVE_GROUP_KEYS for column in group_columns[key]
    ]
    frame["positive_label_count"] = df[positive_label_columns].sum(axis=1).astype(int)
    frame["positive_group_count"] = frame[POSITIVE_GROUP_KEYS].sum(axis=1).astype(int)
    frame["negative_group_count"] = frame[NEGATIVE_GROUP_KEYS].sum(axis=1).astype(int)
    frame["any_positive_group"] = (frame["positive_group_count"] > 0).astype(int)
    frame["any_negative_group"] = (frame["negative_group_count"] > 0).astype(int)
    frame["mixed_affect"] = (
        (frame["positive_group_count"] > 0) & (frame["negative_group_count"] > 0)
    ).astype(int)
    frame["group_count"] = frame[GROUP_ORDER].sum(axis=1).astype(int)
    return frame, group_columns


def group_display_name(key: str) -> str:
    return GROUP_LOOKUP[key]["display"]


def make_group_summary(
    df: pd.DataFrame,
    group_frame: pd.DataFrame,
    group_columns: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    total_rows = len(df)

    for group in GROUP_DEFINITIONS:
        key = group["key"]
        rows.append(
            {
                "group_key": key,
                "group_label": group["display"],
                "group_kind": group["kind"],
                "member_emotions": ", ".join(group["members"]),
                "member_count": len(group_columns[key]),
                "positive_rows": int(group_frame[key].sum()),
                "prevalence": float(group_frame[key].mean()),
                "avg_member_labels_per_entry": float(group_frame[f"{key}_count"].mean()),
                "avg_member_labels_when_active": float(
                    group_frame.loc[group_frame[key] == 1, f"{key}_count"].mean()
                ),
                "dataset_rows": total_rows,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["group_kind", "positive_rows"],
        ascending=[True, False],
    )
    return summary.reset_index(drop=True)


def build_other_positive_label_count(
    df: pd.DataFrame,
    target_group_key: str,
    group_columns: dict[str, list[str]],
) -> pd.Series:
    positive_columns = [column for key in POSITIVE_GROUP_KEYS for column in group_columns[key]]
    target_positive_columns = group_columns[target_group_key] if target_group_key in POSITIVE_GROUP_KEYS else []
    remaining_columns = [column for column in positive_columns if column not in target_positive_columns]
    if not remaining_columns:
        return pd.Series(0, index=df.index, dtype=int)
    return df[remaining_columns].sum(axis=1).astype(int)
