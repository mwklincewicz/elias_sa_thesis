from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd
import seaborn as sns

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BERT_OUTPUT_DIR

MODEL_OUTPUT_DIR = Path(
    os.getenv("LEMOTIF_MODEL_OUTPUT_DIR", str(BERT_OUTPUT_DIR))
).resolve()
FIGURES_DIR = Path(
    os.getenv("LEMOTIF_MODEL_FIGURES_DIR", str(MODEL_OUTPUT_DIR / "figures"))
).resolve()

METRICS_PATH = MODEL_OUTPUT_DIR / "metrics.json"
HISTORY_PATH = MODEL_OUTPUT_DIR / "training_history.csv"
LABEL_DISTRIBUTION_PATH = MODEL_OUTPUT_DIR / "label_distribution.csv"
TEST_PREDICTIONS_PATH = MODEL_OUTPUT_DIR / "test_predictions.csv"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_training_history() -> pd.DataFrame:
    return pd.read_csv(HISTORY_PATH)


def load_label_distribution() -> pd.DataFrame:
    return pd.read_csv(LABEL_DISTRIBUTION_PATH)


def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PREDICTIONS_PATH).fillna("")


def load_per_label_metrics(split_key: str = "test_metrics") -> pd.DataFrame:
    metrics = load_metrics()
    label_order = metrics["label_names"]
    frame = pd.DataFrame(metrics[split_key]["label_metrics"])
    frame["label"] = pd.Categorical(frame["label"], categories=label_order, ordered=True)
    return frame.sort_values("label").reset_index(drop=True)


def count_label_string(value: str) -> int:
    if not isinstance(value, str):
        return 0

    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "none":
        return 0

    return len([part.strip() for part in cleaned.split(",") if part.strip()])


def parse_label_set(value: str) -> set[str]:
    if not isinstance(value, str):
        return set()

    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "none":
        return set()

    return {part.strip() for part in cleaned.split(",") if part.strip()}


def truncate_text(text: str, width: int = 120) -> str:
    cleaned = " ".join(str(text).split())
    return textwrap.shorten(cleaned, width=width, placeholder="...")


def wrap_text(text: str, width: int = 34) -> str:
    return textwrap.fill(str(text), width=width)


def save_figure(fig, filename: str, dpi: int = 220) -> Path:
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out
