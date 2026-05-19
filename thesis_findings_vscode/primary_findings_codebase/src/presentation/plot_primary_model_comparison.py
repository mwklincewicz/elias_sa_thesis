from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
TEST_DATA_DIR = ROOT / "test data"
ASSET_DIR = OUTPUT_DIR / "presentation_assets"

LEMOTIF_METRICS_PATH = OUTPUT_DIR / "bert_emotion_model" / "metrics.json"
EXTERNAL_METRICS_PATH = TEST_DATA_DIR / "output" / "external_evaluation" / "metrics.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"Missing required metrics file: {path}\n"
            "Run `python run_primary_findings.py --full` or copy the metrics artifact into place."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_training_comparison_chart(lemotif_metrics: dict, external_metrics: dict) -> None:
    rows = [
        {
            "metric": "Emotion micro F1",
            "Lemotif split": lemotif_metrics["test_metrics"]["micro"]["f1"],
            "Stories external": external_metrics["test_metrics"]["micro"]["f1"],
        },
        {
            "metric": "Emotion macro F1",
            "Lemotif split": lemotif_metrics["test_metrics"]["macro"]["f1"],
            "Stories external": external_metrics["test_metrics"]["macro"]["f1"],
        },
        {
            "metric": "Sentiment accuracy",
            "Lemotif split": lemotif_metrics["test_sentiment_metrics"]["accuracy"],
            "Stories external": external_metrics["test_sentiment_metrics"]["accuracy"],
        },
        {
            "metric": "Sentiment macro F1",
            "Lemotif split": lemotif_metrics["test_sentiment_metrics"]["macro"]["f1"],
            "Stories external": external_metrics["test_sentiment_metrics"]["macro"]["f1"],
        },
    ]
    frame = pd.DataFrame(rows).melt(id_vars="metric", var_name="run", value_name="score")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.barplot(data=frame, x="metric", y="score", hue="run", palette=["#2F5597", "#5B9BD5"], ax=ax)
    ax.set_title("Training run comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=14)
    ax.legend(title="")

    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.annotate(
            f"{height:.3f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    fig.tight_layout()
    save_figure(fig, ASSET_DIR / "training_run_comparison.png")


def create_external_per_emotion_chart(external_metrics: dict) -> None:
    frame = pd.DataFrame(external_metrics["test_metrics"]["label_metrics"])
    frame = frame.sort_values("f1", ascending=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("crest", n_colors=len(frame))
    ax.barh(frame["label"], frame["f1"], color=colors)
    ax.set_title("Stories external test: per-emotion F1")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.6, float(frame["f1"].max()) + 0.08))

    for index, row in frame.iterrows():
        ax.text(
            float(row["f1"]) + 0.01,
            frame.index.get_loc(index),
            f"n={int(row['support'])}",
            va="center",
            fontsize=9,
            color="#282828",
        )

    fig.tight_layout()
    save_figure(fig, ASSET_DIR / "stories_external_per_emotion_f1.png")


def main() -> None:
    lemotif_metrics = load_json(LEMOTIF_METRICS_PATH)
    external_metrics = load_json(EXTERNAL_METRICS_PATH)
    create_training_comparison_chart(lemotif_metrics, external_metrics)
    create_external_per_emotion_chart(external_metrics)
    print(f"Saved primary comparison figures to: {ASSET_DIR}")


if __name__ == "__main__":
    main()
