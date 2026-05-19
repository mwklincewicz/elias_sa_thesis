from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name, load_analysis_data, load_lemotif_data


STORIES_CSV_PATH = ROOT / "test data" / "stories_data.csv"
OUTPUT_DIR = ROOT / "output" / "dataset_comparison"
FIGURE_PATH = OUTPUT_DIR / "fig_emotion_distribution_comparison.png"
SUMMARY_PATH = OUTPUT_DIR / "emotion_distribution_summary.csv"

LEMOTIF_COLOR = "#2F5597"
STORIES_COLOR = "#5B9BD5"


def build_emotion_summary(dataset_name: str, df: pd.DataFrame, emotion_cols: list[str]) -> pd.DataFrame:
    total_rows = len(df)
    records: list[dict[str, object]] = []

    for column in emotion_cols:
        emotion = display_name(column)
        count = int(pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int).sum())
        records.append(
            {
                "dataset": dataset_name,
                "emotion": emotion,
                "count": count,
                "percentage": (count / total_rows) * 100.0,
                "total_rows": total_rows,
            }
        )

    return pd.DataFrame.from_records(records)


def plot_emotion_comparison(summary: pd.DataFrame) -> None:
    emotion_order = (
        summary[summary["dataset"] == "Lemotif"]
        .sort_values("percentage", ascending=False)["emotion"]
        .tolist()
    )

    lemotif = (
        summary[summary["dataset"] == "Lemotif"]
        .set_index("emotion")
        .reindex(emotion_order)
        .reset_index()
    )
    stories = (
        summary[summary["dataset"] == "Stories"]
        .set_index("emotion")
        .reindex(emotion_order)
        .reset_index()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.6))

    x = np.arange(len(emotion_order))
    width = 0.38

    bars_lemotif = ax.bar(
        x - width / 2,
        lemotif["percentage"],
        width=width,
        color=LEMOTIF_COLOR,
        label="Lemotif",
    )
    bars_stories = ax.bar(
        x + width / 2,
        stories["percentage"],
        width=width,
        color=STORIES_COLOR,
        label="Stories",
    )

    ax.set_title("Emotion label prevalence across datasets", fontsize=16, pad=12)
    ax.set_ylabel("Share of entries (%)")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_order, rotation=45, ha="right")
    ax.legend(loc="upper right")

    for bars in (bars_lemotif, bars_stories):
        for bar in bars:
            value = bar.get_height()
            if value < 4:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.5,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=90,
                color="#333333",
            )

    note = (
        "Percentages show how often each emotion label appears in each dataset. "
        "The datasets share the same 18-label emotion structure."
    )
    fig.text(0.02, 0.01, note, ha="left", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lemotif_df, _, emotion_cols, _ = load_analysis_data(prefer_cleaned=True)
    stories_df, _, stories_emotion_cols, _ = load_lemotif_data(STORIES_CSV_PATH)

    summary = pd.concat(
        [
            build_emotion_summary("Lemotif", lemotif_df, emotion_cols),
            build_emotion_summary("Stories", stories_df, stories_emotion_cols),
        ],
        ignore_index=True,
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_emotion_comparison(summary)

    print(f"Saved chart to: {FIGURE_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
