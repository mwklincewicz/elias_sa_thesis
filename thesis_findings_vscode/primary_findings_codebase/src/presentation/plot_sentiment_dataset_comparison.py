from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name, load_analysis_data, load_lemotif_data


STORIES_CSV_PATH = ROOT / "test data" / "stories_data.csv"
OUTPUT_DIR = ROOT / "output" / "dataset_comparison"
FIGURE_PATH = OUTPUT_DIR / "fig_sentiment_distribution_comparison.png"
SUMMARY_PATH = OUTPUT_DIR / "sentiment_distribution_summary.csv"

SENTIMENT_VALENCE = {
    "afraid": -1.0,
    "angry": -1.0,
    "anxious": -1.0,
    "ashamed": -1.0,
    "awkward": -1.0,
    "bored": -1.0,
    "calm": 1.0,
    "confused": 0.0,
    "disgusted": -1.0,
    "excited": 1.0,
    "frustrated": -1.0,
    "happy": 1.0,
    "jealous": -1.0,
    "nostalgic": 0.0,
    "proud": 1.0,
    "sad": -1.0,
    "satisfied": 1.0,
    "surprised": 0.0,
}

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#C44E52",
    "neutral": "#9E9E9E",
    "positive": "#55A868",
}


def derive_sentiment(binary_labels: pd.Series, label_names: list[str]) -> str:
    active_scores = [
        SENTIMENT_VALENCE[label_name.lower()]
        for label_name, is_active in zip(label_names, binary_labels.to_list())
        if int(is_active) == 1
    ]

    if not active_scores:
        return "neutral"

    score = float(sum(active_scores) / len(active_scores))
    if score > 0.25:
        return "positive"
    if score < -0.25:
        return "negative"
    return "neutral"


def build_sentiment_frame(dataset_name: str, df: pd.DataFrame, emotion_cols: list[str]) -> pd.DataFrame:
    label_names = [display_name(col) for col in emotion_cols]
    sentiments = df[emotion_cols].apply(derive_sentiment, axis=1, label_names=label_names)
    counts = sentiments.value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = int(counts.sum())

    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "sentiment": counts.index,
            "count": counts.values,
            "percentage": (counts.values / total) * 100.0,
            "total_rows": total,
        }
    )


def plot_sentiment_comparison(summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    datasets = summary["dataset"].drop_duplicates().tolist()
    positions = range(len(datasets))
    bottoms = [0.0] * len(datasets)

    for sentiment in SENTIMENT_ORDER:
        subset = summary[summary["sentiment"] == sentiment]
        values = subset["percentage"].tolist()
        bars = ax.bar(
            positions,
            values,
            bottom=bottoms,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
            width=0.58,
            label=sentiment.title(),
        )

        for bar, (_, row) in zip(bars, subset.iterrows()):
            if row["percentage"] < 6:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{row['percentage']:.1f}%\n(n={int(row['count'])})",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Derived sentiment distribution across datasets", fontsize=16, pad=12)
    ax.set_ylabel("Share of entries (%)")
    ax.set_xlabel("")
    ax.set_xticks(list(positions), datasets)
    ax.set_ylim(0, 100)
    ax.legend(title="Sentiment", loc="upper right")

    note = (
        "Sentiment is derived from the multi-label emotion annotations using the same "
        "positive/neutral/negative mapping as the BERT evaluation."
    )
    fig.text(0.02, 0.01, note, ha="left", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lemotif_df, _, emotion_cols, _ = load_analysis_data(prefer_cleaned=True)
    stories_df, _, stories_emotion_cols, _ = load_lemotif_data(STORIES_CSV_PATH)

    summary = pd.concat(
        [
            build_sentiment_frame("Lemotif", lemotif_df, emotion_cols),
            build_sentiment_frame("Stories", stories_df, stories_emotion_cols),
        ],
        ignore_index=True,
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_sentiment_comparison(summary)

    print(f"Saved chart to: {FIGURE_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
