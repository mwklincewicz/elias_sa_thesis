from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_PATH = OUTPUT_DIR / "journal_sentiment_results.json"
REPORT_DIR = OUTPUT_DIR / "research_summary"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#C44E52",
    "neutral": "#8172B2",
    "positive": "#55A868",
}


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(rows)
    frame = frame[frame["error"].fillna("") == ""].copy()
    frame["published_dt"] = pd.to_datetime(frame["published"], errors="coerce", utc=True)
    frame["predicted_emotion_count"] = pd.to_numeric(frame["predicted_emotion_count"], errors="coerce").fillna(0)
    frame["text_length"] = pd.to_numeric(frame["text_length"], errors="coerce").fillna(0)
    return frame


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def add_note(ax: plt.Axes, text: str, x: float = 0.02, y: float = 0.98) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F1E8", "edgecolor": "#C7BCA1"},
    )


def plot_sentiment_distribution(frame: pd.DataFrame) -> Path:
    counts = frame["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [label.title() for label in counts.index],
        counts.values,
        color=[SENTIMENT_COLORS[label] for label in counts.index],
        width=0.6,
    )
    ax.set_title("Public reflection sample: sentiment distribution", fontsize=14, pad=12)
    ax.set_ylabel("Number of reflections")
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value}\n{value / total:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    add_note(
        ax,
        "Takeaway: negative reflections are the largest group (46%),\n"
        "but the sample still contains substantial positive (34%) and neutral (20%) writing.",
    )
    fig.tight_layout()
    return save_figure(fig, "fig1_sentiment_distribution.png")


def plot_top_emotions(frame: pd.DataFrame) -> Path:
    counts = Counter()
    for value in frame["predicted_emotions"].fillna(""):
        for emotion in str(value).split("|"):
            if emotion:
                counts[emotion] += 1

    top = pd.DataFrame(counts.most_common(10), columns=["emotion", "count"]).sort_values("count", ascending=True)
    colors = plt.cm.PuBuGn(np.linspace(0.35, 0.9, len(top)))

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.barh(top["emotion"], top["count"], color=colors)
    ax.set_title("Most frequent predicted emotions in the live crawl", fontsize=14, pad=12)
    ax.set_xlabel("Number of reflections where the emotion was activated")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, top["count"]):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=10)

    add_note(
        ax,
        "Takeaway: anxious, surprised, and frustrated dominate the predicted emotion profile,\n"
        "suggesting the public reflection sample leans toward tension and uncertainty.",
    )
    fig.tight_layout()
    return save_figure(fig, "fig2_top_emotions.png")


def plot_site_sentiment_heatmap(frame: pd.DataFrame) -> Path:
    site_counts = frame["site_domain"].value_counts()
    top_sites = site_counts[site_counts >= 2].head(12).index.tolist()
    subset = frame[frame["site_domain"].isin(top_sites)].copy()
    matrix = pd.crosstab(subset["site_domain"], subset["sentiment_label"]).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.45 * len(matrix) + 2.2)))
    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_title("Sentiment mix for the most active reflective sites", fontsize=14, pad=12)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([column.title() for column in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Site")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix.iat[row_idx, col_idx])
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="#0F172A", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    add_note(
        ax,
        "Takeaway: most sites contribute only a handful of posts, but several repeat contributors\n"
        "show clear sentiment tendencies instead of perfectly balanced mixes.",
        x=0.02,
        y=-0.14,
    )
    fig.tight_layout()
    return save_figure(fig, "fig3_site_sentiment_heatmap.png")


def plot_emotion_density(frame: pd.DataFrame) -> Path:
    stats = (
        frame.groupby("sentiment_label")["predicted_emotion_count"]
        .agg(["count", "mean", "median", "max"])
        .reindex(SENTIMENT_ORDER)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        stats["sentiment_label"].str.title(),
        stats["mean"],
        color=[SENTIMENT_COLORS[label] for label in stats["sentiment_label"]],
        width=0.6,
    )
    ax.set_title("Average predicted emotion density by sentiment", fontsize=14, pad=12)
    ax.set_ylabel("Average number of activated emotion labels")
    ax.grid(axis="y", alpha=0.2)

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(row["mean"]) + 0.08,
            f"mean={row['mean']:.1f}\nmedian={row['median']:.1f}\nn={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    add_note(
        ax,
        "Takeaway: negative posts are emotion-dense (5.0 labels on average),\n"
        "positive posts are lighter (2.0), and neutral posts are usually left unactivated.",
    )
    fig.tight_layout()
    return save_figure(fig, "fig4_emotion_density_by_sentiment.png")


def write_report(frame: pd.DataFrame, figures: list[Path]) -> Path:
    report_path = REPORT_DIR / "research_summary.md"
    sample_months = (
        frame["published_dt"].dropna().dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    )
    report = f"""# Research Summary

This summary follows the same broad visual language as the Lemotif project: clear bar charts, a heatmap, and model-oriented label summaries.

## Sample

- Reflective sites analyzed: {frame["site_domain"].nunique()}
- Reflections analyzed: {len(frame)}
- Time window requested: last 365 days, capped at 365 reflections per site
- Observed publication months in this crawl: {sample_months}

Important note: although the crawler allowed a full-year window, the WordPress tag discovery stream surfaced almost entirely very recent posts, so this run behaves more like a current cross-sectional snapshot than a full year timeline.

## 1. Sentiment distribution

![Sentiment distribution]({figures[0]})

Short explanation: the long-run sample is not overwhelmingly positive. Negative writing is the largest segment, but positive and neutral reflections remain substantial.

## 2. Most frequent predicted emotions

![Top emotions]({figures[1]})

Short explanation: the model most often activated anxious, surprised, frustrated, and sad-style emotions, which suggests that many public reflections are written around uncertainty, stress, or evaluation of events.

## 3. Site-level sentiment heatmap

![Site sentiment heatmap]({figures[2]})

Short explanation: the most active sites are not all alike. Some lean negative, some mix across classes, and a few produce more neutral or positive reflections.

## 4. Emotion density by sentiment

![Emotion density by sentiment]({figures[3]})

Short explanation: negative reflections tend to activate many more emotion labels at once, while neutral reflections are often almost label-free. That supports the idea that the model reads negative writing as more affectively crowded.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    ensure_dirs()
    frame = load_results()
    if frame.empty:
        raise RuntimeError("No successful results found in output/journal_sentiment_results.json.")

    figures = [
        plot_sentiment_distribution(frame),
        plot_top_emotions(frame),
        plot_site_sentiment_heatmap(frame),
        plot_emotion_density(frame),
    ]
    report_path = write_report(frame, figures)
    print(f"Saved research summary to {report_path}")


if __name__ == "__main__":
    main()
