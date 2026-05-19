from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_DIR = OUTPUT_DIR / "findings_report"
RESULTS_PATH = OUTPUT_DIR / "journal_sentiment_results.json"


def resolve_metrics_path() -> Path:
    candidates = [
        WORKSPACE_ROOT / "primary_findings_codebase" / "output" / "bert_emotion_model" / "metrics.json",
        WORKSPACE_ROOT / "model_rebuild_codebase" / "output" / "bert_emotion_model" / "metrics.json",
        WORKSPACE_ROOT.parent / "output" / "bert_emotion_model" / "metrics.json",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


METRICS_PATH = resolve_metrics_path()

SENTIMENT_COLORS = {
    "negative": "#C44E52",
    "neutral": "#8172B2",
    "positive": "#55A868",
}


def ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, dict]:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(results)
    frame = frame[frame["error"].fillna("") == ""].copy()
    return frame, metrics


def shorten(text: str, max_chars: int = 44) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def domain_from_url(url: str) -> str:
    return urlparse(url).netloc or url


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = REPORT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_live_sentiment_distribution(frame: pd.DataFrame) -> Path:
    counts = (
        frame["sentiment_label"]
        .value_counts()
        .reindex(["negative", "neutral", "positive"], fill_value=0)
    )
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [SENTIMENT_COLORS[label] for label in counts.index]
    bars = ax.bar(counts.index.str.title(), counts.values, color=colors, width=0.6)
    ax.set_title("Live sample sentiment distribution", fontsize=14, pad=14)
    ax.set_ylabel("Number of posts")
    ax.set_ylim(0, max(counts.max() + 0.8, 1.8))
    ax.grid(axis="y", alpha=0.2)

    for bar, count in zip(bars, counts.values):
        pct = (count / total * 100) if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{count}\n{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.96,
        f"Annotated takeaway: the first live sample is perfectly split\nacross negative, neutral, and positive ({total} posts total).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F1E8", "edgecolor": "#C7BCA1"},
    )
    fig.tight_layout()
    return save_fig(fig, "fig1_live_sentiment_distribution.png")


def plot_live_sentiment_scores(frame: pd.DataFrame) -> Path:
    ordered = frame.copy()
    ordered["display"] = ordered.apply(
        lambda row: f"{shorten(row['title'])}\n{domain_from_url(row['site'])}", axis=1
    )
    ordered = ordered.sort_values("sentiment_score")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [SENTIMENT_COLORS[label] for label in ordered["sentiment_label"]]
    bars = ax.barh(ordered["display"], ordered["sentiment_score"], color=colors)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Derived sentiment score")
    ax.set_title("Live sample sentiment score by journal post", fontsize=14, pad=14)
    ax.grid(axis="x", alpha=0.2)

    for bar, label, score in zip(bars, ordered["sentiment_label"], ordered["sentiment_score"]):
        x = score + 0.03 if score >= 0 else score - 0.03
        ha = "left" if score >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{label} ({score:.2f})", va="center", ha=ha, fontsize=9)

    ax.text(
        0.02,
        0.04,
        "Annotated takeaway: the recovery-themed post was scored strongly negative,\n"
        "the sensory 'Lemon' piece strongly positive, and the reflective advice post landed neutral.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F1E8", "edgecolor": "#C7BCA1"},
    )
    fig.tight_layout()
    return save_fig(fig, "fig2_live_sentiment_scores.png")


def plot_model_task_comparison(metrics: dict) -> Path:
    values = pd.DataFrame(
        [
            {"metric": "Emotion micro F1", "value": metrics["test_metrics"]["micro"]["f1"], "group": "Emotion"},
            {"metric": "Emotion macro F1", "value": metrics["test_metrics"]["macro"]["f1"], "group": "Emotion"},
            {
                "metric": "Sentiment accuracy",
                "value": metrics["test_sentiment_metrics"]["accuracy"],
                "group": "Sentiment",
            },
            {
                "metric": "Sentiment macro F1",
                "value": metrics["test_sentiment_metrics"]["macro"]["f1"],
                "group": "Sentiment",
            },
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {"Emotion": "#4C72B0", "Sentiment": "#DD8452"}
    bars = ax.bar(values["metric"], values["value"], color=[palette[group] for group in values["group"]], width=0.65)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Lemotif BERT test-set performance", fontsize=14, pad=14)
    ax.grid(axis="y", alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

    for bar, value in zip(bars, values["value"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.96,
        "Annotated takeaway: the model is much better at coarse sentiment than at\n"
        "fine-grained 18-label emotion recognition, which matches the thesis framing.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F1E8", "edgecolor": "#C7BCA1"},
    )
    fig.tight_layout()
    return save_fig(fig, "fig3_model_task_comparison.png")


def plot_model_sentiment_class_f1(metrics: dict) -> Path:
    label_metrics = pd.DataFrame(metrics["test_sentiment_metrics"]["label_metrics"])
    label_metrics["color"] = label_metrics["label"].map(SENTIMENT_COLORS)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(
        label_metrics["label"].str.title(),
        label_metrics["f1"],
        color=label_metrics["color"],
        width=0.6,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class sentiment F1 on the Lemotif test set", fontsize=14, pad=14)
    ax.grid(axis="y", alpha=0.2)

    for bar, (_, row) in zip(bars, label_metrics.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["f1"] + 0.02,
            f"{row['f1']:.3f}\n(n={int(row['support'])})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.96,
        "Annotated takeaway: positive and negative are learned well, but neutral is weak\n"
        "and rare, making it the main error source in the derived sentiment task.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F1E8", "edgecolor": "#C7BCA1"},
    )
    fig.tight_layout()
    return save_fig(fig, "fig4_model_sentiment_class_f1.png")


def write_markdown_report(fig_paths: list[Path], frame: pd.DataFrame, metrics: dict) -> Path:
    report_path = REPORT_DIR / "findings_summary.md"
    sentiment_accuracy = metrics["test_sentiment_metrics"]["accuracy"]
    sentiment_macro_f1 = metrics["test_sentiment_metrics"]["macro"]["f1"]
    emotion_macro_f1 = metrics["test_metrics"]["macro"]["f1"]

    report = f"""# Findings Summary

This summary combines:

- a live sample of {len(frame)} public journal posts processed by the new pipeline
- the saved Lemotif BERT evaluation metrics from the fine-tuned checkpoint

## 1. Live sample sentiment distribution

![Live sentiment distribution]({fig_paths[0]})

Short explanation: the current live crawl is evenly split across negative, neutral, and positive posts, so the pipeline is already picking up varied tone rather than collapsing everything into one class.

## 2. Live sample sentiment score by post

![Live sentiment scores]({fig_paths[1]})

Short explanation: the model separates very different writing styles clearly. The recovery-focused post was scored negative, the sensory/creative post positive, and the reflective advice post neutral.

## 3. BERT test-set performance

![Model task comparison]({fig_paths[2]})

Short explanation: the fine-tuned model performs much better on derived sentiment ({sentiment_accuracy:.3f} accuracy, {sentiment_macro_f1:.3f} macro F1) than on the full 18-emotion task ({emotion_macro_f1:.3f} macro F1). That means sentiment is the stronger and more reliable downstream use case.

## 4. Per-class sentiment F1

![Per-class sentiment F1]({fig_paths[3]})

Short explanation: positive and negative sentiment are learned well, but neutral is the weak point. In practice, that means the model is good at detecting clear affect but less reliable on balanced or low-affect entries.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    ensure_report_dir()
    frame, metrics = load_inputs()
    if frame.empty:
        raise RuntimeError("No successful journal sentiment rows were found in output/journal_sentiment_results.json.")

    fig_paths = [
        plot_live_sentiment_distribution(frame),
        plot_live_sentiment_scores(frame),
        plot_model_task_comparison(metrics),
        plot_model_sentiment_class_f1(metrics),
    ]
    report_path = write_markdown_report(fig_paths, frame, metrics)
    print(f"Saved findings report to {report_path}")


if __name__ == "__main__":
    main()
