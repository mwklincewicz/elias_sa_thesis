from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINED_DIR = PROJECT_ROOT / "test data" / "output" / "external_evaluation"
UNTRAINED_DIR = PROJECT_ROOT / "test data" / "output" / "untrained_bert"
COMPARISON_DIR = UNTRAINED_DIR / "comparisons"

TRAINED_METRICS_PATH = TRAINED_DIR / "metrics.json"
TRAINED_PREDICTIONS_PATH = TRAINED_DIR / "stories_test_predictions.csv"
UNTRAINED_METRICS_PATH = UNTRAINED_DIR / "metrics.json"
UNTRAINED_PREDICTIONS_PATH = UNTRAINED_DIR / "stories_test_predictions.csv"

SUMMARY_CSV_PATH = UNTRAINED_DIR / "comparison_summary.csv"
SUMMARY_TEXT_PATH = UNTRAINED_DIR / "comparison_summary.txt"
PER_EMOTION_CSV_PATH = UNTRAINED_DIR / "per_emotion_f1_comparison.csv"
PER_SENTIMENT_CSV_PATH = UNTRAINED_DIR / "per_sentiment_f1_comparison.csv"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]

sns.set_theme(style="whitegrid")
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def save_figure(fig, destination: Path, dpi: int = 220) -> None:
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_summary_frame(trained_metrics: dict, untrained_metrics: dict) -> pd.DataFrame:
    trained_emotion = trained_metrics["test_metrics"]
    untrained_emotion = untrained_metrics["test_metrics"]
    trained_sentiment = trained_metrics["test_sentiment_metrics"]
    untrained_sentiment = untrained_metrics["test_sentiment_metrics"]

    rows = [
        ("Emotion", "Micro precision", trained_emotion["micro"]["precision"], untrained_emotion["micro"]["precision"]),
        ("Emotion", "Micro recall", trained_emotion["micro"]["recall"], untrained_emotion["micro"]["recall"]),
        ("Emotion", "Micro F1", trained_emotion["micro"]["f1"], untrained_emotion["micro"]["f1"]),
        ("Emotion", "Macro F1", trained_emotion["macro"]["f1"], untrained_emotion["macro"]["f1"]),
        ("Emotion", "Subset accuracy", trained_emotion["subset_accuracy"], untrained_emotion["subset_accuracy"]),
        ("Emotion", "Hamming loss", trained_emotion["hamming_loss"], untrained_emotion["hamming_loss"]),
        (
            "Emotion",
            "Avg predicted labels",
            trained_emotion["avg_predicted_labels_per_entry"],
            untrained_emotion["avg_predicted_labels_per_entry"],
        ),
        ("Sentiment", "Accuracy", trained_sentiment["accuracy"], untrained_sentiment["accuracy"]),
        ("Sentiment", "Macro precision", trained_sentiment["macro"]["precision"], untrained_sentiment["macro"]["precision"]),
        ("Sentiment", "Macro recall", trained_sentiment["macro"]["recall"], untrained_sentiment["macro"]["recall"]),
        ("Sentiment", "Macro F1", trained_sentiment["macro"]["f1"], untrained_sentiment["macro"]["f1"]),
    ]

    frame = pd.DataFrame(rows, columns=["task", "metric", "trained", "untrained"])
    frame["delta_trained_minus_untrained"] = frame["trained"] - frame["untrained"]
    return frame


def build_label_frame(
    trained_metrics: dict,
    untrained_metrics: dict,
    section_key: str,
) -> pd.DataFrame:
    trained = pd.DataFrame(trained_metrics[section_key]["label_metrics"]).rename(
        columns={"precision": "trained_precision", "recall": "trained_recall", "f1": "trained_f1", "support": "support"}
    )
    untrained = pd.DataFrame(untrained_metrics[section_key]["label_metrics"]).rename(
        columns={"precision": "untrained_precision", "recall": "untrained_recall", "f1": "untrained_f1", "support": "untrained_support"}
    )

    merged = trained.merge(untrained, on="label", how="outer")
    merged["support"] = merged["support"].fillna(merged["untrained_support"]).fillna(0).astype(int)
    merged = merged.drop(columns=["untrained_support"])
    merged["delta_f1"] = merged["trained_f1"] - merged["untrained_f1"]
    return merged


def build_summary_text(
    summary_frame: pd.DataFrame,
    trained_predictions: pd.DataFrame,
    untrained_predictions: pd.DataFrame,
) -> str:
    emotion_micro_f1 = summary_frame.loc[summary_frame["metric"] == "Micro F1"].iloc[0]
    sentiment_accuracy = summary_frame.loc[summary_frame["metric"] == "Accuracy"].iloc[0]
    trained_none_rate = (trained_predictions["predicted_emotions"] == "none").mean()
    untrained_none_rate = (untrained_predictions["predicted_emotions"] == "none").mean()
    trained_top_pattern = trained_predictions["predicted_emotions"].value_counts().index[0]
    trained_top_pattern_count = int(trained_predictions["predicted_emotions"].value_counts().iloc[0])
    untrained_top_pattern = untrained_predictions["predicted_emotions"].value_counts().index[0]
    untrained_top_pattern_count = int(untrained_predictions["predicted_emotions"].value_counts().iloc[0])
    trained_sentiment_counts = trained_predictions["predicted_sentiment"].value_counts()
    untrained_sentiment_counts = untrained_predictions["predicted_sentiment"].value_counts()

    lines = [
        "Trained vs untrained BERT on Stories",
        "===================================",
        f"Emotion micro F1: trained {emotion_micro_f1['trained']:.4f} vs untrained {emotion_micro_f1['untrained']:.4f}",
        f"Sentiment accuracy: trained {sentiment_accuracy['trained']:.4f} vs untrained {sentiment_accuracy['untrained']:.4f}",
        f"Predicted 'none' emotion set rate: trained {trained_none_rate:.4f} vs untrained {untrained_none_rate:.4f}",
        f"Most common trained prediction: {trained_top_pattern} ({trained_top_pattern_count} rows)",
        f"Most common untrained prediction: {untrained_top_pattern} ({untrained_top_pattern_count} rows)",
        (
            "Predicted sentiment counts: "
            f"trained positive={int(trained_sentiment_counts.get('positive', 0))}, "
            f"neutral={int(trained_sentiment_counts.get('neutral', 0))}, "
            f"negative={int(trained_sentiment_counts.get('negative', 0))}; "
            f"untrained positive={int(untrained_sentiment_counts.get('positive', 0))}, "
            f"neutral={int(untrained_sentiment_counts.get('neutral', 0))}, "
            f"negative={int(untrained_sentiment_counts.get('negative', 0))}"
        ),
        "",
        "Interpretation:",
        "- The trained model benefits substantially from Lemotif supervision before facing Stories.",
        "- The untrained model acts as a true random-weight baseline with the same label space and threshold.",
        "- The untrained model mostly collapses into one repeated label pattern instead of learning varied emotion combinations.",
    ]
    return "\n".join(lines) + "\n"


def plot_compare_overall_emotion_metrics(summary_frame: pd.DataFrame) -> None:
    frame = summary_frame[summary_frame["task"] == "Emotion"].copy()
    plot_frame = frame[frame["metric"] != "Avg predicted labels"]
    melted = plot_frame.melt(id_vars=["metric"], value_vars=["trained", "untrained"], var_name="model", value_name="value")
    melted["model"] = melted["model"].map({"trained": "Trained BERT", "untrained": "Untrained BERT"})

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.barplot(data=melted, x="metric", y="value", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Stories emotion metrics: trained vs untrained BERT")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_overall_emotion_metrics.png")


def plot_compare_overall_sentiment_metrics(summary_frame: pd.DataFrame) -> None:
    frame = summary_frame[summary_frame["task"] == "Sentiment"].copy()
    melted = frame.melt(id_vars=["metric"], value_vars=["trained", "untrained"], var_name="model", value_name="value")
    melted["model"] = melted["model"].map({"trained": "Trained BERT", "untrained": "Untrained BERT"})

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    sns.barplot(data=melted, x="metric", y="value", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Stories sentiment metrics: trained vs untrained BERT")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_overall_sentiment_metrics.png")


def plot_compare_per_sentiment_f1(per_sentiment_frame: pd.DataFrame) -> None:
    frame = per_sentiment_frame.copy()
    frame["label"] = pd.Categorical(frame["label"], categories=SENTIMENT_ORDER, ordered=True)
    melted = frame.melt(
        id_vars=["label", "support"],
        value_vars=["trained_f1", "untrained_f1"],
        var_name="model",
        value_name="f1",
    )
    melted["model"] = melted["model"].map({"trained_f1": "Trained BERT", "untrained_f1": "Untrained BERT"})

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.barplot(data=melted, x="label", y="f1", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Per-sentiment F1 on Stories")
    ax.set_xlabel("")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_per_sentiment_f1.png")


def plot_compare_per_emotion_f1(per_emotion_frame: pd.DataFrame) -> None:
    frame = per_emotion_frame.sort_values("trained_f1", ascending=True).copy()
    melted = frame.melt(
        id_vars=["label", "support"],
        value_vars=["trained_f1", "untrained_f1"],
        var_name="model",
        value_name="f1",
    )
    melted["model"] = melted["model"].map({"trained_f1": "Trained BERT", "untrained_f1": "Untrained BERT"})

    fig, ax = plt.subplots(figsize=(10.2, 8.8))
    sns.barplot(data=melted, x="f1", y="label", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Per-emotion F1 on Stories")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)
    ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_per_emotion_f1.png")


def plot_compare_predicted_sentiment_distribution(
    trained_predictions: pd.DataFrame,
    untrained_predictions: pd.DataFrame,
) -> None:
    trained_counts = trained_predictions["predicted_sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    untrained_counts = untrained_predictions["predicted_sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)

    frame = pd.DataFrame(
        {
            "sentiment": SENTIMENT_ORDER * 2,
            "model": ["Trained BERT"] * len(SENTIMENT_ORDER) + ["Untrained BERT"] * len(SENTIMENT_ORDER),
            "count": list(trained_counts.values) + list(untrained_counts.values),
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=frame, x="sentiment", y="count", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Predicted sentiment distribution on Stories")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_predicted_sentiment_distribution.png")


def main() -> None:
    trained_metrics = load_json(TRAINED_METRICS_PATH)
    untrained_metrics = load_json(UNTRAINED_METRICS_PATH)
    trained_predictions = load_predictions(TRAINED_PREDICTIONS_PATH)
    untrained_predictions = load_predictions(UNTRAINED_PREDICTIONS_PATH)

    summary_frame = build_summary_frame(trained_metrics, untrained_metrics)
    per_emotion_frame = build_label_frame(trained_metrics, untrained_metrics, "test_metrics")
    per_sentiment_frame = build_label_frame(trained_metrics, untrained_metrics, "test_sentiment_metrics")

    summary_frame.to_csv(SUMMARY_CSV_PATH, index=False)
    per_emotion_frame.to_csv(PER_EMOTION_CSV_PATH, index=False)
    per_sentiment_frame.to_csv(PER_SENTIMENT_CSV_PATH, index=False)
    SUMMARY_TEXT_PATH.write_text(
        build_summary_text(summary_frame, trained_predictions, untrained_predictions),
        encoding="utf-8",
    )

    plot_compare_overall_emotion_metrics(summary_frame)
    plot_compare_overall_sentiment_metrics(summary_frame)
    plot_compare_per_sentiment_f1(per_sentiment_frame)
    plot_compare_per_emotion_f1(per_emotion_frame)
    plot_compare_predicted_sentiment_distribution(trained_predictions, untrained_predictions)

    print(f"Saved summary table to: {SUMMARY_CSV_PATH}")
    print(f"Saved summary text to: {SUMMARY_TEXT_PATH}")
    print(f"Saved per-emotion comparison to: {PER_EMOTION_CSV_PATH}")
    print(f"Saved per-sentiment comparison to: {PER_SENTIMENT_CSV_PATH}")
    print(f"Saved comparison figures to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
