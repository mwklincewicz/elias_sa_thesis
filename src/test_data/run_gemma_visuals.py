from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEMMA_MODEL_DIR = (
    PROJECT_ROOT
    / "test data"
    / "output"
    / "gemma_sentiment"
    / "google_gemma-3n-E2B-it"
)
GEMMA_FIGURES_DIR = GEMMA_MODEL_DIR / "figures"
COMPARISON_DIR = PROJECT_ROOT / "test data" / "output" / "gemma_sentiment" / "comparisons"

GEMMA_METRICS_PATH = GEMMA_MODEL_DIR / "metrics.json"
GEMMA_PREDICTIONS_PATH = GEMMA_MODEL_DIR / "stories_test_predictions.csv"

BERT_METRICS_PATH = PROJECT_ROOT / "test data" / "output" / "external_evaluation" / "metrics.json"
BERT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "test data" / "output" / "external_evaluation" / "stories_test_predictions.csv"
)

SENTIMENT_ORDER = ["negative", "neutral", "positive"]

sns.set_theme(style="whitegrid")
GEMMA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def save_figure(fig, destination: Path, dpi: int = 220) -> None:
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def truncate_text(text: str, width: int = 115) -> str:
    cleaned = " ".join(str(text).split())
    return textwrap.shorten(cleaned, width=width, placeholder="...")


def wrap_text(text: str, width: int = 36) -> str:
    return textwrap.fill(str(text), width=width)


def plot_fig1_overall_sentiment_metrics(metrics: dict) -> None:
    sentiment_metrics = metrics["test_sentiment_metrics"]
    rows = [
        {"metric": "Accuracy", "value": sentiment_metrics["accuracy"]},
        {"metric": "Macro precision", "value": sentiment_metrics["macro"]["precision"]},
        {"metric": "Macro recall", "value": sentiment_metrics["macro"]["recall"]},
        {"metric": "Macro F1", "value": sentiment_metrics["macro"]["f1"]},
    ]
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=frame, x="metric", y="value", color="#2F6FB0", ax=ax)
    ax.set_title("Gemma 2B overall sentiment metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=18)

    for patch, value in zip(ax.patches, frame["value"]):
        ax.annotate(
            f"{value:.3f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            textcoords="offset points",
            xytext=(0, 4),
            fontsize=10,
        )

    fig.tight_layout()
    save_figure(fig, GEMMA_FIGURES_DIR / "fig1_overall_sentiment_metrics.png")


def plot_fig2_sentiment_confusion_matrix(predictions: pd.DataFrame) -> None:
    confusion = pd.crosstab(
        predictions["true_sentiment"],
        predictions["predicted_sentiment"],
    ).reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0)

    row_totals = confusion.sum(axis=1).replace(0, 1)
    normalized = confusion.div(row_totals, axis=0)
    annotations = confusion.astype(str) + "\n" + (normalized * 100).round(1).astype(str) + "%"

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    sns.heatmap(confusion, annot=annotations, fmt="", cmap="Blues", cbar_kws={"label": "Count"}, ax=ax)
    ax.set_title("Gemma 2B sentiment confusion matrix")
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("True sentiment")

    fig.tight_layout()
    save_figure(fig, GEMMA_FIGURES_DIR / "fig2_sentiment_confusion_matrix.png")


def plot_fig3_per_sentiment_f1(metrics: dict) -> None:
    frame = pd.DataFrame(metrics["test_sentiment_metrics"]["label_metrics"])
    frame["label"] = pd.Categorical(frame["label"], categories=SENTIMENT_ORDER, ordered=True)
    frame = frame.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.barplot(data=frame, x="f1", y="label", orient="h", palette="crest", ax=ax)
    ax.set_title("Gemma 2B per-sentiment F1")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)

    for patch, (_, row) in zip(ax.patches, frame.iterrows()):
        ax.text(
            float(row["f1"]) + 0.02,
            patch.get_y() + patch.get_height() / 2,
            f"n={int(row['support'])}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, GEMMA_FIGURES_DIR / "fig3_per_sentiment_f1.png")


def plot_fig4_precision_recall_scatter(metrics: dict) -> None:
    frame = pd.DataFrame(metrics["test_sentiment_metrics"]["label_metrics"])

    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    sns.scatterplot(
        data=frame,
        x="precision",
        y="recall",
        size="support",
        sizes=(180, 1100),
        hue="label",
        palette="Set2",
        legend=False,
        ax=ax,
    )

    for row in frame.itertuples(index=False):
        ax.text(float(row.precision) + 0.015, float(row.recall), str(row.label), fontsize=9)

    ax.set_title("Gemma 2B precision vs recall by sentiment")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")

    fig.tight_layout()
    save_figure(fig, GEMMA_FIGURES_DIR / "fig4_precision_recall_scatter.png")


def plot_fig5_true_vs_predicted_distribution(predictions: pd.DataFrame) -> None:
    true_counts = predictions["true_sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    pred_counts = predictions["predicted_sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)

    frame = pd.DataFrame(
        {
            "sentiment": SENTIMENT_ORDER * 2,
            "series": ["True"] * len(SENTIMENT_ORDER) + ["Predicted"] * len(SENTIMENT_ORDER),
            "count": list(true_counts.values) + list(pred_counts.values),
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(data=frame, x="sentiment", y="count", hue="series", palette=["#4C72B0", "#DD8452"], ax=ax)
    ax.set_title("Gemma 2B true vs predicted sentiment distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")

    fig.tight_layout()
    save_figure(fig, GEMMA_FIGURES_DIR / "fig5_true_vs_predicted_sentiment_distribution.png")


def plot_fig6_latency_distribution(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(predictions["latency_seconds"], bins=20, kde=True, color="#55A868", ax=ax)
    mean_latency = float(predictions["latency_seconds"].mean())
    median_latency = float(predictions["latency_seconds"].median())
    ax.axvline(mean_latency, color="#C44E52", linestyle="--", linewidth=1.8, label=f"Mean {mean_latency:.1f}s")
    ax.axvline(median_latency, color="#8172B2", linestyle=":", linewidth=1.8, label=f"Median {median_latency:.1f}s")
    ax.set_title("Gemma 2B latency distribution on Stories SA")
    ax.set_xlabel("Latency per example (seconds)")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, GEMMA_FIGURES_DIR / "fig6_latency_distribution.png")


def _choose_rows(frame: pd.DataFrame, selected_ids: set[int], amount: int) -> list[pd.Series]:
    chosen: list[pd.Series] = []
    for _, row in frame.iterrows():
        row_id = int(row["row_id"])
        if row_id in selected_ids:
            continue
        chosen.append(row)
        selected_ids.add(row_id)
        if len(chosen) == amount:
            break
    return chosen


def plot_fig7_example_prediction_table(predictions: pd.DataFrame) -> None:
    predictions = predictions.copy()
    predictions["is_correct"] = predictions["true_sentiment"] == predictions["predicted_sentiment"]

    selected_ids: set[int] = set()

    strong_rows = _choose_rows(
        predictions[predictions["is_correct"]].sort_values(["latency_seconds", "row_id"]),
        selected_ids,
        2,
    )
    neutral_rows = _choose_rows(
        predictions[
            (predictions["true_sentiment"] == "neutral")
            | (predictions["predicted_sentiment"] == "neutral")
        ].sort_values(["is_correct", "latency_seconds", "row_id"], ascending=[False, True, True]),
        selected_ids,
        2,
    )
    miss_rows = _choose_rows(
        predictions[~predictions["is_correct"]].sort_values(["latency_seconds", "row_id"], ascending=[False, True]),
        selected_ids,
        2,
    )

    chosen_rows = strong_rows + neutral_rows + miss_rows
    row_labels = [
        "Strong match 1",
        "Strong match 2",
        "Neutral case 1",
        "Neutral case 2",
        "Mismatch 1",
        "Mismatch 2",
    ]

    rows = []
    for row in chosen_rows:
        rows.append(
            [
                wrap_text(truncate_text(row["text"], 118), 38),
                wrap_text(row["true_emotions"], 26),
                str(row["true_sentiment"]),
                str(row["predicted_sentiment"]),
                wrap_text(str(row["raw_model_output"]), 22),
            ]
        )

    fig, ax = plt.subplots(figsize=(15.8, 7.9))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Reflection excerpt",
            "True emotions",
            "True sentiment",
            "Predicted sentiment",
            "Raw model output",
        ],
        rowLabels=row_labels[: len(rows)],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.3)

    row_colors = {
        1: "#eaf4ea",
        2: "#eaf4ea",
        3: "#eef4fb",
        4: "#eef4fb",
        5: "#fdecec",
        6: "#fdecec",
    }
    for row_index, color in row_colors.items():
        for column_index in range(-1, 5):
            if (row_index, column_index) in table.get_celld():
                table[(row_index, column_index)].set_facecolor(color)

    ax.set_title("Representative Gemma 2B prediction patterns", pad=18)
    fig.tight_layout()
    save_figure(fig, GEMMA_FIGURES_DIR / "fig7_example_prediction_table.png", dpi=240)


def plot_compare_overall_metrics(gemma_metrics: dict, bert_metrics: dict) -> None:
    gemma_sentiment = gemma_metrics["test_sentiment_metrics"]
    bert_sentiment = bert_metrics["test_sentiment_metrics"]

    rows = [
        {"metric": "Accuracy", "model": "Gemma 2B", "value": gemma_sentiment["accuracy"]},
        {"metric": "Accuracy", "model": "BERT baseline", "value": bert_sentiment["accuracy"]},
        {"metric": "Macro precision", "model": "Gemma 2B", "value": gemma_sentiment["macro"]["precision"]},
        {"metric": "Macro precision", "model": "BERT baseline", "value": bert_sentiment["macro"]["precision"]},
        {"metric": "Macro recall", "model": "Gemma 2B", "value": gemma_sentiment["macro"]["recall"]},
        {"metric": "Macro recall", "model": "BERT baseline", "value": bert_sentiment["macro"]["recall"]},
        {"metric": "Macro F1", "model": "Gemma 2B", "value": gemma_sentiment["macro"]["f1"]},
        {"metric": "Macro F1", "model": "BERT baseline", "value": bert_sentiment["macro"]["f1"]},
    ]
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    sns.barplot(data=frame, x="metric", y="value", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Stories sentiment metrics: Gemma 2B vs BERT baseline")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=18)

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_overall_sentiment_metrics.png")


def plot_compare_per_class_f1(gemma_metrics: dict, bert_metrics: dict) -> None:
    gemma_frame = pd.DataFrame(gemma_metrics["test_sentiment_metrics"]["label_metrics"])
    gemma_frame["model"] = "Gemma 2B"
    bert_frame = pd.DataFrame(bert_metrics["test_sentiment_metrics"]["label_metrics"])
    bert_frame["model"] = "BERT baseline"
    frame = pd.concat([gemma_frame, bert_frame], ignore_index=True)
    frame["label"] = pd.Categorical(frame["label"], categories=SENTIMENT_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.barplot(data=frame, x="label", y="f1", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Per-sentiment F1: Gemma 2B vs BERT baseline")
    ax.set_xlabel("")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_per_class_f1.png")


def main() -> None:
    gemma_metrics = load_json(GEMMA_METRICS_PATH)
    gemma_predictions = load_predictions(GEMMA_PREDICTIONS_PATH)
    bert_metrics = load_json(BERT_METRICS_PATH)

    plot_fig1_overall_sentiment_metrics(gemma_metrics)
    plot_fig2_sentiment_confusion_matrix(gemma_predictions)
    plot_fig3_per_sentiment_f1(gemma_metrics)
    plot_fig4_precision_recall_scatter(gemma_metrics)
    plot_fig5_true_vs_predicted_distribution(gemma_predictions)
    plot_fig6_latency_distribution(gemma_predictions)
    plot_fig7_example_prediction_table(gemma_predictions)
    plot_compare_overall_metrics(gemma_metrics, bert_metrics)
    plot_compare_per_class_f1(gemma_metrics, bert_metrics)

    print(f"Saved Gemma figures to: {GEMMA_FIGURES_DIR}")
    print(f"Saved comparison figures to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
