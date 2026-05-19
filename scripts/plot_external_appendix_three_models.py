from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "overleaf_update_work_20260510_gemma" / "figures" / "results" / "appendix"

BERT_SUMMARY = ROOT / "test data" / "output" / "stories_model_variants" / "stories_model_variant_summary.csv"
GEMMA_SUMMARY = ROOT / "test data" / "output" / "gemma_emotion_stories" / "comparison" / "gemma_stories_emotion_summary.csv"

PER_EMOTION = {
    "BERT Optuna": ROOT / "test data" / "output" / "stories_model_variants" / "optuna_checkpoint" / "per_emotion_metrics.csv",
    "BERT fixed inference": ROOT / "test data" / "output" / "stories_model_variants" / "fixed_inference" / "per_emotion_metrics.csv",
    "Gemma fine-tuned": ROOT / "test data" / "output" / "gemma_emotion_stories" / "finetuned" / "per_emotion_metrics.csv",
}

PREDICTIONS = {
    "BERT Optuna": ROOT / "test data" / "output" / "stories_model_variants" / "optuna_checkpoint" / "stories_predictions.csv",
    "BERT fixed inference": ROOT / "test data" / "output" / "stories_model_variants" / "fixed_inference" / "stories_predictions.csv",
    "Gemma fine-tuned": ROOT / "test data" / "output" / "gemma_emotion_stories" / "finetuned" / "stories_predictions.csv",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def load_metric_rows() -> pd.DataFrame:
    bert = pd.read_csv(BERT_SUMMARY)
    gemma = pd.read_csv(GEMMA_SUMMARY)
    optuna = bert.loc[bert["variant_key"] == "optuna_checkpoint"].iloc[0]
    fixed = bert.loc[bert["variant_key"] == "fixed_inference"].iloc[0]
    gemma_ft = gemma.loc[gemma["run"] == "Gemma fine-tuned"].iloc[0]

    rows = []
    for name, row in [
        ("BERT Optuna", optuna),
        ("BERT fixed inference", fixed),
        ("Gemma fine-tuned", gemma_ft),
    ]:
        rows.append(
            {
                "model": name,
                "Emotion micro-F1": row["emotion_micro_f1"],
                "Emotion macro-F1": row["emotion_macro_f1"],
                "Sentiment accuracy": row["sentiment_accuracy"],
                "Sentiment macro-F1": row["sentiment_macro_f1"],
                "Hamming loss": row["emotion_hamming_loss"],
                "Avg. labels": row["avg_predicted_labels"],
            }
        )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame) -> None:
    cols = ["Emotion micro-F1", "Emotion macro-F1", "Sentiment accuracy", "Sentiment macro-F1"]
    colors = ["#405B8F", "#A96A50", "#6BA06F"]
    ax = metrics.set_index("model")[cols].T.plot(kind="bar", figsize=(8.8, 4.8), color=colors, width=0.78)
    ax.set_title("Stories external metrics for selected model tradeoffs")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=10)
    ax.legend(loc="upper left", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(OUT / "app_fig16_external_selected_model_metrics.png", dpi=300)
    plt.close()


def plot_per_emotion() -> None:
    frames = []
    for model, path in PER_EMOTION.items():
        frame = pd.read_csv(path)[["label", "f1", "support"]].copy()
        frame["model"] = model
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    order = (
        data.groupby("label")["support"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    pivot = data.pivot(index="label", columns="model", values="f1").loc[order]
    ax = pivot.plot(kind="barh", figsize=(8.8, 7.0), color=["#405B8F", "#A96A50", "#6BA06F"], width=0.75)
    ax.set_title("Stories per-emotion F1 for selected external models")
    ax.set_xlabel("$F_1$")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUT / "app_fig17_external_selected_per_emotion_f1.png", dpi=300)
    plt.close()


def plot_cardinality() -> None:
    prediction_frames = []
    true_counts = None
    for model, path in PREDICTIONS.items():
        frame = pd.read_csv(path)
        if true_counts is None:
            true_counts = frame["true_label_count"].value_counts().sort_index()
        counts = frame["predicted_label_count"].value_counts().sort_index()
        for label_count, rows in counts.items():
            prediction_frames.append({"model": model, "label_count": int(label_count), "rows": int(rows)})

    pred = pd.DataFrame(prediction_frames)
    max_count = int(max(pred["label_count"].max(), true_counts.index.max()))
    idx = list(range(0, max_count + 1))
    pred_pivot = pred.pivot(index="label_count", columns="model", values="rows").reindex(idx).fillna(0)

    ax = pred_pivot.plot(kind="bar", figsize=(8.8, 4.8), color=["#405B8F", "#A96A50", "#6BA06F"], width=0.82)
    ax.plot(idx, true_counts.reindex(idx).fillna(0).values, color="#1F1F1F", marker="o", linewidth=1.8, label="Gold label count")
    ax.set_title("Stories predicted label cardinality for selected external models")
    ax.set_xlabel("Emotion labels per reflection")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUT / "app_fig18_external_selected_cardinality.png", dpi=300)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    metrics = load_metric_rows()
    metrics.to_csv(OUT / "external_selected_model_metrics.csv", index=False)
    plot_metrics(metrics)
    plot_per_emotion()
    plot_cardinality()


if __name__ == "__main__":
    main()
