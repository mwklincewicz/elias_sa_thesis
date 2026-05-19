from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BERT_SUMMARY = ROOT / "test data" / "output" / "stories_model_variants" / "stories_model_variant_summary.csv"
GEMMA_SUMMARY = ROOT / "test data" / "output" / "gemma_emotion_stories" / "comparison" / "gemma_stories_emotion_summary.csv"
FIGURE_DIR = ROOT / "test data" / "output" / "gemma_emotion_stories" / "figures"


def main() -> None:
    bert = pd.read_csv(BERT_SUMMARY)
    gemma = pd.read_csv(GEMMA_SUMMARY)

    bert_optuna = bert.loc[bert["variant_key"] == "optuna_checkpoint"].iloc[0]
    rows = [
        {
            "model": "BERT Optuna",
            "emotion_micro_f1": bert_optuna["emotion_micro_f1"],
            "emotion_macro_f1": bert_optuna["emotion_macro_f1"],
            "sentiment_accuracy": bert_optuna["sentiment_accuracy"],
            "sentiment_macro_f1": bert_optuna["sentiment_macro_f1"],
            "avg_predicted_labels": bert_optuna["avg_predicted_labels"],
        }
    ]

    for _, row in gemma.iterrows():
        rows.append(
            {
                "model": row["run"],
                "emotion_micro_f1": row["emotion_micro_f1"],
                "emotion_macro_f1": row["emotion_macro_f1"],
                "sentiment_accuracy": row["sentiment_accuracy"],
                "sentiment_macro_f1": row["sentiment_macro_f1"],
                "avg_predicted_labels": row["avg_predicted_labels"],
            }
        )

    comparison = pd.DataFrame(rows)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(FIGURE_DIR / "stories_bert_optuna_gemma_18label_comparison.csv", index=False)

    metric_cols = [
        "emotion_micro_f1",
        "emotion_macro_f1",
        "sentiment_accuracy",
        "sentiment_macro_f1",
    ]
    metric_labels = ["Emotion micro-F1", "Emotion macro-F1", "Sentiment accuracy", "Sentiment macro-F1"]

    colors = ["#405B8F", "#6BA06F", "#C9855F"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.8),
        gridspec_kw={"height_ratios": [3, 1.45]},
        constrained_layout=True,
    )

    metric_df = comparison.set_index("model")[metric_cols].T
    metric_df.index = metric_labels
    metric_df.plot(kind="bar", ax=axes[0], color=colors, width=0.78)
    axes[0].set_title("Stories external evaluation: BERT Optuna vs. Gemma 18-label emotion models")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=10)
    axes[0].legend(loc="upper left", frameon=True)

    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    card = comparison[["model", "avg_predicted_labels"]].copy()
    card.plot(
        kind="bar",
        x="model",
        y="avg_predicted_labels",
        ax=axes[1],
        color=colors,
        legend=False,
        width=0.55,
    )
    axes[1].axhline(float(bert_optuna["avg_true_labels"]), color="#1F1F1F", linestyle="--", linewidth=1.4)
    axes[1].text(
        2.45,
        float(bert_optuna["avg_true_labels"]) + 0.08,
        "Gold avg. = 2.123",
        fontsize=9,
        color="#1F1F1F",
        ha="right",
    )
    axes[1].set_ylabel("Avg. predicted labels")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 3.5)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=0)

    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    output = FIGURE_DIR / "fig_stories_bert_optuna_vs_gemma_18label.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
