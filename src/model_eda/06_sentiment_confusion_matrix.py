from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import SENTIMENT_ORDER, load_test_predictions, save_figure


def main() -> None:
    predictions = load_test_predictions()

    confusion = pd.crosstab(
        predictions["true_sentiment"],
        predictions["predicted_sentiment"],
    ).reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0)

    row_totals = confusion.sum(axis=1).replace(0, 1)
    normalized = confusion.div(row_totals, axis=0)

    annotations = confusion.astype(str) + "\n" + (normalized * 100).round(1).astype(str) + "%"

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    sns.heatmap(
        confusion,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title("Derived sentiment confusion matrix")
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("True sentiment")

    fig.tight_layout()
    out = save_figure(fig, "fig6_sentiment_confusion_matrix.png")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
