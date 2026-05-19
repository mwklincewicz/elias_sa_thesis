from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import count_label_string, load_test_predictions, save_figure


def main() -> None:
    predictions = load_test_predictions().copy()
    predictions["true_label_count"] = predictions["true_emotions"].apply(count_label_string)
    predictions["predicted_label_count"] = predictions["predicted_emotions"].apply(count_label_string)

    true_distribution = predictions["true_label_count"].value_counts().sort_index()
    predicted_distribution = predictions["predicted_label_count"].value_counts().sort_index()
    count_axis = sorted(set(true_distribution.index).union(predicted_distribution.index))

    true_values = true_distribution.reindex(count_axis, fill_value=0)
    predicted_values = predicted_distribution.reindex(count_axis, fill_value=0)

    x = np.arange(len(count_axis))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.bar(x - width / 2, true_values.values, width=width, label="True labels")
    ax.bar(x + width / 2, predicted_values.values, width=width, label="Predicted labels")

    ax.set_title("True versus predicted emotion-label counts")
    ax.set_xlabel("Emotion labels per reflection")
    ax.set_ylabel("Number of reflections")
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in count_axis])
    ax.legend()

    summary = (
        f"Average true labels: {predictions['true_label_count'].mean():.2f}\n"
        f"Average predicted labels: {predictions['predicted_label_count'].mean():.2f}"
    )
    ax.text(
        0.98,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    out = save_figure(fig, "fig5_true_vs_predicted_label_counts.png")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
