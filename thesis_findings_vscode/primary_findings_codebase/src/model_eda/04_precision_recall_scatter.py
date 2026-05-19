from __future__ import annotations

import matplotlib.pyplot as plt

from common import load_per_label_metrics, save_figure


def main() -> None:
    label_metrics = load_per_label_metrics("test_metrics")
    sizes = 110 + label_metrics["support"].astype(float) * 8

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.scatter(
        label_metrics["precision"],
        label_metrics["recall"],
        s=sizes,
        alpha=0.78,
        edgecolors="black",
        linewidths=0.6,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)

    for index, row in label_metrics.reset_index(drop=True).iterrows():
        offset_x = 6 if index % 2 == 0 else -10
        offset_y = 6 if index % 3 else -10
        ax.annotate(
            row["label"],
            (row["precision"], row["recall"]),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=9,
        )

    ax.set_title("Per-emotion precision vs recall")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    out = save_figure(fig, "fig4_precision_recall_scatter.png")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
