from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from common import load_per_label_metrics, save_figure


def main() -> None:
    label_metrics = load_per_label_metrics("test_metrics").sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, max(5.2, 0.38 * len(label_metrics))))
    colors = sns.color_palette("crest", len(label_metrics))
    bars = ax.barh(label_metrics["label"], label_metrics["f1"], color=colors)

    for bar, support in zip(bars, label_metrics["support"]):
        x = min(0.97, max(0.02, bar.get_width() + 0.015))
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x, y, f"n={int(support)}", va="center", fontsize=9)

    ax.set_title("Per-emotion test F1 scores")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    out = save_figure(fig, "fig3_per_emotion_f1.png")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
