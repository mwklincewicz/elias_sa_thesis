from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import load_metrics, save_figure


def main() -> None:
    metrics = load_metrics()
    validation = metrics["validation_metrics"]
    test = metrics["test_metrics"]

    metric_specs = [
        ("Micro precision", validation["micro"]["precision"], test["micro"]["precision"]),
        ("Micro recall", validation["micro"]["recall"], test["micro"]["recall"]),
        ("Micro F1", validation["micro"]["f1"], test["micro"]["f1"]),
        ("Macro precision", validation["macro"]["precision"], test["macro"]["precision"]),
        ("Macro recall", validation["macro"]["recall"], test["macro"]["recall"]),
        ("Macro F1", validation["macro"]["f1"], test["macro"]["f1"]),
        ("Samples F1", validation["samples"]["f1"], test["samples"]["f1"]),
    ]

    rows = []
    for metric_name, validation_value, test_value in metric_specs:
        rows.append({"metric": metric_name, "split": "Validation", "value": validation_value})
        rows.append({"metric": metric_name, "split": "Test", "value": test_value})

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.barplot(data=plot_df, x="metric", y="value", hue="split", ax=ax)
    ax.set_title("Validation and test metric comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out = save_figure(fig, "fig2_validation_vs_test_metrics.png")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
