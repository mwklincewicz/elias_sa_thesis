from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import load_training_history, save_figure


def main() -> None:
    history = load_training_history()
    for column in ["train_loss", "val_loss", "val_micro_f1", "val_macro_f1"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    has_train_loss = "train_loss" in history.columns and history["train_loss"].notna().any()
    has_val_loss = "val_loss" in history.columns and history["val_loss"].notna().any()
    has_val_micro_f1 = "val_micro_f1" in history.columns and history["val_micro_f1"].notna().any()
    has_val_macro_f1 = "val_macro_f1" in history.columns and history["val_macro_f1"].notna().any()

    if has_train_loss:
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", linewidth=2, label="Train loss")
    if has_val_loss:
        axes[0].plot(history["epoch"], history["val_loss"], marker="o", linewidth=2, label="Validation loss")
    if has_train_loss or has_val_loss:
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            "No loss history available\nin the saved run.",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(history["epoch"])

    if has_val_micro_f1:
        axes[1].plot(
            history["epoch"],
            history["val_micro_f1"],
            marker="o",
            linewidth=2,
            label="Validation micro F1",
        )
    if has_val_macro_f1:
        axes[1].plot(
            history["epoch"],
            history["val_macro_f1"],
            marker="o",
            linewidth=2,
            label="Validation macro F1",
        )
    if has_val_micro_f1 or has_val_macro_f1:
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "Validation F1 history was not\nsaved for this run.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    axes[1].set_title("Validation F1 by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    axes[1].set_xticks(history["epoch"])
    axes[1].set_ylim(0, 1)

    fig.suptitle("Training progress", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = save_figure(fig, "fig1_training_progress.png")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
