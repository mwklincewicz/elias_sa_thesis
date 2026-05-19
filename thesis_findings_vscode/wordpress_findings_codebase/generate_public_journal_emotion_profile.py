from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "output" / "selected_site_analysis" / "public_reflection_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "output" / "selected_site_analysis" / "report"
FIGURE_PATH = REPORT_DIR / "fig5_public_journals_per_emotion_prevalence.png"
SUMMARY_PATH = REPORT_DIR / "public_journal_emotion_profile.csv"


def friendly_label(column: str) -> str:
    return column.split("__", 1)[1].replace("_", " ").title()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    emotion_cols = [c for c in df.columns if c.startswith("emotion__") and not c.startswith("emotion_prob__")]

    summary = pd.DataFrame(
        {
            "column": emotion_cols,
            "label": [friendly_label(c) for c in emotion_cols],
            "count": [int(df[c].sum()) for c in emotion_cols],
            "prevalence": [float(df[c].mean()) for c in emotion_cols],
        }
    ).sort_values("prevalence", ascending=True)

    return summary


def plot(summary: pd.DataFrame, reflection_count: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10.8, max(5.4, 0.38 * len(summary))))
    colors = plt.cm.PuBuGn(np.linspace(0.45, 0.95, len(summary)))
    bars = ax.barh(summary["label"], summary["prevalence"], color=colors, edgecolor="none")

    for bar, (_, row) in zip(bars, summary.iterrows()):
        x = min(0.97, max(0.02, float(row["prevalence"]) + 0.01))
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x, y, f"n={int(row['count'])}", va="center", fontsize=9, color="#333333")

    ax.set_title("Public Journals: Per-emotion prevalence", fontsize=15, pad=10)
    ax.set_xlabel("Share of reflections with predicted emotion label")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.6, float(summary["prevalence"].max()) + 0.06))

    note = (
        "This mirrors the per-emotion F1 layout, but uses prevalence because the public-journal set has no gold labels. "
        f"n shows the number of predicted reflections per emotion across {reflection_count:,} reflections."
    )
    fig.text(0.02, 0.01, note, ha="left", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["error"].fillna("") == ""].copy()
    summary = build_summary(df)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot(summary, len(df))

    print(f"Saved figure to: {FIGURE_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
