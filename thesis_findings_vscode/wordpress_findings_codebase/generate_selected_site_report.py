from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "selected_site_analysis"
REPORT_DIR = OUTPUT_DIR / "report"
RESULTS_PATH = OUTPUT_DIR / "public_reflection_predictions.csv"
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"

SENTIMENT_COLORS = {
    "negative": "#C44E52",
    "neutral": "#8172B2",
    "positive": "#55A868",
}


def friendly_label(column: str) -> str:
    return column.split("__", 1)[1].replace("_", " ").title()


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def add_note(ax: plt.Axes, text: str, x: float = 0.02, y: float = 0.98) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F1E8", "edgecolor": "#C7BCA1"},
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def load_inputs() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["error"].fillna("") == ""].copy()
    summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    return df, summary


def build_frequency_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    emotion_cols = [c for c in df.columns if c.startswith("emotion__") and not c.startswith("emotion_prob__")]
    theme_cols = [c for c in df.columns if c.startswith("theme__") and not c.startswith("theme_prob__")]

    emotion_frequency = pd.DataFrame(
        {
            "column": emotion_cols,
            "label": [friendly_label(c) for c in emotion_cols],
            "count": [int(df[c].sum()) for c in emotion_cols],
            "prevalence": [float(df[c].mean()) for c in emotion_cols],
        }
    ).sort_values("prevalence", ascending=False)

    theme_frequency = pd.DataFrame(
        {
            "column": theme_cols,
            "label": [friendly_label(c) for c in theme_cols],
            "count": [int(df[c].sum()) for c in theme_cols],
            "prevalence": [float(df[c].mean()) for c in theme_cols],
        }
    ).sort_values("prevalence", ascending=False)

    emotion_frequency.to_csv(REPORT_DIR / "emotion_frequency.csv", index=False)
    theme_frequency.to_csv(REPORT_DIR / "theme_frequency.csv", index=False)
    return emotion_frequency, theme_frequency, emotion_cols, theme_cols


def build_association_tables(df: pd.DataFrame, emotion_cols: list[str], theme_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditional = pd.DataFrame(
        index=[friendly_label(c) for c in theme_cols],
        columns=[friendly_label(c) for c in emotion_cols],
        dtype=float,
    )
    for theme_col in theme_cols:
        mask = df[theme_col] == 1
        for emotion_col in emotion_cols:
            conditional.loc[friendly_label(theme_col), friendly_label(emotion_col)] = (
                float(df.loc[mask, emotion_col].mean()) if mask.any() else np.nan
            )

    emotion_corr = df[emotion_cols].corr()
    emotion_corr.index = [friendly_label(c) for c in emotion_cols]
    emotion_corr.columns = [friendly_label(c) for c in emotion_cols]

    conditional.to_csv(REPORT_DIR / "theme_emotion_conditional_prevalence.csv")
    emotion_corr.to_csv(REPORT_DIR / "emotion_correlation_matrix.csv")
    return conditional, emotion_corr


def plot_theme_prevalence(theme_frequency: pd.DataFrame) -> Path:
    top = theme_frequency.copy().sort_values("prevalence", ascending=True)
    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(top)))

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.barh(top["label"], top["prevalence"] * 100, color=colors)
    ax.set_title("Theme prevalence across selected reflective sites", fontsize=14, pad=12)
    ax.set_xlabel("Share of reflections with theme label (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(
            float(row["prevalence"]) * 100 + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{row['prevalence']*100:.1f}%\n(n={int(row['count'])})",
            va="center",
            fontsize=9,
        )

    add_note(
        ax,
        "Takeaway: the strongest inferred Lemotif themes in this site list are God,\n"
        "Recreation, Work, and Family, while School is almost absent.",
    )
    fig.tight_layout()
    return save_figure(fig, "fig1_theme_prevalence.png")


def plot_emotion_prevalence(emotion_frequency: pd.DataFrame) -> Path:
    top = emotion_frequency.head(12).copy().sort_values("prevalence", ascending=True)
    colors = plt.cm.OrRd(np.linspace(0.4, 0.9, len(top)))

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    bars = ax.barh(top["label"], top["prevalence"] * 100, color=colors)
    ax.set_title("Most frequent emotion labels across selected reflective sites", fontsize=14, pad=12)
    ax.set_xlabel("Share of reflections with emotion label (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(
            float(row["prevalence"]) * 100 + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{row['prevalence']*100:.1f}%\n(n={int(row['count'])})",
            va="center",
            fontsize=9,
        )

    add_note(
        ax,
        "Takeaway: anxious, frustrated, and surprised dominate this public-reflection sample,\n"
        "with happy also common but clearly below the main negative/tense cluster.",
    )
    fig.tight_layout()
    return save_figure(fig, "fig2_emotion_prevalence.png")


def plot_theme_emotion_heatmap(conditional: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    im = ax.imshow(conditional.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.01, np.nanmax(conditional.values)))
    ax.set_title("Emotion prevalence conditional on theme", fontsize=14, pad=12)
    ax.set_xticks(range(len(conditional.columns)))
    ax.set_xticklabels(conditional.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(conditional.index)))
    ax.set_yticklabels(conditional.index)

    for row_idx in range(conditional.shape[0]):
        for col_idx in range(conditional.shape[1]):
            value = conditional.iat[row_idx, col_idx]
            if pd.isna(value):
                label = "-"
            else:
                label = f"{value*100:.0f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7, color="#0F172A")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Conditional prevalence")
    add_note(
        ax,
        "Takeaway: Health, Sleep, and Work reflections are especially loaded with anxious/\n"
        "frustrated emotion labels, while God and Food skew relatively more positive.",
        x=0.01,
        y=-0.18,
    )
    fig.tight_layout()
    return save_figure(fig, "fig3_theme_emotion_heatmap.png")


def plot_emotion_correlation_matrix(emotion_corr: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 9.6))
    im = ax.imshow(emotion_corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_title("Emotion correlation matrix across predicted labels", fontsize=14, pad=12)
    ax.set_xticks(range(len(emotion_corr.columns)))
    ax.set_xticklabels(emotion_corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(emotion_corr.index)))
    ax.set_yticklabels(emotion_corr.index, fontsize=8)

    for row_idx in range(emotion_corr.shape[0]):
        for col_idx in range(emotion_corr.shape[1]):
            value = emotion_corr.iat[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#0F172A",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    add_note(
        ax,
        "Takeaway: the dominant correlation structure is a broad anxious/frustrated/sad/\n"
        "surprised cluster, while positive emotions tend to move together in a smaller block.",
        x=0.01,
        y=-0.15,
    )
    fig.tight_layout()
    return save_figure(fig, "fig4_emotion_correlation_matrix.png")


def write_report(
    df: pd.DataFrame,
    summary: dict,
    emotion_frequency: pd.DataFrame,
    theme_frequency: pd.DataFrame,
    figures: list[Path],
) -> Path:
    report_path = REPORT_DIR / "selected_site_summary.md"
    report = f"""# Selected Site Summary

This report uses the human-annotated Lemotif label space for inference:

- 18 Lemotif emotion labels from the fine-tuned emotion BERT model
- 11 Lemotif theme labels from a separately trained BERT theme model

## Scope

- Requested sites: {summary['site_count_requested']}
- Successful sites: {summary['site_count_successful']}
- Reflections analyzed: {summary['reflection_count_successful']}
- Window: last {summary['days_back_limit']} days
- Per-site cap: {summary['posts_per_site_limit']} reflections

## Important evaluation note

You were right: because this selected-site dataset is not human-labeled in the Lemotif format, we **cannot** compute macro F1, accuracy, or other true evaluation metrics for these reflections. Those metrics require gold labels. What we *can* do here is descriptive analysis of the model outputs: frequencies, co-occurrence patterns, and correlations.

## 1. Theme prevalence

![Theme prevalence]({figures[0]})

Short explanation: the strongest inferred themes are **God**, **Recreation**, **Work**, and **Family**. This suggests the selected blogs are not only personal diaries in a narrow sense, but often reflective writing about faith, leisure, and daily obligations.

## 2. Emotion prevalence

![Emotion prevalence]({figures[1]})

Short explanation: the most frequent emotions are **Anxious**, **Frustrated**, and **Surprised**, followed by **Sad**, **Angry**, **Confused**, and **Happy**. The public reflections in this sample lean toward emotionally tense or evaluative writing rather than mostly calm positivity.

## 3. Theme to emotion structure

![Theme-emotion heatmap]({figures[2]})

Short explanation: different themes pull different emotional profiles. In this run, **Health**, **Sleep**, and **Work** are especially associated with anxious/frustrated outputs, while **God** and **Food** show relatively more positive emotion activation.

## 4. Emotion correlation matrix

![Emotion correlation matrix]({figures[3]})

Short explanation: the emotion labels are not independent. A clear negative/tension cluster emerges around anxious, frustrated, sad, and surprised, while positive emotions cohere in a smaller block. That correlation structure is important if you later compare this model-inferred dataset against the original human-annotated Lemotif sample.

## Top frequencies

Top theme: **{theme_frequency.iloc[0]['label']}** ({theme_frequency.iloc[0]['prevalence']*100:.1f}%)

Top emotion: **{emotion_frequency.iloc[0]['label']}** ({emotion_frequency.iloc[0]['prevalence']*100:.1f}%)
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    ensure_dirs()
    df, summary = load_inputs()
    emotion_frequency, theme_frequency, emotion_cols, theme_cols = build_frequency_tables(df)
    conditional, emotion_corr = build_association_tables(df, emotion_cols, theme_cols)

    figures = [
        plot_theme_prevalence(theme_frequency),
        plot_emotion_prevalence(emotion_frequency),
        plot_theme_emotion_heatmap(conditional),
        plot_emotion_correlation_matrix(emotion_corr),
    ]
    report_path = write_report(df, summary, emotion_frequency, theme_frequency, figures)
    print(f"Saved selected-site summary to {report_path}")


if __name__ == "__main__":
    main()
