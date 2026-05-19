from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "output" / "bert_emotion_model"
OPTUNA_DIR = PROJECT_ROOT / "output" / "bert_emotion_model_optuna"
COMPARISON_DIR = OPTUNA_DIR / "comparisons"

BASELINE_METRICS_PATH = BASELINE_DIR / "metrics.json"
OPTUNA_METRICS_PATH = OPTUNA_DIR / "metrics.json"
STUDY_TRIALS_PATH = OPTUNA_DIR / "study_trials.csv"
PARAM_IMPORTANCE_PATH = OPTUNA_DIR / "parameter_importances.csv"

SUMMARY_CSV_PATH = COMPARISON_DIR / "metrics_comparison.csv"
HYPERPARAMETER_CSV_PATH = COMPARISON_DIR / "hyperparameter_comparison.csv"
SUMMARY_TEXT_PATH = COMPARISON_DIR / "comparison_summary.txt"

sns.set_theme(style="whitegrid")
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig, destination: Path, dpi: int = 220) -> None:
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_metric_comparison_frame(baseline_metrics: dict, optuna_metrics: dict) -> pd.DataFrame:
    rows = []
    metric_map = [
        ("Validation", "Emotion micro F1", baseline_metrics["validation_metrics"]["micro"]["f1"], optuna_metrics["validation_metrics"]["micro"]["f1"]),
        ("Validation", "Emotion macro F1", baseline_metrics["validation_metrics"]["macro"]["f1"], optuna_metrics["validation_metrics"]["macro"]["f1"]),
        ("Validation", "Sentiment accuracy", baseline_metrics["validation_sentiment_metrics"]["accuracy"], optuna_metrics["validation_sentiment_metrics"]["accuracy"]),
        ("Validation", "Sentiment macro F1", baseline_metrics["validation_sentiment_metrics"]["macro"]["f1"], optuna_metrics["validation_sentiment_metrics"]["macro"]["f1"]),
        ("Test", "Emotion micro F1", baseline_metrics["test_metrics"]["micro"]["f1"], optuna_metrics["test_metrics"]["micro"]["f1"]),
        ("Test", "Emotion macro F1", baseline_metrics["test_metrics"]["macro"]["f1"], optuna_metrics["test_metrics"]["macro"]["f1"]),
        ("Test", "Sentiment accuracy", baseline_metrics["test_sentiment_metrics"]["accuracy"], optuna_metrics["test_sentiment_metrics"]["accuracy"]),
        ("Test", "Sentiment macro F1", baseline_metrics["test_sentiment_metrics"]["macro"]["f1"], optuna_metrics["test_sentiment_metrics"]["macro"]["f1"]),
    ]
    for split, metric, baseline_value, optuna_value in metric_map:
        rows.append({"split": split, "metric": metric, "model": "Baseline", "value": baseline_value})
        rows.append({"split": split, "metric": metric, "model": "Optuna", "value": optuna_value})
    return pd.DataFrame(rows)


def normalize_hyperparameter_value(name: str, value: float | int, spec: dict[str, object]) -> float:
    spec_type = str(spec["type"])
    if spec_type == "categorical":
        choices = list(spec["choices"])
        if len(choices) == 1:
            return 1.0
        return choices.index(value) / (len(choices) - 1)
    low = float(spec["low"])
    high = float(spec["high"])
    if high == low:
        return 1.0
    if spec_type == "float_log":
        log_low = np.log10(low)
        log_high = np.log10(high)
        return float((np.log10(float(value)) - log_low) / (log_high - log_low))
    return float((float(value) - low) / (high - low))


def build_hyperparameter_frame(optuna_metrics: dict) -> pd.DataFrame:
    defaults = optuna_metrics["default_hyperparameters"]
    selected = optuna_metrics["selected_hyperparameters"]
    search_space = optuna_metrics["search_space"]
    rows = []

    for name, spec in search_space.items():
        default_value = defaults[name]
        optuna_value = selected[name]
        rows.append(
            {
                "parameter": name,
                "model": "Baseline",
                "value": default_value,
                "normalized_value": normalize_hyperparameter_value(name, default_value, spec),
            }
        )
        rows.append(
            {
                "parameter": name,
                "model": "Optuna",
                "value": optuna_value,
                "normalized_value": normalize_hyperparameter_value(name, optuna_value, spec),
            }
        )

    return pd.DataFrame(rows)


def build_label_comparison_frame(
    baseline_metrics: dict,
    optuna_metrics: dict,
    split_key: str,
) -> pd.DataFrame:
    baseline_frame = pd.DataFrame(baseline_metrics[split_key]["label_metrics"])
    baseline_frame["model"] = "Baseline"
    optuna_frame = pd.DataFrame(optuna_metrics[split_key]["label_metrics"])
    optuna_frame["model"] = "Optuna"
    frame = pd.concat([baseline_frame, optuna_frame], ignore_index=True)
    return frame


def plot_hyperparameter_choices(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    sns.barplot(
        data=frame,
        x="parameter",
        y="normalized_value",
        hue="model",
        palette=["#4C72B0", "#DD8452"],
        ax=ax,
    )
    ax.set_title("Manual defaults vs Optuna-selected hyperparameters")
    ax.set_xlabel("")
    ax.set_ylabel("Position within search space")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")

    for patch, (_, row) in zip(ax.patches, frame.iterrows()):
        ax.annotate(
            f"{row['value']}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            textcoords="offset points",
            xytext=(0, 4),
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_hyperparameters.png")


def plot_overall_metrics(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for ax, split in zip(axes, ["Validation", "Test"]):
        subset = frame[frame["split"] == split]
        sns.barplot(data=subset, x="metric", y="value", hue="model", palette=["#4C72B0", "#DD8452"], ax=ax)
        ax.set_title(f"{split} metrics")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_overall_metrics.png")


def plot_per_emotion_f1(frame: pd.DataFrame) -> None:
    order = (
        frame[frame["model"] == "Optuna"]
        .sort_values("f1", ascending=True)["label"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(10.8, max(5.6, 0.4 * len(order))))
    sns.barplot(
        data=frame,
        x="f1",
        y="label",
        hue="model",
        order=order,
        palette=["#4C72B0", "#DD8452"],
        ax=ax,
    )
    ax.set_title("Per-emotion test F1: baseline vs Optuna")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)
    ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_per_emotion_f1.png")


def plot_sentiment_metrics(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=frame, x="label", y="f1", hue="model", palette=["#4C72B0", "#DD8452"], ax=ax)
    ax.set_title("Per-sentiment F1: baseline vs Optuna")
    ax.set_xlabel("")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_compare_sentiment_f1.png")


def plot_optimization_history(trials: pd.DataFrame) -> None:
    completed = trials[trials["state"] == "COMPLETE"].sort_values("trial_number").copy()
    completed["best_so_far"] = completed["objective_value"].cummax()

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.plot(completed["trial_number"], completed["objective_value"], marker="o", linewidth=1.6, label="Trial score")
    ax.plot(completed["trial_number"], completed["best_so_far"], linewidth=2.2, linestyle="--", label="Best so far")
    ax.set_title("Optuna validation micro-F1 across trials")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation micro-F1")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_optuna_optimization_history.png")


def plot_parameter_importance(path: Path) -> None:
    frame = pd.read_csv(path).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, max(4.4, 0.55 * len(frame))))
    sns.barplot(data=frame, x="importance", y="parameter", color="#55A868", ax=ax)
    ax.set_title("Optuna parameter importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")

    fig.tight_layout()
    save_figure(fig, COMPARISON_DIR / "fig_optuna_parameter_importance.png")


def build_summary_text(baseline_metrics: dict, optuna_metrics: dict) -> str:
    return (
        "Baseline vs Optuna-tuned BERT\n"
        "=============================\n"
        f"Validation micro F1: baseline {baseline_metrics['validation_metrics']['micro']['f1']:.4f} "
        f"vs Optuna {optuna_metrics['validation_metrics']['micro']['f1']:.4f}\n"
        f"Test micro F1: baseline {baseline_metrics['test_metrics']['micro']['f1']:.4f} "
        f"vs Optuna {optuna_metrics['test_metrics']['micro']['f1']:.4f}\n"
        f"Test macro F1: baseline {baseline_metrics['test_metrics']['macro']['f1']:.4f} "
        f"vs Optuna {optuna_metrics['test_metrics']['macro']['f1']:.4f}\n"
        f"Test sentiment accuracy: baseline {baseline_metrics['test_sentiment_metrics']['accuracy']:.4f} "
        f"vs Optuna {optuna_metrics['test_sentiment_metrics']['accuracy']:.4f}\n"
    )


def main() -> None:
    baseline_metrics = load_json(BASELINE_METRICS_PATH)
    optuna_metrics = load_json(OPTUNA_METRICS_PATH)
    trials = pd.read_csv(STUDY_TRIALS_PATH)

    metric_frame = build_metric_comparison_frame(baseline_metrics, optuna_metrics)
    hyperparameter_frame = build_hyperparameter_frame(optuna_metrics)
    per_emotion_frame = build_label_comparison_frame(baseline_metrics, optuna_metrics, "test_metrics")
    per_sentiment_frame = build_label_comparison_frame(
        baseline_metrics,
        optuna_metrics,
        "test_sentiment_metrics",
    )

    metric_frame.to_csv(SUMMARY_CSV_PATH, index=False)
    hyperparameter_frame.to_csv(HYPERPARAMETER_CSV_PATH, index=False)
    SUMMARY_TEXT_PATH.write_text(build_summary_text(baseline_metrics, optuna_metrics), encoding="utf-8")

    plot_hyperparameter_choices(hyperparameter_frame)
    plot_overall_metrics(metric_frame)
    plot_per_emotion_f1(per_emotion_frame)
    plot_sentiment_metrics(per_sentiment_frame)
    plot_optimization_history(trials)

    if PARAM_IMPORTANCE_PATH.exists():
        plot_parameter_importance(PARAM_IMPORTANCE_PATH)

    print(f"Saved comparison summary to: {SUMMARY_CSV_PATH}")
    print(f"Saved hyperparameter comparison to: {HYPERPARAMETER_CSV_PATH}")
    print(f"Saved comparison figures to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
