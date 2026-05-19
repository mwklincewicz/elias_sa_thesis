from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_EDA_DIR = Path(__file__).resolve().parent

SEGMENT_NAME = "BERT-Hyperparamter-Fixed"
BASELINE_DIR = PROJECT_ROOT / "output" / "bert_emotion_model"
THRESHOLD_DIR = BASELINE_DIR / "threshold_cardinality_test"
FIXED_DIR = PROJECT_ROOT / "output" / SEGMENT_NAME
FIGURES_DIR = FIXED_DIR / "figures"
COMPARISON_DIR = FIXED_DIR / "comparisons"

BASELINE_METRICS_PATH = BASELINE_DIR / "metrics.json"
BASELINE_PREDICTIONS_PATH = BASELINE_DIR / "test_predictions.csv"
THRESHOLD_METRICS_PATH = THRESHOLD_DIR / "metrics.json"
THRESHOLD_PREDICTIONS_PATH = THRESHOLD_DIR / "test_predictions_threshold_0_59_min1.csv"

FIXED_METRICS_PATH = FIXED_DIR / "metrics.json"
FIXED_PREDICTIONS_PATH = FIXED_DIR / "test_predictions.csv"
FIXED_RUN_SUMMARY_PATH = FIXED_DIR / "run_summary.txt"

MODEL_EDA_SCRIPTS = [
    "01_training_progress.py",
    "02_validation_vs_test_metrics.py",
    "03_per_emotion_f1.py",
    "04_precision_recall_scatter.py",
    "05_true_vs_predicted_label_counts.py",
    "06_sentiment_confusion_matrix.py",
    "07_example_prediction_table.py",
]

sns.set_theme(style="whitegrid")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_label_set(value: str) -> set[str]:
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "none":
        return set()
    return {part.strip() for part in cleaned.split(",") if part.strip()}


def count_label_string(value: str) -> int:
    return len(parse_label_set(value))


def compute_prediction_shape(predictions: pd.DataFrame) -> dict[str, float | int]:
    true_counts = predictions["true_emotions"].apply(count_label_string)
    predicted_counts = predictions["predicted_emotions"].apply(count_label_string)
    surplus = predicted_counts - true_counts
    return {
        "avg_true_labels_per_entry": float(true_counts.mean()),
        "avg_predicted_labels_per_entry": float(predicted_counts.mean()),
        "zero_predicted_rows": int((predicted_counts == 0).sum()),
        "overpredicted_by_at_least_2_rows": int((surplus >= 2).sum()),
        "overpredicted_any_rows": int((surplus > 0).sum()),
        "underpredicted_any_rows": int((surplus < 0).sum()),
    }


def compute_samples_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    precisions = []
    recalls = []
    f1_scores = []
    for row in predictions.itertuples(index=False):
        true_set = parse_label_set(row.true_emotions)
        predicted_set = parse_label_set(row.predicted_emotions)
        tp = len(true_set & predicted_set)
        fp = len(predicted_set - true_set)
        fn = len(true_set - predicted_set)

        precision_denominator = tp + fp
        recall_denominator = tp + fn
        f1_denominator = (2 * tp) + fp + fn
        precisions.append(tp / precision_denominator if precision_denominator else 0.0)
        recalls.append(tp / recall_denominator if recall_denominator else 0.0)
        f1_scores.append((2 * tp) / f1_denominator if f1_denominator else 0.0)

    return {
        "precision": float(pd.Series(precisions).mean()),
        "recall": float(pd.Series(recalls).mean()),
        "f1": float(pd.Series(f1_scores).mean()),
    }


def ensure_threshold_artifacts() -> None:
    if THRESHOLD_METRICS_PATH.exists() and THRESHOLD_PREDICTIONS_PATH.exists():
        return
    threshold_script = MODEL_EDA_DIR / "run_threshold_cardinality_test.py"
    result = subprocess.run([sys.executable, str(threshold_script)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit("Failed to build threshold/cardinality artifacts.")


def copy_baseline_support_files() -> None:
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    for filename in [
        "training_history.csv",
        "label_distribution.csv",
        "validation_threshold_scan.csv",
    ]:
        source = BASELINE_DIR / filename
        if source.exists():
            shutil.copy2(source, FIXED_DIR / filename)


def build_fixed_metrics(
    baseline_metrics: dict,
    threshold_metrics: dict,
    fixed_predictions: pd.DataFrame,
) -> dict:
    fixed_metrics = copy.deepcopy(baseline_metrics)
    fixed_test_metrics = copy.deepcopy(threshold_metrics["test_metrics"])
    fixed_test_metrics["samples"] = compute_samples_metrics(fixed_predictions)

    shape = compute_prediction_shape(fixed_predictions)
    fixed_test_metrics.update(shape)

    fixed_metrics.update(
        {
            "segment_name": SEGMENT_NAME,
            "annotation": "BERT hyperparameter-fixed inference segment",
            "model_weights": threshold_metrics.get("model_weights", "unchanged fine-tuned BERT"),
            "source_model_output_dir": str(BASELINE_DIR),
            "source_threshold_metrics_path": str(THRESHOLD_METRICS_PATH),
            "baseline_threshold": threshold_metrics.get("baseline_threshold"),
            "best_threshold": threshold_metrics["tested_threshold"],
            "tested_threshold": threshold_metrics["tested_threshold"],
            "minimum_predicted_labels": threshold_metrics["minimum_predicted_labels"],
            "test_metrics": fixed_test_metrics,
            "test_sentiment_metrics": threshold_metrics["test_sentiment_metrics"],
            "validation_metrics_note": (
                "Validation metrics are inherited from the baseline BERT training run; "
                "the fixed segment changes held-out-test inference only."
            ),
        }
    )
    return fixed_metrics


def metric_value(payload: dict, path: tuple[str, ...]) -> float | int | None:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return current
    return None


def build_overall_comparison_table(baseline_metrics: dict, fixed_metrics: dict) -> pd.DataFrame:
    metric_specs = [
        ("emotion_micro_precision", ("test_metrics", "micro", "precision")),
        ("emotion_micro_recall", ("test_metrics", "micro", "recall")),
        ("emotion_micro_f1", ("test_metrics", "micro", "f1")),
        ("emotion_macro_precision", ("test_metrics", "macro", "precision")),
        ("emotion_macro_recall", ("test_metrics", "macro", "recall")),
        ("emotion_macro_f1", ("test_metrics", "macro", "f1")),
        ("emotion_samples_f1", ("test_metrics", "samples", "f1")),
        ("emotion_subset_accuracy", ("test_metrics", "subset_accuracy")),
        ("emotion_hamming_loss", ("test_metrics", "hamming_loss")),
        ("avg_predicted_labels_per_entry", ("test_metrics", "avg_predicted_labels_per_entry")),
        ("zero_predicted_rows", ("test_metrics", "zero_predicted_rows")),
        ("overpredicted_by_at_least_2_rows", ("test_metrics", "overpredicted_by_at_least_2_rows")),
        ("sentiment_accuracy", ("test_sentiment_metrics", "accuracy")),
        ("sentiment_macro_f1", ("test_sentiment_metrics", "macro", "f1")),
    ]

    baseline_predictions = pd.read_csv(BASELINE_PREDICTIONS_PATH).fillna("")
    baseline_shape = compute_prediction_shape(baseline_predictions)
    enriched_baseline = copy.deepcopy(baseline_metrics)
    enriched_baseline["test_metrics"].update(
        {
            key: value
            for key, value in baseline_shape.items()
            if key not in enriched_baseline["test_metrics"]
        }
    )

    rows = []
    for metric_name, path in metric_specs:
        baseline_value = metric_value(enriched_baseline, path)
        fixed_value = metric_value(fixed_metrics, path)
        rows.append(
            {
                "metric": metric_name,
                "baseline_bert": baseline_value,
                SEGMENT_NAME: fixed_value,
                "delta_fixed_minus_baseline": (
                    None
                    if baseline_value is None or fixed_value is None
                    else float(fixed_value) - float(baseline_value)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_per_emotion_comparison_table(baseline_metrics: dict, fixed_metrics: dict) -> pd.DataFrame:
    baseline = pd.DataFrame(baseline_metrics["test_metrics"]["label_metrics"]).add_prefix("baseline_")
    fixed = pd.DataFrame(fixed_metrics["test_metrics"]["label_metrics"]).add_prefix("fixed_")
    frame = baseline.merge(fixed, left_on="baseline_label", right_on="fixed_label", how="outer")
    frame["label"] = frame["baseline_label"].fillna(frame["fixed_label"])
    frame["delta_f1_fixed_minus_baseline"] = frame["fixed_f1"] - frame["baseline_f1"]
    columns = [
        "label",
        "baseline_support",
        "fixed_support",
        "baseline_precision",
        "fixed_precision",
        "baseline_recall",
        "fixed_recall",
        "baseline_f1",
        "fixed_f1",
        "delta_f1_fixed_minus_baseline",
    ]
    return frame[columns].sort_values("label").reset_index(drop=True)


def build_prediction_cardinality_table(
    baseline_predictions: pd.DataFrame,
    fixed_predictions: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_predictions.copy()
    fixed = fixed_predictions.copy()
    baseline["true_count"] = baseline["true_emotions"].apply(count_label_string)
    baseline["predicted_count"] = baseline["predicted_emotions"].apply(count_label_string)
    fixed["predicted_count"] = fixed["predicted_emotions"].apply(count_label_string)

    count_axis = sorted(
        set(baseline["true_count"])
        | set(baseline["predicted_count"])
        | set(fixed["predicted_count"])
    )
    rows = []
    true_distribution = baseline["true_count"].value_counts()
    baseline_distribution = baseline["predicted_count"].value_counts()
    fixed_distribution = fixed["predicted_count"].value_counts()

    for count in count_axis:
        rows.append(
            {
                "label_count": int(count),
                "true_rows": int(true_distribution.get(count, 0)),
                "baseline_predicted_rows": int(baseline_distribution.get(count, 0)),
                "fixed_predicted_rows": int(fixed_distribution.get(count, 0)),
                "delta_fixed_minus_baseline": int(
                    fixed_distribution.get(count, 0) - baseline_distribution.get(count, 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def write_comparison_tables(baseline_metrics: dict, fixed_metrics: dict) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    baseline_predictions = pd.read_csv(BASELINE_PREDICTIONS_PATH).fillna("")
    fixed_predictions = pd.read_csv(FIXED_PREDICTIONS_PATH).fillna("")

    build_overall_comparison_table(baseline_metrics, fixed_metrics).to_csv(
        COMPARISON_DIR / "overall_metrics_comparison.csv",
        index=False,
    )
    build_per_emotion_comparison_table(baseline_metrics, fixed_metrics).to_csv(
        COMPARISON_DIR / "per_emotion_metrics_comparison.csv",
        index=False,
    )
    build_prediction_cardinality_table(baseline_predictions, fixed_predictions).to_csv(
        COMPARISON_DIR / "prediction_cardinality_comparison.csv",
        index=False,
    )


def annotate_bars(ax, decimals: int = 3) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        label = f"{height:.{decimals}f}" if abs(height) < 10 else f"{height:.0f}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
        )


def plot_overall_comparison_graph(overall: pd.DataFrame) -> None:
    display = {
        "emotion_micro_f1": "Emotion\nmicro-F1",
        "emotion_macro_f1": "Emotion\nmacro-F1",
        "emotion_hamming_loss": "Hamming\nloss",
        "emotion_subset_accuracy": "Exact\nmatch",
        "sentiment_accuracy": "Sentiment\naccuracy",
        "sentiment_macro_f1": "Sentiment\nmacro-F1",
    }
    score_frame = overall[overall["metric"].isin(display)].copy()
    score_frame["metric_label"] = score_frame["metric"].map(display)
    score_frame = score_frame.melt(
        id_vars=["metric", "metric_label"],
        value_vars=["baseline_bert", SEGMENT_NAME],
        var_name="run",
        value_name="value",
    )
    score_frame["run"] = score_frame["run"].replace(
        {"baseline_bert": "Baseline BERT", SEGMENT_NAME: "Hyperparameter-fixed"}
    )

    avg_frame = overall[overall["metric"] == "avg_predicted_labels_per_entry"].melt(
        id_vars=["metric"],
        value_vars=["baseline_bert", SEGMENT_NAME],
        var_name="run",
        value_name="value",
    )
    avg_frame["run"] = avg_frame["run"].replace(
        {"baseline_bert": "Baseline BERT", SEGMENT_NAME: "Hyperparameter-fixed"}
    )

    count_display = {
        "zero_predicted_rows": "Zero predicted\nemotions",
        "overpredicted_by_at_least_2_rows": ">=2 extra\nemotions",
    }
    count_frame = overall[overall["metric"].isin(count_display)].copy()
    count_frame["metric_label"] = count_frame["metric"].map(count_display)
    count_frame = count_frame.melt(
        id_vars=["metric", "metric_label"],
        value_vars=["baseline_bert", SEGMENT_NAME],
        var_name="run",
        value_name="value",
    )
    count_frame["run"] = count_frame["run"].replace(
        {"baseline_bert": "Baseline BERT", SEGMENT_NAME: "Hyperparameter-fixed"}
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.5, 5.2),
        gridspec_kw={"width_ratios": [2.3, 0.9, 1.2]},
    )

    palette = ["#4C72B0", "#55A868"]
    sns.barplot(
        data=score_frame,
        x="metric_label",
        y="value",
        hue="run",
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Held-out test scores")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(title="")
    annotate_bars(axes[0])

    sns.barplot(
        data=avg_frame,
        x="run",
        y="value",
        hue="run",
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Prediction density")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Avg predicted emotions")
    axes[1].tick_params(axis="x", rotation=15)
    annotate_bars(axes[1])

    sns.barplot(
        data=count_frame,
        x="metric_label",
        y="value",
        hue="run",
        palette=palette,
        ax=axes[2],
    )
    axes[2].set_title("Overestimation checks")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Rows")
    axes[2].legend(title="")
    annotate_bars(axes[2], decimals=0)

    fig.suptitle("Baseline BERT vs BERT-Hyperparamter-Fixed", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(COMPARISON_DIR / "fig_compare_overall_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_emotion_comparison_graph(per_emotion: pd.DataFrame) -> None:
    frame = per_emotion.sort_values("delta_f1_fixed_minus_baseline").reset_index(drop=True)
    y_positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(10.5, max(6.0, 0.42 * len(frame))))
    for y, row in zip(y_positions, frame.itertuples(index=False)):
        color = "#55A868" if row.delta_f1_fixed_minus_baseline >= 0 else "#C44E52"
        ax.plot(
            [row.baseline_f1, row.fixed_f1],
            [y, y],
            color=color,
            linewidth=2,
            alpha=0.75,
        )

    ax.scatter(frame["baseline_f1"], y_positions, color="#4C72B0", s=64, label="Baseline BERT", zorder=3)
    ax.scatter(frame["fixed_f1"], y_positions, color="#55A868", s=64, label="Hyperparameter-fixed", zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["label"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_title("Per-emotion F1 shift after hyperparameter-fixed inference")
    ax.legend(loc="lower right")

    for y, row in zip(y_positions, frame.itertuples(index=False)):
        ax.text(
            1.01,
            y,
            f"{row.delta_f1_fixed_minus_baseline:+.3f}",
            va="center",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
            color="#2A6F3F" if row.delta_f1_fixed_minus_baseline >= 0 else "#9A332F",
        )
    ax.text(1.01, 1.02, "Delta", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "fig_compare_per_emotion_f1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_cardinality_graph(cardinality: pd.DataFrame) -> None:
    frame = cardinality.copy()
    x = np.arange(len(frame))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.bar(
        x - width / 2,
        frame["baseline_predicted_rows"],
        width=width,
        label="Baseline predicted",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        frame["fixed_predicted_rows"],
        width=width,
        label="Hyperparameter-fixed predicted",
        color="#55A868",
    )
    ax.plot(
        x,
        frame["true_rows"],
        color="#222222",
        linewidth=2,
        marker="o",
        label="True label count",
    )

    ax.set_title("Predicted emotion-label counts move closer to the true distribution")
    ax.set_xlabel("Emotion labels per reflection")
    ax.set_ylabel("Rows")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label_count"].astype(str))
    ax.legend()

    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "fig_compare_prediction_cardinality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_comparison_graphs() -> None:
    overall = pd.read_csv(COMPARISON_DIR / "overall_metrics_comparison.csv")
    per_emotion = pd.read_csv(COMPARISON_DIR / "per_emotion_metrics_comparison.csv")
    cardinality = pd.read_csv(COMPARISON_DIR / "prediction_cardinality_comparison.csv")

    plot_overall_comparison_graph(overall)
    plot_per_emotion_comparison_graph(per_emotion)
    plot_prediction_cardinality_graph(cardinality)


def build_summary_text(baseline_metrics: dict, fixed_metrics: dict) -> str:
    baseline_shape = compute_prediction_shape(pd.read_csv(BASELINE_PREDICTIONS_PATH).fillna(""))
    fixed_shape = fixed_metrics["test_metrics"]
    return (
        f"{SEGMENT_NAME}\n"
        "==========================\n"
        "Source model: output/bert_emotion_model\n"
        f"Model weights: {fixed_metrics['model_weights']}\n"
        f"Baseline threshold: {fixed_metrics.get('baseline_threshold')}\n"
        f"Fixed threshold: {fixed_metrics['tested_threshold']}\n"
        f"Minimum predicted labels: {fixed_metrics['minimum_predicted_labels']}\n\n"
        "Held-out test comparison\n"
        "------------------------\n"
        f"Micro F1: baseline {baseline_metrics['test_metrics']['micro']['f1']:.4f} "
        f"vs fixed {fixed_metrics['test_metrics']['micro']['f1']:.4f}\n"
        f"Macro F1: baseline {baseline_metrics['test_metrics']['macro']['f1']:.4f} "
        f"vs fixed {fixed_metrics['test_metrics']['macro']['f1']:.4f}\n"
        f"Sentiment accuracy: baseline {baseline_metrics['test_sentiment_metrics']['accuracy']:.4f} "
        f"vs fixed {fixed_metrics['test_sentiment_metrics']['accuracy']:.4f}\n"
        f"Average predicted labels: baseline {baseline_shape['avg_predicted_labels_per_entry']:.4f} "
        f"vs fixed {fixed_shape['avg_predicted_labels_per_entry']:.4f}\n"
        f"Rows with zero predicted emotions: baseline {baseline_shape['zero_predicted_rows']} "
        f"vs fixed {fixed_shape['zero_predicted_rows']}\n"
        f"Rows with at least two extra predicted emotions: "
        f"baseline {baseline_shape['overpredicted_by_at_least_2_rows']} "
        f"vs fixed {fixed_shape['overpredicted_by_at_least_2_rows']}\n\n"
        "Comparison graphs\n"
        "-----------------\n"
        "fig_compare_overall_metrics.png\n"
        "fig_compare_per_emotion_f1.png\n"
        "fig_compare_prediction_cardinality.png\n\n"
        "This segment is reproducible from the baseline BERT predictions plus the "
        "threshold/cardinality inference script. It does not retrain BERT weights.\n"
    )


def run_standard_figures() -> None:
    env = os.environ.copy()
    env["LEMOTIF_MODEL_OUTPUT_DIR"] = str(FIXED_DIR)
    env["LEMOTIF_MODEL_FIGURES_DIR"] = str(FIGURES_DIR)

    for script_name in MODEL_EDA_SCRIPTS:
        script_path = MODEL_EDA_DIR / script_name
        print(f"Running fixed-segment figure script: {script_path}")
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            raise SystemExit(f"Failed while generating {script_name} for {SEGMENT_NAME}")


def main() -> None:
    if not BASELINE_METRICS_PATH.exists() or not BASELINE_PREDICTIONS_PATH.exists():
        raise SystemExit(
            "Baseline BERT artifacts are missing. Run the bert-baseline task first."
        )

    ensure_threshold_artifacts()
    copy_baseline_support_files()

    baseline_metrics = load_json(BASELINE_METRICS_PATH)
    threshold_metrics = load_json(THRESHOLD_METRICS_PATH)

    fixed_predictions = pd.read_csv(THRESHOLD_PREDICTIONS_PATH).fillna("")
    fixed_predictions.to_csv(FIXED_PREDICTIONS_PATH, index=False)

    fixed_metrics = build_fixed_metrics(baseline_metrics, threshold_metrics, fixed_predictions)
    write_json(FIXED_METRICS_PATH, fixed_metrics)
    FIXED_RUN_SUMMARY_PATH.write_text(
        build_summary_text(baseline_metrics, fixed_metrics),
        encoding="utf-8",
    )

    write_comparison_tables(baseline_metrics, fixed_metrics)
    write_comparison_graphs()
    run_standard_figures()

    print(f"Saved fixed BERT segment to: {FIXED_DIR}")
    print(f"Saved 7 standard figures to: {FIGURES_DIR}")
    print(f"Saved 3 comparison tables and 3 comparison graphs to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
