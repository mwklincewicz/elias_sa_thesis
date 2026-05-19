from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent if PROJECT_ROOT.parent.name == "thesis_findings_vscode" else PROJECT_ROOT

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name, load_analysis_data, safe_name
from train_bert import (
    BATCH_SIZE,
    MAX_LENGTH,
    SEED,
    LemotifEmotionDataset,
    derive_sentiment,
    split_indices,
    run_epoch,
    set_seed,
)


MODEL_DISPLAY_NAME = "BERT Optuna"
SEGMENT_NAME = "bert_emotion_model_optuna"
OPTUNA_THRESHOLD = 0.50
MIN_PREDICTED_LABELS = 0

OPTUNA_DIR = PROJECT_ROOT / "output" / SEGMENT_NAME
OPTUNA_MODEL_DIR = OPTUNA_DIR / "model"
if not OPTUNA_MODEL_DIR.exists():
    outer_model_dir = WORKSPACE_ROOT / "output" / SEGMENT_NAME / "model"
    if outer_model_dir.exists():
        OPTUNA_MODEL_DIR = outer_model_dir
DIAGNOSTICS_DIR = OPTUNA_DIR / "diagnostics"

OPTUNA_METRICS_PATH = OPTUNA_DIR / "metrics.json"
OPTUNA_PREDICTIONS_PATH = OPTUNA_DIR / "test_predictions.csv"

BOOTSTRAP_DIR = DIAGNOSTICS_DIR / "01_bootstrap_ci"
RARE_LABEL_DIR = DIAGNOSTICS_DIR / "02_rare_label_errors"
MULTILABEL_DIR = DIAGNOSTICS_DIR / "03_multilabel_errors"
THRESHOLD_DIR = DIAGNOSTICS_DIR / "04_threshold_calibration"
ABLATION_DIR = DIAGNOSTICS_DIR / "05_feature_ablation"
LIME_DIR = DIAGNOSTICS_DIR / "06_lime"
COMPARATIVE_DIR = DIAGNOSTICS_DIR / "comparative_statistics"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]

EMOTION_LEXICON = {
    "afraid": ["afraid", "fear", "scared", "terrified", "worried", "unsafe"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated", "rage"],
    "anxious": ["anxious", "anxiety", "nervous", "stress", "stressed", "tense"],
    "ashamed": ["ashamed", "shame", "guilty", "embarrassed", "regret"],
    "awkward": ["awkward", "uncomfortable", "weird", "strange", "embarrassing"],
    "bored": ["bored", "boring", "dull", "tedious", "empty"],
    "calm": ["calm", "peaceful", "relaxed", "ease", "quiet", "settled"],
    "confused": ["confused", "unsure", "uncertain", "lost", "unclear"],
    "disgusted": ["disgusted", "gross", "nasty", "repulsed", "sick"],
    "excited": ["excited", "thrilled", "eager", "fun", "amazing", "great"],
    "frustrated": ["frustrated", "frustrating", "stuck", "blocked", "annoying"],
    "happy": ["happy", "joy", "glad", "smile", "delighted", "cheerful"],
    "jealous": ["jealous", "envy", "envious", "resentful"],
    "nostalgic": ["nostalgic", "memory", "remember", "past", "miss", "childhood"],
    "proud": ["proud", "accomplished", "achievement", "success", "confident"],
    "sad": ["sad", "upset", "cry", "lonely", "down", "depressed"],
    "satisfied": ["satisfied", "content", "fulfilled", "pleased", "relieved"],
    "surprised": ["surprised", "unexpected", "shocked", "sudden", "amazed"],
}

sns.set_theme(style="whitegrid")


def ensure_directories() -> None:
    for path in [
        BOOTSTRAP_DIR,
        RARE_LABEL_DIR,
        MULTILABEL_DIR,
        THRESHOLD_DIR,
        ABLATION_DIR,
        LIME_DIR,
        COMPARATIVE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def save_figure(fig: plt.Figure, path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_label_set(value: object) -> set[str]:
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "none":
        return set()
    return {part.strip() for part in cleaned.split(",") if part.strip()}


def format_active_labels(binary_labels: np.ndarray, label_names: list[str]) -> str:
    labels = [label for label, active in zip(label_names, binary_labels) if int(active) == 1]
    return ", ".join(labels) if labels else "none"


def labels_to_matrix(values: pd.Series, label_names: list[str]) -> np.ndarray:
    rows = []
    for value in values:
        active = parse_label_set(value)
        rows.append([1 if label in active else 0 for label in label_names])
    return np.array(rows, dtype=int)


def probability_matrix_from_predictions(predictions: pd.DataFrame, label_names: list[str]) -> np.ndarray:
    columns = [f"prob_{safe_name(label.lower())}" for label in label_names]
    missing = [column for column in columns if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    return predictions[columns].to_numpy(dtype=float)


def apply_threshold_with_min_labels(
    probabilities: np.ndarray,
    threshold: float,
    min_labels: int = MIN_PREDICTED_LABELS,
) -> np.ndarray:
    y_pred = (probabilities >= threshold).astype(int)
    if min_labels <= 0:
        return y_pred

    for row_idx in range(y_pred.shape[0]):
        active_count = int(y_pred[row_idx].sum())
        if active_count >= min_labels:
            continue
        ranked = np.argsort(probabilities[row_idx])[::-1]
        needed = min_labels - active_count
        for label_idx in ranked:
            if y_pred[row_idx, label_idx] == 0:
                y_pred[row_idx, label_idx] = 1
                needed -= 1
            if needed == 0:
                break
    return y_pred


def sentiment_array(y_matrix: np.ndarray, label_names: list[str]) -> np.ndarray:
    return np.array([derive_sentiment(row, label_names)[0] for row in y_matrix], dtype=object)


def metrics_from_binary(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, object]:
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    samples_precision, samples_recall, samples_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", zero_division=0
    )
    label_precision, label_recall, label_f1, label_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "micro": {
            "precision": float(micro_precision),
            "recall": float(micro_recall),
            "f1": float(micro_f1),
        },
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "samples": {
            "precision": float(samples_precision),
            "recall": float(samples_recall),
            "f1": float(samples_f1),
        },
        "subset_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(np.not_equal(y_true, y_pred).mean()),
        "avg_true_labels_per_entry": float(y_true.sum(axis=1).mean()),
        "avg_predicted_labels_per_entry": float(y_pred.sum(axis=1).mean()),
        "label_metrics": [
            {
                "label": label,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
            for label, precision, recall, f1, support in zip(
                label_names,
                label_precision,
                label_recall,
                label_f1,
                label_support,
            )
        ],
    }


def sentiment_metrics_from_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, object]:
    true_sentiments = sentiment_array(y_true, label_names)
    predicted_sentiments = sentiment_array(y_pred, label_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_sentiments,
        predicted_sentiments,
        labels=SENTIMENT_ORDER,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_sentiments,
        predicted_sentiments,
        labels=SENTIMENT_ORDER,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float((true_sentiments == predicted_sentiments).mean()),
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "label_metrics": [
            {
                "label": label,
                "precision": float(label_precision),
                "recall": float(label_recall),
                "f1": float(label_f1),
                "support": int(label_support),
            }
            for label, label_precision, label_recall, label_f1, label_support in zip(
                SENTIMENT_ORDER,
                precision,
                recall,
                f1,
                support,
            )
        ],
    }


def prediction_frame_from_binary(
    source_df: pd.DataFrame,
    row_ids: list[int],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    text_col: str,
) -> pd.DataFrame:
    rows = []
    for position, row_id in enumerate(row_ids):
        true_sentiment, true_score = derive_sentiment(y_true[position], label_names)
        predicted_sentiment, predicted_score = derive_sentiment(y_pred[position], label_names)
        row = {
            "row_id": int(row_id),
            "text": source_df.iloc[row_id][text_col],
            "true_emotions": format_active_labels(y_true[position], label_names),
            "predicted_emotions": format_active_labels(y_pred[position], label_names),
            "true_sentiment": true_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "true_sentiment_score": true_score,
            "predicted_sentiment_score": predicted_score,
            "true_label_count": int(y_true[position].sum()),
            "predicted_label_count": int(y_pred[position].sum()),
        }
        for label_idx, label in enumerate(label_names):
            row[f"prob_{safe_name(label.lower())}"] = float(y_prob[position, label_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_overall_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    emotion = metrics_from_binary(y_true, y_pred, label_names)
    sentiment = sentiment_metrics_from_binary(y_true, y_pred, label_names)
    return {
        "emotion_micro_f1": float(emotion["micro"]["f1"]),
        "emotion_macro_f1": float(emotion["macro"]["f1"]),
        "emotion_samples_f1": float(emotion["samples"]["f1"]),
        "subset_accuracy": float(emotion["subset_accuracy"]),
        "hamming_loss": float(emotion["hamming_loss"]),
        "sentiment_accuracy": float(sentiment["accuracy"]),
        "sentiment_macro_f1": float(sentiment["macro"]["f1"]),
    }


def bootstrap_ci(
    values: list[dict[str, float]],
    estimates: dict[str, float],
    model_name: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(values)
    rows = []
    for metric in frame.columns:
        sample_values = frame[metric].dropna().to_numpy(dtype=float)
        if sample_values.size == 0:
            lower = upper = float("nan")
        else:
            lower, upper = np.quantile(sample_values, [0.025, 0.975])
        row = {
            "metric": metric,
            "estimate": estimates.get(metric, float("nan")),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }
        if model_name is not None:
            row["model"] = model_name
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_metric_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_rows = y_true.shape[0]
    overall_samples: list[dict[str, float]] = []
    per_label_samples: list[dict[str, float]] = []

    for _ in range(iterations):
        idx = rng.integers(0, n_rows, size=n_rows)
        sampled_true = y_true[idx]
        sampled_pred = y_pred[idx]
        overall_samples.append(compute_overall_scores(sampled_true, sampled_pred, label_names))
        _, _, f1, _ = precision_recall_fscore_support(
            sampled_true,
            sampled_pred,
            average=None,
            zero_division=0,
        )
        per_label_samples.append({label: float(value) for label, value in zip(label_names, f1)})

    return pd.DataFrame(overall_samples), pd.DataFrame(per_label_samples)


def plot_ci_forest(frame: pd.DataFrame, path: Path, title: str, value_label: str = "Score") -> None:
    plot_frame = frame.copy()
    plot_frame["metric_label"] = plot_frame["metric"].str.replace("_", " ").str.title()
    plot_frame = plot_frame.sort_values("estimate")

    fig, ax = plt.subplots(figsize=(9.4, max(4.8, 0.45 * len(plot_frame))))
    y_pos = np.arange(len(plot_frame))
    lower = plot_frame["estimate"] - plot_frame["ci_lower"]
    upper = plot_frame["ci_upper"] - plot_frame["estimate"]
    ax.errorbar(
        plot_frame["estimate"],
        y_pos,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#2F6FB0",
        ecolor="#555555",
        elinewidth=1.5,
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_frame["metric_label"])
    ax.set_xlabel(value_label)
    ax.set_title(title)
    ax.set_xlim(
        min(0.0, float(plot_frame["ci_lower"].min()) - 0.05),
        max(1.0, float(plot_frame["ci_upper"].max()) + 0.05),
    )
    fig.tight_layout()
    save_figure(fig, path)


def run_bootstrap_ci(
    predictions: pd.DataFrame,
    label_names: list[str],
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = labels_to_matrix(predictions["true_emotions"], label_names)
    y_pred = labels_to_matrix(predictions["predicted_emotions"], label_names)
    probabilities = probability_matrix_from_predictions(predictions, label_names)

    overall_samples, per_label_samples = bootstrap_metric_samples(
        y_true,
        y_pred,
        label_names,
        iterations=iterations,
        seed=SEED,
    )
    estimates = compute_overall_scores(y_true, y_pred, label_names)
    overall_ci = bootstrap_ci(overall_samples.to_dict("records"), estimates)

    label_metrics = pd.DataFrame(metrics_from_binary(y_true, y_pred, label_names)["label_metrics"])
    per_label_estimates = dict(zip(label_metrics["label"], label_metrics["f1"]))
    per_label_ci = bootstrap_ci(per_label_samples.to_dict("records"), per_label_estimates)
    per_label_ci = per_label_ci.rename(columns={"metric": "label", "estimate": "f1"})

    overall_samples.to_csv(BOOTSTRAP_DIR / "bootstrap_overall_samples.csv", index=False)
    per_label_samples.to_csv(BOOTSTRAP_DIR / "bootstrap_per_emotion_f1_samples.csv", index=False)
    overall_ci.to_csv(BOOTSTRAP_DIR / "bootstrap_overall_ci.csv", index=False)
    per_label_ci.to_csv(BOOTSTRAP_DIR / "bootstrap_per_emotion_f1_ci.csv", index=False)

    plot_ci_forest(
        overall_ci,
        BOOTSTRAP_DIR / "fig_bootstrap_metric_ci.png",
        "BERT Optuna metric bootstrap intervals",
    )

    plot_label_frame = per_label_ci.copy()
    plot_label_frame["metric"] = plot_label_frame["label"]
    plot_label_frame["estimate"] = plot_label_frame["f1"]
    plot_ci_forest(
        plot_label_frame,
        BOOTSTRAP_DIR / "fig_bootstrap_per_emotion_f1_ci.png",
        "BERT Optuna per-emotion F1 bootstrap intervals",
        value_label="F1 score",
    )

    return y_true, y_pred, probabilities


def run_rare_label_analysis(
    predictions: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> None:
    rows = []
    example_rows = []
    rare_label_indices = []

    for label_idx, label in enumerate(label_names):
        true_col = y_true[:, label_idx]
        pred_col = y_pred[:, label_idx]
        tp = int(((true_col == 1) & (pred_col == 1)).sum())
        fp = int(((true_col == 0) & (pred_col == 1)).sum())
        fn = int(((true_col == 1) & (pred_col == 0)).sum())
        support = int(true_col.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * tp) / ((2 * tp) + fp + fn) if (2 * tp) + fp + fn else 0.0
        is_rare = support <= 10
        if is_rare:
            rare_label_indices.append(label_idx)
        rows.append(
            {
                "label": label,
                "support": support,
                "is_rare": is_rare,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "error_rate_per_test_row": (fp + fn) / len(predictions),
            }
        )

        if not is_rare:
            continue

        false_negative_idx = np.where((true_col == 1) & (pred_col == 0))[0]
        false_positive_idx = np.where((true_col == 0) & (pred_col == 1))[0]
        ranked_fn = sorted(false_negative_idx, key=lambda idx: probabilities[idx, label_idx], reverse=True)[:3]
        ranked_fp = sorted(false_positive_idx, key=lambda idx: probabilities[idx, label_idx], reverse=True)[:3]
        for error_type, selected in [("false_negative", ranked_fn), ("false_positive", ranked_fp)]:
            for idx in selected:
                row = predictions.iloc[idx]
                example_rows.append(
                    {
                        "label": label,
                        "error_type": error_type,
                        "row_id": int(row["row_id"]),
                        "probability": float(probabilities[idx, label_idx]),
                        "true_emotions": row["true_emotions"],
                        "predicted_emotions": row["predicted_emotions"],
                        "text": row["text"],
                    }
                )

    summary = pd.DataFrame(rows)
    rare_summary = summary[summary["is_rare"]].copy()
    examples = pd.DataFrame(example_rows)
    summary.to_csv(RARE_LABEL_DIR / "label_error_summary.csv", index=False)
    rare_summary.to_csv(RARE_LABEL_DIR / "rare_label_error_summary.csv", index=False)
    examples.to_csv(RARE_LABEL_DIR / "rare_label_error_examples.csv", index=False)

    plot_counts = rare_summary.melt(
        id_vars=["label", "support"],
        value_vars=["false_positive", "false_negative"],
        var_name="error_type",
        value_name="count",
    )
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.42 * len(rare_summary))))
    sns.barplot(data=plot_counts, y="label", x="count", hue="error_type", palette=["#C44E52", "#4C72B0"], ax=ax)
    ax.set_title("Rare-label false positives and false negatives")
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    ax.legend(title="")
    fig.tight_layout()
    save_figure(fig, RARE_LABEL_DIR / "fig_rare_label_fp_fn.png")

    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.42 * len(rare_summary))))
    rare_plot = rare_summary.sort_values("error_rate_per_test_row", ascending=True)
    ax.barh(rare_plot["label"], rare_plot["error_rate_per_test_row"], color="#DD8452")
    ax.set_title("Rare-label error rate per held-out row")
    ax.set_xlabel("(false positives + false negatives) / test rows")
    ax.set_ylabel("")
    for index, row in rare_plot.reset_index(drop=True).iterrows():
        ax.text(
            row["error_rate_per_test_row"] + 0.003,
            index,
            f"n={int(row['support'])}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, RARE_LABEL_DIR / "fig_rare_label_error_rate.png")


def cooccurrence_matrix(y_matrix: np.ndarray) -> np.ndarray:
    matrix = y_matrix.T @ y_matrix
    np.fill_diagonal(matrix, 0)
    return matrix


def run_multilabel_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> None:
    flow = np.zeros((len(label_names), len(label_names)), dtype=int)
    for true_row, pred_row in zip(y_true, y_pred):
        false_negative = np.where((true_row == 1) & (pred_row == 0))[0]
        false_positive = np.where((true_row == 0) & (pred_row == 1))[0]
        for fn_idx in false_negative:
            for fp_idx in false_positive:
                flow[fn_idx, fp_idx] += 1

    true_cooccurrence = cooccurrence_matrix(y_true)
    pred_cooccurrence = cooccurrence_matrix(y_pred)
    cooccurrence_delta = pred_cooccurrence - true_cooccurrence

    flow_df = pd.DataFrame(flow, index=label_names, columns=label_names)
    true_df = pd.DataFrame(true_cooccurrence, index=label_names, columns=label_names)
    pred_df = pd.DataFrame(pred_cooccurrence, index=label_names, columns=label_names)
    delta_df = pd.DataFrame(cooccurrence_delta, index=label_names, columns=label_names)

    flow_df.to_csv(MULTILABEL_DIR / "false_negative_to_false_positive_flow.csv")
    true_df.to_csv(MULTILABEL_DIR / "true_label_cooccurrence.csv")
    pred_df.to_csv(MULTILABEL_DIR / "predicted_label_cooccurrence.csv")
    delta_df.to_csv(MULTILABEL_DIR / "predicted_minus_true_cooccurrence.csv")

    fig, ax = plt.subplots(figsize=(13.0, 10.5))
    sns.heatmap(flow_df, cmap="Reds", ax=ax)
    ax.set_title("False-negative to false-positive co-error flow")
    ax.set_xlabel("False positive label")
    ax.set_ylabel("False negative label")
    fig.tight_layout()
    save_figure(fig, MULTILABEL_DIR / "fig_label_error_flow_heatmap.png")

    fig, ax = plt.subplots(figsize=(13.0, 10.5))
    vmax = max(1, int(np.abs(delta_df.to_numpy()).max()))
    sns.heatmap(delta_df, cmap="vlag", center=0, vmin=-vmax, vmax=vmax, ax=ax)
    ax.set_title("Predicted minus true label co-occurrence")
    ax.set_xlabel("Label")
    ax.set_ylabel("Label")
    fig.tight_layout()
    save_figure(fig, MULTILABEL_DIR / "fig_cooccurrence_delta_heatmap.png")


def calibration_bins(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    flat_true = y_true.reshape(-1)
    flat_prob = probabilities.reshape(-1)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end == 1.0:
            mask = (flat_prob >= start) & (flat_prob <= end)
        else:
            mask = (flat_prob >= start) & (flat_prob < end)
        if not mask.any():
            rows.append(
                {
                    "bin_start": float(start),
                    "bin_end": float(end),
                    "count": 0,
                    "mean_probability": float("nan"),
                    "observed_positive_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin_start": float(start),
                "bin_end": float(end),
                "count": int(mask.sum()),
                "mean_probability": float(flat_prob[mask].mean()),
                "observed_positive_rate": float(flat_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_threshold_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> None:
    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        rounded = float(round(threshold, 2))
        y_pred = apply_threshold_with_min_labels(probabilities, rounded, MIN_PREDICTED_LABELS)
        emotion = metrics_from_binary(y_true, y_pred, label_names)
        sentiment = sentiment_metrics_from_binary(y_true, y_pred, label_names)
        threshold_rows.append(
            {
                "threshold": rounded,
                "minimum_predicted_labels": MIN_PREDICTED_LABELS,
                "emotion_micro_f1": emotion["micro"]["f1"],
                "emotion_macro_f1": emotion["macro"]["f1"],
                "emotion_samples_f1": emotion["samples"]["f1"],
                "hamming_loss": emotion["hamming_loss"],
                "subset_accuracy": emotion["subset_accuracy"],
                "avg_predicted_labels_per_entry": emotion["avg_predicted_labels_per_entry"],
                "sentiment_accuracy": sentiment["accuracy"],
                "sentiment_macro_f1": sentiment["macro"]["f1"],
                "diagnostic_note": "post_hoc_threshold_sensitivity_not_final_model_selection",
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(THRESHOLD_DIR / "threshold_sensitivity.csv", index=False)

    auprc_rows = []
    for label_idx, label in enumerate(label_names):
        support = int(y_true[:, label_idx].sum())
        if support == 0:
            auprc = float("nan")
        else:
            auprc = float(average_precision_score(y_true[:, label_idx], probabilities[:, label_idx]))
        auprc_rows.append({"label": label, "support": support, "average_precision": auprc})
    auprc_frame = pd.DataFrame(auprc_rows).sort_values("average_precision", ascending=True)
    auprc_frame.to_csv(THRESHOLD_DIR / "per_label_auprc.csv", index=False)

    calibration = calibration_bins(y_true, probabilities)
    calibration.to_csv(THRESHOLD_DIR / "calibration_bins.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for metric, label in [
        ("emotion_micro_f1", "Micro F1"),
        ("emotion_macro_f1", "Macro F1"),
        ("emotion_samples_f1", "Samples F1"),
        ("sentiment_accuracy", "Sentiment accuracy"),
    ]:
        ax.plot(threshold_frame["threshold"], threshold_frame[metric], marker="o", label=label)
    ax.axvline(OPTUNA_THRESHOLD, color="#333333", linestyle="--", linewidth=1.2, label="Optuna threshold")
    ax.set_title("Post hoc threshold sensitivity for BERT Optuna probabilities")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, THRESHOLD_DIR / "fig_threshold_sensitivity_curves.png")

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.barh(auprc_frame["label"], auprc_frame["average_precision"], color="#4C72B0")
    ax.set_title("Per-label area under the precision-recall curve")
    ax.set_xlabel("Average precision")
    ax.set_ylabel("")
    ax.set_xlim(0, max(1.0, float(auprc_frame["average_precision"].max(skipna=True)) + 0.05))
    fig.tight_layout()
    save_figure(fig, THRESHOLD_DIR / "fig_per_label_auprc.png")

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    plot_calibration = calibration.dropna(subset=["mean_probability", "observed_positive_rate"])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    ax.plot(
        plot_calibration["mean_probability"],
        plot_calibration["observed_positive_rate"],
        marker="o",
        color="#55A868",
        label="Observed",
    )
    ax.set_title("Overall multilabel calibration bins")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, THRESHOLD_DIR / "fig_calibration_bins.png")


def build_length_features(texts: list[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        cleaned = str(text)
        words = cleaned.split()
        rows.append(
            {
                "word_count": len(words),
                "char_count": len(cleaned),
                "exclamation_count": cleaned.count("!"),
                "question_count": cleaned.count("?"),
                "avg_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_lexicon_features(texts: list[str], label_names: list[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        lowered = str(text).lower()
        row = {}
        positive_total = 0
        negative_total = 0
        for label in label_names:
            label_key = label.lower()
            terms = EMOTION_LEXICON.get(label_key, [label_key])
            count = 0
            for term in terms:
                count += len(re.findall(r"\b" + re.escape(term.lower()) + r"\b", lowered))
            row[f"lexicon_{safe_name(label_key)}"] = count
            valence = derive_sentiment(np.array([1 if candidate == label else 0 for candidate in label_names]), label_names)[1]
            if valence > 0:
                positive_total += count
            elif valence < 0:
                negative_total += count
        row["lexicon_positive_total"] = positive_total
        row["lexicon_negative_total"] = negative_total
        row["lexicon_any_total"] = positive_total + negative_total
        rows.append(row)
    return pd.DataFrame(rows)


def scaled_sparse_numeric(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    scaler = StandardScaler(with_mean=False)
    train_scaled = scaler.fit_transform(train_frame.astype(float))
    val_scaled = scaler.transform(val_frame.astype(float))
    test_scaled = scaler.transform(test_frame.astype(float))
    return sparse.csr_matrix(train_scaled), sparse.csr_matrix(val_scaled), sparse.csr_matrix(test_scaled)


def build_feature_matrices(
    variant: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    topic_cols: list[str],
    label_names: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    train_texts = train_df[text_col].fillna("").astype(str).tolist()
    val_texts = val_df[text_col].fillna("").astype(str).tolist()
    test_texts = test_df[text_col].fillna("").astype(str).tolist()

    matrices: list[tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]] = []

    if variant in {"text_tfidf", "all_features"}:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
            strip_accents="unicode",
        )
        matrices.append(
            (
                vectorizer.fit_transform(train_texts),
                vectorizer.transform(val_texts),
                vectorizer.transform(test_texts),
            )
        )

    if variant in {"topics", "all_features"}:
        matrices.append(
            (
                sparse.csr_matrix(train_df[topic_cols].astype(float).to_numpy()),
                sparse.csr_matrix(val_df[topic_cols].astype(float).to_numpy()),
                sparse.csr_matrix(test_df[topic_cols].astype(float).to_numpy()),
            )
        )

    if variant in {"length", "all_features"}:
        matrices.append(
            scaled_sparse_numeric(
                build_length_features(train_texts),
                build_length_features(val_texts),
                build_length_features(test_texts),
            )
        )

    if variant in {"lexicon", "all_features"}:
        matrices.append(
            scaled_sparse_numeric(
                build_lexicon_features(train_texts, label_names),
                build_lexicon_features(val_texts, label_names),
                build_lexicon_features(test_texts, label_names),
            )
        )

    if not matrices:
        raise ValueError(f"Unknown feature-ablation variant: {variant}")

    if len(matrices) == 1:
        return matrices[0]

    return (
        sparse.hstack([matrix_set[0] for matrix_set in matrices]).tocsr(),
        sparse.hstack([matrix_set[1] for matrix_set in matrices]).tocsr(),
        sparse.hstack([matrix_set[2] for matrix_set in matrices]).tocsr(),
    )


def fit_multilabel_logistic(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_val: sparse.csr_matrix,
    x_test: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    val_probs = np.zeros((x_val.shape[0], y_train.shape[1]), dtype=float)
    test_probs = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=float)

    for label_idx in range(y_train.shape[1]):
        y_col = y_train[:, label_idx]
        unique_values = np.unique(y_col)
        if unique_values.size < 2:
            constant_probability = float(unique_values[0])
            val_probs[:, label_idx] = constant_probability
            test_probs[:, label_idx] = constant_probability
            continue

        clf = LogisticRegression(
            max_iter=1200,
            solver="liblinear",
            class_weight="balanced",
            random_state=SEED,
        )
        clf.fit(x_train, y_col)
        val_probs[:, label_idx] = clf.predict_proba(x_val)[:, 1]
        test_probs[:, label_idx] = clf.predict_proba(x_test)[:, 1]

    return val_probs, test_probs


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    best_threshold = 0.50
    best_micro_f1 = -1.0
    for threshold in np.arange(0.10, 0.91, 0.05):
        rounded = float(round(threshold, 2))
        y_pred = (probabilities >= rounded).astype(int)
        micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)[2]
        macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
        rows.append({"threshold": rounded, "validation_micro_f1": micro, "validation_macro_f1": macro})
        if micro > best_micro_f1:
            best_micro_f1 = float(micro)
            best_threshold = rounded
    return best_threshold, pd.DataFrame(rows)


def run_feature_ablation(
    df: pd.DataFrame,
    text_col: str,
    emotion_cols: list[str],
    topic_cols: list[str],
    label_names: list[str],
    split_map: dict[str, np.ndarray],
) -> None:
    variants = [
        ("text_tfidf", "Text TF-IDF"),
        ("topics", "Topics only"),
        ("length", "Length only"),
        ("lexicon", "Lexicon only"),
        ("all_features", "All features"),
    ]
    labels = df[emotion_cols].to_numpy(dtype=int)
    train_idx, val_idx, test_idx = split_map["train"], split_map["val"], split_map["test"]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    summary_rows = []
    label_rows = []
    threshold_rows = []

    for variant_key, variant_label in variants:
        print(f"Running lightweight feature ablation: {variant_label}")
        x_train, x_val, x_test = build_feature_matrices(
            variant_key,
            train_df,
            val_df,
            test_df,
            text_col,
            topic_cols,
            label_names,
        )
        val_prob, test_prob = fit_multilabel_logistic(x_train, y_train, x_val, x_test)
        threshold, scan = select_threshold(y_val, val_prob)
        scan.insert(0, "variant", variant_label)
        threshold_rows.extend(scan.to_dict("records"))

        y_test_pred = (test_prob >= threshold).astype(int)
        emotion = metrics_from_binary(y_test, y_test_pred, label_names)
        sentiment = sentiment_metrics_from_binary(y_test, y_test_pred, label_names)
        summary_rows.append(
            {
                "variant": variant_label,
                "selected_threshold": threshold,
                "emotion_micro_f1": emotion["micro"]["f1"],
                "emotion_macro_f1": emotion["macro"]["f1"],
                "emotion_samples_f1": emotion["samples"]["f1"],
                "subset_accuracy": emotion["subset_accuracy"],
                "hamming_loss": emotion["hamming_loss"],
                "avg_predicted_labels_per_entry": emotion["avg_predicted_labels_per_entry"],
                "sentiment_accuracy": sentiment["accuracy"],
                "sentiment_macro_f1": sentiment["macro"]["f1"],
            }
        )
        for row in emotion["label_metrics"]:
            label_rows.append({"variant": variant_label, **row})

    summary = pd.DataFrame(summary_rows)
    per_label = pd.DataFrame(label_rows)
    threshold_scan = pd.DataFrame(threshold_rows)
    summary.to_csv(ABLATION_DIR / "feature_ablation_metrics.csv", index=False)
    per_label.to_csv(ABLATION_DIR / "feature_ablation_per_emotion.csv", index=False)
    threshold_scan.to_csv(ABLATION_DIR / "feature_ablation_validation_threshold_scan.csv", index=False)

    metric_order = [
        "emotion_micro_f1",
        "emotion_macro_f1",
        "emotion_samples_f1",
        "sentiment_accuracy",
        "sentiment_macro_f1",
        "hamming_loss",
    ]
    plot_frame = summary.melt(
        id_vars=["variant"],
        value_vars=metric_order,
        var_name="metric",
        value_name="value",
    )
    plot_frame["metric"] = plot_frame["metric"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    sns.barplot(data=plot_frame, x="metric", y="value", hue="variant", ax=ax)
    ax.set_title("Lightweight feature-ablation model metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=16)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, ABLATION_DIR / "fig_feature_ablation_overall_metrics.png")

    heatmap_frame = per_label.pivot(index="label", columns="variant", values="f1").reindex(label_names)
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    sns.heatmap(heatmap_frame, cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Feature-ablation per-emotion F1")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, ABLATION_DIR / "fig_feature_ablation_per_emotion_heatmap.png")

    text_row = summary.loc[summary["variant"] == "Text TF-IDF"].iloc[0]
    delta = summary.copy()
    for metric in metric_order:
        delta[f"delta_{metric}_vs_text_tfidf"] = delta[metric] - float(text_row[metric])
    delta_plot = delta[delta["variant"] != "Text TF-IDF"].melt(
        id_vars=["variant"],
        value_vars=[f"delta_{metric}_vs_text_tfidf" for metric in metric_order],
        var_name="metric",
        value_name="delta",
    )
    delta_plot["metric"] = (
        delta_plot["metric"]
        .str.replace("delta_", "", regex=False)
        .str.replace("_vs_text_tfidf", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    sns.barplot(data=delta_plot, x="metric", y="delta", hue="variant", ax=ax)
    ax.axhline(0, color="#333333", linewidth=1.1)
    ax.set_title("Feature-ablation delta versus text TF-IDF")
    ax.set_xlabel("")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", rotation=16)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, ABLATION_DIR / "fig_feature_group_delta_vs_text.png")


def row_f1(true_set: set[str], predicted_set: set[str]) -> float:
    tp = len(true_set & predicted_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)
    denominator = (2 * tp) + fp + fn
    return (2 * tp) / denominator if denominator else 1.0


def choose_lime_examples(predictions: pd.DataFrame, label_names: list[str], limit: int) -> pd.DataFrame:
    frame = predictions.copy()
    frame["true_set"] = frame["true_emotions"].apply(parse_label_set)
    frame["predicted_set"] = frame["predicted_emotions"].apply(parse_label_set)
    frame["tp"] = frame.apply(lambda row: len(row["true_set"] & row["predicted_set"]), axis=1)
    frame["fp"] = frame.apply(lambda row: len(row["predicted_set"] - row["true_set"]), axis=1)
    frame["fn"] = frame.apply(lambda row: len(row["true_set"] - row["predicted_set"]), axis=1)
    frame["row_f1"] = frame.apply(lambda row: row_f1(row["true_set"], row["predicted_set"]), axis=1)
    frame["exact_match"] = frame["true_set"] == frame["predicted_set"]

    support = labels_to_matrix(frame["true_emotions"], label_names).sum(axis=0)
    rare_labels = {label for label, count in zip(label_names, support) if int(count) <= 10}
    frame["has_rare_error"] = frame.apply(
        lambda row: bool(((row["true_set"] ^ row["predicted_set"]) & rare_labels)),
        axis=1,
    )

    candidates = [
        ("strong", frame.sort_values(["exact_match", "row_f1", "tp"], ascending=[False, False, False]).head(3)),
        ("overprediction", frame[frame["fp"] > 0].sort_values(["fp", "row_f1"], ascending=[False, True]).head(3)),
        ("underprediction", frame[frame["fn"] > 0].sort_values(["fn", "row_f1"], ascending=[False, True]).head(3)),
        ("rare_error", frame[frame["has_rare_error"]].sort_values(["row_f1", "fp", "fn"], ascending=[True, False, False]).head(3)),
    ]

    selected = []
    selected_ids: set[int] = set()
    for category, candidate_frame in candidates:
        for _, row in candidate_frame.iterrows():
            row_id = int(row["row_id"])
            if row_id in selected_ids:
                continue
            item = row.to_dict()
            item["lime_category"] = category
            selected.append(item)
            selected_ids.add(row_id)
            if len(selected) >= limit:
                return pd.DataFrame(selected)

    fallback = frame.sort_values(["row_f1", "row_id"], ascending=[True, True])
    for _, row in fallback.iterrows():
        row_id = int(row["row_id"])
        if row_id in selected_ids:
            continue
        item = row.to_dict()
        item["lime_category"] = "fallback"
        selected.append(item)
        selected_ids.add(row_id)
        if len(selected) >= limit:
            break

    return pd.DataFrame(selected)


def label_indices_for_lime(row: pd.Series, label_names: list[str], probabilities: np.ndarray) -> list[int]:
    true_set = parse_label_set(row["true_emotions"])
    predicted_set = parse_label_set(row["predicted_emotions"])
    focus_labels = list((predicted_set - true_set) | (true_set - predicted_set) | (true_set & predicted_set))
    if not focus_labels:
        focus_labels = list(predicted_set or true_set)
    focus_indices = [label_names.index(label) for label in focus_labels if label in label_names]
    if len(focus_indices) < 2:
        row_position = int(row["_position"])
        ranked = list(np.argsort(probabilities[row_position])[::-1])
        for idx in ranked:
            if idx not in focus_indices:
                focus_indices.append(int(idx))
            if len(focus_indices) >= 2:
                break
    return focus_indices[:3]


def load_bert_classifier(label_names: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(OPTUNA_MODEL_DIR, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(OPTUNA_MODEL_DIR, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def classifier_fn(texts: list[str]) -> np.ndarray:
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                batch_texts = [str(text) for text in texts[start : start + BATCH_SIZE]]
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                encodings = {key: value.to(device) for key, value in encodings.items()}
                logits = model(**encodings).logits
                all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.vstack(all_probs)

    return classifier_fn


def run_lime_explanations(
    predictions: pd.DataFrame,
    probabilities: np.ndarray,
    label_names: list[str],
    limit: int,
    num_samples: int,
) -> None:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:
        raise SystemExit(
            "The `lime` package is required for section 06. "
            "Install it with `python -m pip install -r requirements.txt`, "
            "or run this script with `--skip-lime`."
        ) from exc

    skipped_path = LIME_DIR / "LIME_SKIPPED.txt"
    if skipped_path.exists():
        skipped_path.unlink()

    selected = choose_lime_examples(predictions, label_names, limit=limit)
    if selected.empty:
        (LIME_DIR / "lime_no_examples.txt").write_text("No LIME examples were selected.\n", encoding="utf-8")
        return

    position_map = {int(row_id): position for position, row_id in enumerate(predictions["row_id"].astype(int))}
    selected["_position"] = selected["row_id"].astype(int).map(position_map)
    selected.to_csv(LIME_DIR / "lime_selected_examples.csv", index=False)

    classifier_fn = load_bert_classifier(label_names)
    explainer = LimeTextExplainer(
        class_names=label_names,
        bow=True,
        random_state=SEED,
    )

    weight_rows = []
    panel_rows = []
    for example_number, (_, row_series) in enumerate(selected.iterrows(), start=1):
        labels_to_explain = label_indices_for_lime(row_series, label_names, probabilities)
        text = str(row_series["text"])
        explanation = explainer.explain_instance(
            text,
            classifier_fn,
            labels=labels_to_explain,
            num_features=12,
            num_samples=num_samples,
        )

        base_name = f"example_{example_number:02d}_row_{int(row_series['row_id'])}"
        html_path = LIME_DIR / f"{base_name}.html"
        html_path.write_text(explanation.as_html(labels=labels_to_explain), encoding="utf-8")

        example_token_summaries = []
        for label_idx in labels_to_explain:
            label = label_names[label_idx]
            for token, weight in explanation.as_list(label=label_idx):
                weight_rows.append(
                    {
                        "example": example_number,
                        "row_id": int(row_series["row_id"]),
                        "lime_category": row_series["lime_category"],
                        "label": label,
                        "token": token,
                        "weight": float(weight),
                        "absolute_weight": abs(float(weight)),
                    }
                )
            top_tokens = explanation.as_list(label=label_idx)[:4]
            example_token_summaries.append(
                f"{label}: "
                + ", ".join([f"{token} ({weight:+.2f})" for token, weight in top_tokens])
            )

        panel_rows.append(
            {
                "example": example_number,
                "row_id": int(row_series["row_id"]),
                "category": row_series["lime_category"],
                "true_emotions": row_series["true_emotions"],
                "predicted_emotions": row_series["predicted_emotions"],
                "lime_summary": "; ".join(example_token_summaries),
                "text": textwrap.shorten(" ".join(text.split()), width=130, placeholder="..."),
            }
        )

    weights = pd.DataFrame(weight_rows)
    panel = pd.DataFrame(panel_rows)
    weights.to_csv(LIME_DIR / "lime_token_weights.csv", index=False)
    panel.to_csv(LIME_DIR / "lime_representative_panel.csv", index=False)

    aggregate = (
        weights.assign(token_clean=weights["token"].str.lower())
        .groupby(["label", "token_clean"], as_index=False)
        .agg(mean_weight=("weight", "mean"), mean_abs_weight=("absolute_weight", "mean"), occurrences=("token", "count"))
        .sort_values("mean_abs_weight", ascending=False)
    )
    aggregate.to_csv(LIME_DIR / "lime_aggregate_token_weights.csv", index=False)

    plot_tokens = (
        aggregate.groupby("label", as_index=False, group_keys=False)
        .head(4)
        .copy()
        .head(32)
    )
    plot_tokens["token_label"] = plot_tokens["token_clean"] + " | " + plot_tokens["label"]
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.35 * len(plot_tokens))))
    sns.barplot(data=plot_tokens, x="mean_abs_weight", y="token_label", hue="label", dodge=False, ax=ax)
    ax.set_title("Top LIME token contributions by label")
    ax.set_xlabel("Mean absolute LIME weight")
    ax.set_ylabel("")
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, LIME_DIR / "fig_lime_top_tokens_by_label.png")

    table_rows = panel[["category", "true_emotions", "predicted_emotions", "lime_summary", "text"]].copy()
    table_rows = table_rows.map(lambda value: textwrap.fill(str(value), width=34))
    fig, ax = plt.subplots(figsize=(18, max(6.5, 0.75 * len(table_rows))))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows.values,
        colLabels=["Category", "True", "Predicted", "LIME summary", "Text excerpt"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 2.2)
    ax.set_title("Representative LIME explanations for BERT Optuna", pad=18)
    fig.tight_layout()
    save_figure(fig, LIME_DIR / "fig_lime_representative_panel.png", dpi=240)


def evaluate_untrained_lemotif(
    df: pd.DataFrame,
    text_col: str,
    emotion_cols: list[str],
    label_names: list[str],
    split_map: dict[str, np.ndarray],
) -> tuple[dict, pd.DataFrame]:
    output_dir = COMPARATIVE_DIR / "lemotif_untrained_bert"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    if metrics_path.exists() and predictions_path.exists():
        return read_json(metrics_path), pd.read_csv(predictions_path).fillna("")

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[emotion_cols].to_numpy(dtype=float)
    test_idx = split_map["test"]

    tokenizer = AutoTokenizer.from_pretrained(OPTUNA_MODEL_DIR, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(OPTUNA_MODEL_DIR, local_files_only=True)
    config.num_labels = len(label_names)
    config.problem_type = "multi_label_classification"
    config.id2label = {idx: label for idx, label in enumerate(label_names)}
    config.label2id = {label: idx for idx, label in enumerate(label_names)}

    dataset = LemotifEmotionDataset(
        texts=[texts[idx] for idx in test_idx],
        labels=labels[test_idx],
        row_ids=test_idx.tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    _, y_true, probabilities, row_ids = run_epoch(
        model=model,
        data_loader=loader,
        loss_fn=loss_fn,
        device=device,
    )
    y_pred = apply_threshold_with_min_labels(probabilities, OPTUNA_THRESHOLD, MIN_PREDICTED_LABELS)
    emotion = metrics_from_binary(y_true.astype(int), y_pred, label_names)
    sentiment = sentiment_metrics_from_binary(y_true.astype(int), y_pred, label_names)
    predictions = prediction_frame_from_binary(
        df,
        row_ids,
        y_true.astype(int),
        probabilities,
        y_pred,
        label_names,
        text_col,
    )
    metrics = {
        "model_name": "bert-base-uncased",
        "evaluation_mode": "untrained_random_init_lemotif_split",
        "threshold": OPTUNA_THRESHOLD,
        "minimum_predicted_labels": MIN_PREDICTED_LABELS,
        "split_sizes": {"test": len(test_idx)},
        "label_names": label_names,
        "test_metrics": emotion,
        "test_sentiment_metrics": sentiment,
    }
    write_json(metrics_path, metrics)
    predictions.to_csv(predictions_path, index=False)
    return metrics, predictions


def metric_summary_row(model_name: str, metrics: dict) -> dict[str, float | str]:
    emotion = metrics["test_metrics"]
    sentiment = metrics["test_sentiment_metrics"]
    return {
        "model": model_name,
        "emotion_micro_f1": emotion["micro"]["f1"],
        "emotion_macro_f1": emotion["macro"]["f1"],
        "emotion_samples_f1": emotion.get("samples", {}).get("f1", float("nan")),
        "subset_accuracy": emotion["subset_accuracy"],
        "hamming_loss": emotion["hamming_loss"],
        "sentiment_accuracy": sentiment["accuracy"],
        "sentiment_macro_f1": sentiment["macro"]["f1"],
    }


def plot_model_metric_bars(summary: pd.DataFrame, path: Path, title: str) -> None:
    metric_cols = [column for column in summary.columns if column != "model"]
    plot_frame = summary.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value")
    plot_frame["metric"] = plot_frame["metric"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    sns.barplot(data=plot_frame, x="metric", y="value", hue="model", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=16)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, path)


def build_model_ci_frame(
    model_predictions: dict[str, pd.DataFrame],
    label_names: list[str],
    iterations: int,
) -> pd.DataFrame:
    rows = []
    for index, (model_name, predictions) in enumerate(model_predictions.items()):
        y_true = labels_to_matrix(predictions["true_emotions"], label_names)
        y_pred = labels_to_matrix(predictions["predicted_emotions"], label_names)
        samples, _ = bootstrap_metric_samples(
            y_true,
            y_pred,
            label_names,
            iterations=iterations,
            seed=SEED + index + 101,
        )
        estimates = compute_overall_scores(y_true, y_pred, label_names)
        ci = bootstrap_ci(samples.to_dict("records"), estimates, model_name=model_name)
        rows.extend(ci.to_dict("records"))
    return pd.DataFrame(rows)


def plot_model_ci(ci_frame: pd.DataFrame, path: Path, title: str) -> None:
    display_metrics = ["emotion_micro_f1", "emotion_macro_f1", "emotion_samples_f1", "sentiment_accuracy", "sentiment_macro_f1"]
    plot_frame = ci_frame[ci_frame["metric"].isin(display_metrics)].copy()
    plot_frame["metric_label"] = plot_frame["metric"].str.replace("_", " ").str.title()
    plot_frame["model_metric"] = plot_frame["model"] + " | " + plot_frame["metric_label"]
    plot_frame = plot_frame.sort_values(["metric", "estimate"])
    y_pos = np.arange(len(plot_frame))

    fig, ax = plt.subplots(figsize=(12.0, max(6.2, 0.4 * len(plot_frame))))
    lower = plot_frame["estimate"] - plot_frame["ci_lower"]
    upper = plot_frame["ci_upper"] - plot_frame["estimate"]
    ax.errorbar(
        plot_frame["estimate"],
        y_pos,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#2F6FB0",
        ecolor="#555555",
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_frame["model_metric"])
    ax.set_xlabel("Bootstrap score")
    ax.set_xlim(0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, path)


def build_paired_delta_frame(
    baseline_name: str,
    baseline_predictions: pd.DataFrame,
    comparison_predictions: dict[str, pd.DataFrame],
    label_names: list[str],
    iterations: int,
) -> pd.DataFrame:
    rows = []
    base_true = labels_to_matrix(baseline_predictions["true_emotions"], label_names)
    base_pred = labels_to_matrix(baseline_predictions["predicted_emotions"], label_names)
    n_base = len(baseline_predictions)
    rng = np.random.default_rng(SEED + 303)
    metrics = ["emotion_micro_f1", "emotion_macro_f1", "sentiment_accuracy", "sentiment_macro_f1"]

    for model_name, predictions in comparison_predictions.items():
        other_true = labels_to_matrix(predictions["true_emotions"], label_names)
        other_pred = labels_to_matrix(predictions["predicted_emotions"], label_names)
        n_rows = min(n_base, len(predictions))
        delta_samples = {metric: [] for metric in metrics}
        for _ in range(iterations):
            idx = rng.integers(0, n_rows, size=n_rows)
            base_scores = compute_overall_scores(base_true[:n_rows][idx], base_pred[:n_rows][idx], label_names)
            other_scores = compute_overall_scores(other_true[:n_rows][idx], other_pred[:n_rows][idx], label_names)
            for metric in metrics:
                delta_samples[metric].append(base_scores[metric] - other_scores[metric])

        for metric, values in delta_samples.items():
            arr = np.array(values, dtype=float)
            lower, upper = np.quantile(arr, [0.025, 0.975])
            rows.append(
                {
                    "baseline_model": baseline_name,
                    "comparison_model": model_name,
                    "metric": metric,
                    "delta_baseline_minus_comparison": float(arr.mean()),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "paired_rows_used": int(n_rows),
                }
            )
    return pd.DataFrame(rows)


def plot_paired_delta(delta_frame: pd.DataFrame, path: Path, title: str) -> None:
    plot_frame = delta_frame.copy()
    plot_frame["label"] = (
        plot_frame["comparison_model"]
        + " | "
        + plot_frame["metric"].str.replace("_", " ").str.title()
    )
    plot_frame = plot_frame.sort_values("delta_baseline_minus_comparison")
    y_pos = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(10.5, max(5.6, 0.42 * len(plot_frame))))
    lower = plot_frame["delta_baseline_minus_comparison"] - plot_frame["ci_lower"]
    upper = plot_frame["ci_upper"] - plot_frame["delta_baseline_minus_comparison"]
    ax.errorbar(
        plot_frame["delta_baseline_minus_comparison"],
        y_pos,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#55A868",
        ecolor="#555555",
        capsize=4,
    )
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_frame["label"])
    ax.set_xlabel("Delta: BERT Optuna minus comparison")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, path)


def build_per_emotion_model_frame(model_metrics: dict[str, dict], label_names: list[str]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in model_metrics.items():
        for row in metrics["test_metrics"]["label_metrics"]:
            rows.append({"model": model_name, **row})
    frame = pd.DataFrame(rows)
    frame["label"] = pd.Categorical(frame["label"], categories=label_names, ordered=True)
    return frame.sort_values(["label", "model"]).reset_index(drop=True)


def plot_per_emotion_model_comparison(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    sns.barplot(data=frame, y="label", x="f1", hue="model", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, path)


def run_lemotif_comparisons(
    fixed_metrics: dict,
    fixed_predictions: pd.DataFrame,
    untrained_metrics: dict,
    untrained_predictions: pd.DataFrame,
    label_names: list[str],
    iterations: int,
) -> None:
    gemma_metrics_path = first_existing(
        [
            PROJECT_ROOT / "output" / "gemma_2b_emotion_model" / "finetuned" / "metrics.json",
            WORKSPACE_ROOT / "output" / "gemma_2b_emotion_model" / "finetuned" / "metrics.json",
        ]
    )
    gemma_predictions_path = first_existing(
        [
            PROJECT_ROOT / "output" / "gemma_2b_emotion_model" / "finetuned" / "test_predictions.csv",
            WORKSPACE_ROOT / "output" / "gemma_2b_emotion_model" / "finetuned" / "test_predictions.csv",
        ]
    )
    if gemma_metrics_path is None or gemma_predictions_path is None:
        (COMPARATIVE_DIR / "lemotif_gemma_missing.txt").write_text(
            "Fine-tuned Gemma Lemotif metrics/predictions were not found, so Lemotif comparison uses BERT models only.\n",
            encoding="utf-8",
        )
        model_metrics = {
            "Untrained BERT": untrained_metrics,
            MODEL_DISPLAY_NAME: fixed_metrics,
        }
        model_predictions = {
            "Untrained BERT": untrained_predictions,
            MODEL_DISPLAY_NAME: fixed_predictions,
        }
    else:
        gemma_metrics = read_json(gemma_metrics_path)
        gemma_predictions = pd.read_csv(gemma_predictions_path).fillna("")
        model_metrics = {
            "Untrained BERT": untrained_metrics,
            MODEL_DISPLAY_NAME: fixed_metrics,
            "Fine-tuned Gemma": gemma_metrics,
        }
        model_predictions = {
            "Untrained BERT": untrained_predictions,
            MODEL_DISPLAY_NAME: fixed_predictions,
            "Fine-tuned Gemma": gemma_predictions,
        }

    summary = pd.DataFrame(
        [metric_summary_row(model_name, metrics) for model_name, metrics in model_metrics.items()]
    )
    summary.to_csv(COMPARATIVE_DIR / "lemotif_model_metric_summary.csv", index=False)
    plot_model_metric_bars(
        summary,
        COMPARATIVE_DIR / "fig_lemotif_model_metric_summary.png",
        "Lemotif held-out test model comparison",
    )

    ci_frame = build_model_ci_frame(model_predictions, label_names, iterations)
    ci_frame.to_csv(COMPARATIVE_DIR / "lemotif_model_metric_bootstrap_ci.csv", index=False)
    plot_model_ci(
        ci_frame,
        COMPARATIVE_DIR / "fig_lemotif_model_metric_ci.png",
        "Lemotif model bootstrap confidence intervals",
    )

    comparisons = {name: preds for name, preds in model_predictions.items() if name != MODEL_DISPLAY_NAME}
    delta_frame = build_paired_delta_frame(
        MODEL_DISPLAY_NAME,
        fixed_predictions,
        comparisons,
        label_names,
        iterations,
    )
    delta_frame.to_csv(COMPARATIVE_DIR / "lemotif_paired_bootstrap_deltas.csv", index=False)
    plot_paired_delta(
        delta_frame,
        COMPARATIVE_DIR / "fig_lemotif_paired_bootstrap_deltas.png",
        "Paired bootstrap deltas on Lemotif",
    )

    per_emotion = build_per_emotion_model_frame(model_metrics, label_names)
    per_emotion.to_csv(COMPARATIVE_DIR / "lemotif_per_emotion_f1_comparison.csv", index=False)
    plot_per_emotion_model_comparison(
        per_emotion,
        COMPARATIVE_DIR / "fig_lemotif_per_emotion_f1_comparison.png",
        "Lemotif per-emotion F1 comparison",
    )


def run_stories_comparisons() -> None:
    trained_metrics_path = first_existing(
        [
            PROJECT_ROOT / "test data" / "output" / "stories_model_variants" / "optuna_checkpoint" / "metrics.json",
            WORKSPACE_ROOT / "test data" / "output" / "stories_model_variants" / "optuna_checkpoint" / "metrics.json",
        ]
    )
    untrained_metrics_path = first_existing(
        [
            PROJECT_ROOT / "test data" / "output" / "untrained_bert" / "metrics.json",
            WORKSPACE_ROOT / "test data" / "output" / "untrained_bert" / "metrics.json",
        ]
    )
    gemma_sentiment_metrics_path = first_existing(
        [
            PROJECT_ROOT / "test data" / "output" / "gemma_sentiment" / "google_gemma-3n-E2B-it" / "metrics.json",
            WORKSPACE_ROOT / "test data" / "output" / "gemma_sentiment" / "google_gemma-3n-E2B-it" / "metrics.json",
        ]
    )
    if trained_metrics_path is None or untrained_metrics_path is None:
        (COMPARATIVE_DIR / "stories_missing.txt").write_text(
            "Stories Optuna/untrained BERT metrics were not found, so Stories comparisons were skipped.\n",
            encoding="utf-8",
        )
        return

    trained = read_json(trained_metrics_path)
    untrained = read_json(untrained_metrics_path)

    emotion_rows = []
    for model_name, metrics in [(MODEL_DISPLAY_NAME, trained), ("Untrained BERT", untrained)]:
        emotion = metrics["test_metrics"]
        emotion_rows.extend(
            [
                {"model": model_name, "metric": "Micro F1", "value": emotion["micro"]["f1"]},
                {"model": model_name, "metric": "Macro F1", "value": emotion["macro"]["f1"]},
                {"model": model_name, "metric": "Samples F1", "value": emotion.get("samples", {}).get("f1", float("nan"))},
                {"model": model_name, "metric": "Subset accuracy", "value": emotion["subset_accuracy"]},
                {"model": model_name, "metric": "Hamming loss", "value": emotion["hamming_loss"]},
            ]
        )
    emotion_frame = pd.DataFrame(emotion_rows)
    emotion_frame.to_csv(COMPARATIVE_DIR / "stories_bert_emotion_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    sns.barplot(data=emotion_frame, x="metric", y="value", hue="model", palette=["#2F6FB0", "#DD8452"], ax=ax)
    ax.set_title("Stories emotion comparison: BERT Optuna vs untrained BERT")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=16)
    ax.legend(title="")
    fig.tight_layout()
    save_figure(fig, COMPARATIVE_DIR / "fig_stories_bert_emotion_comparison.png")

    sentiment_rows = []
    for model_name, metrics in [(MODEL_DISPLAY_NAME, trained), ("Untrained BERT", untrained)]:
        sentiment = metrics["test_sentiment_metrics"]
        sentiment_rows.extend(
            [
                {"model": model_name, "metric": "Accuracy", "value": sentiment["accuracy"]},
                {"model": model_name, "metric": "Macro precision", "value": sentiment["macro"]["precision"]},
                {"model": model_name, "metric": "Macro recall", "value": sentiment["macro"]["recall"]},
                {"model": model_name, "metric": "Macro F1", "value": sentiment["macro"]["f1"]},
            ]
        )

    if gemma_sentiment_metrics_path is not None:
        gemma = read_json(gemma_sentiment_metrics_path)
        sentiment = gemma["test_sentiment_metrics"]
        sentiment_rows.extend(
            [
                {"model": "Gemma sentiment", "metric": "Accuracy", "value": sentiment["accuracy"]},
                {"model": "Gemma sentiment", "metric": "Macro precision", "value": sentiment["macro"]["precision"]},
                {"model": "Gemma sentiment", "metric": "Macro recall", "value": sentiment["macro"]["recall"]},
                {"model": "Gemma sentiment", "metric": "Macro F1", "value": sentiment["macro"]["f1"]},
            ]
        )
    else:
        (COMPARATIVE_DIR / "stories_gemma_sentiment_missing.txt").write_text(
            "Stories Gemma sentiment metrics were not found.\n",
            encoding="utf-8",
        )

    sentiment_frame = pd.DataFrame(sentiment_rows)
    sentiment_frame.to_csv(COMPARATIVE_DIR / "stories_three_model_sentiment_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    sns.barplot(data=sentiment_frame, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Stories sentiment comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=16)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, COMPARATIVE_DIR / "fig_stories_three_model_sentiment_comparison.png")


def write_summary(args: argparse.Namespace) -> None:
    lines = [
        "# BERT Optuna Diagnostics",
        "",
        "Generated sections:",
        "- 01_bootstrap_ci: Optuna held-out bootstrap intervals.",
        "- 02_rare_label_errors: rare-label FP/FN and example rows.",
        "- 03_multilabel_errors: co-error flow and co-occurrence deltas.",
        "- 04_threshold_calibration: post hoc threshold, AUPRC, and calibration analysis.",
        "- 05_feature_ablation: lightweight One-vs-Rest logistic-regression ablations.",
        "- 06_lime: BERT Optuna LIME explanations." if not args.skip_lime else "- 06_lime: skipped by --skip-lime.",
        "- comparative_statistics: Lemotif and Stories model comparison figures.",
        "",
        f"Bootstrap iterations: {args.bootstrap_iterations}",
        f"LIME examples: {args.lime_examples}",
        f"LIME samples per explanation: {args.lime_num_samples}",
        "",
        "Caveats:",
        "- Threshold sweeps are diagnostic/post hoc and are not final model-selection claims.",
        "- Stories Gemma comparison uses existing sentiment artifacts only.",
        "- LIME explanations are local interpretability artifacts, not performance metrics.",
    ]
    (DIAGNOSTICS_DIR / "diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diagnostics and comparisons for BERT Optuna."
    )
    parser.add_argument(
        "--skip-lime",
        action="store_true",
        help="Skip the LIME explanation section.",
    )
    parser.add_argument(
        "--only-lime",
        action="store_true",
        help="Only regenerate the LIME explanation section from existing Optuna predictions.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--lime-examples",
        type=int,
        default=12,
        help="Number of representative rows to explain with LIME.",
    )
    parser.add_argument(
        "--lime-num-samples",
        type=int,
        default=400,
        help="Perturbed text samples per LIME explanation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    if not OPTUNA_METRICS_PATH.exists() or not OPTUNA_PREDICTIONS_PATH.exists():
        raise SystemExit(
            "BERT Optuna artifacts are missing. "
            "Run src/model_eda/run_optuna_comparison.py first."
        )
    if not OPTUNA_MODEL_DIR.exists():
        raise SystemExit(
            "The trained BERT Optuna checkpoint is missing at output/bert_emotion_model_optuna/model. "
            "LIME and untrained comparison need the saved tokenizer/config."
        )

    fixed_metrics = read_json(OPTUNA_METRICS_PATH)
    fixed_predictions = pd.read_csv(OPTUNA_PREDICTIONS_PATH).fillna("")
    label_names = fixed_metrics["label_names"]
    probabilities = probability_matrix_from_predictions(fixed_predictions, label_names)

    if args.only_lime:
        print("Section 6: LIME explanations")
        fixed_predictions_with_position = fixed_predictions.copy()
        fixed_predictions_with_position["_position"] = np.arange(len(fixed_predictions_with_position))
        run_lime_explanations(
            fixed_predictions_with_position,
            probabilities,
            label_names,
            limit=args.lime_examples,
            num_samples=args.lime_num_samples,
        )
        write_summary(args)
        print(f"Saved BERT Optuna LIME diagnostics to: {LIME_DIR}")
        return

    print("Section 1: bootstrap confidence intervals")
    y_true, y_pred, probabilities = run_bootstrap_ci(
        fixed_predictions,
        label_names,
        iterations=args.bootstrap_iterations,
    )

    print("Section 2: rare-label error analysis")
    run_rare_label_analysis(fixed_predictions, y_true, y_pred, probabilities, label_names)

    print("Section 3: multilabel error analysis")
    run_multilabel_error_analysis(y_true, y_pred, label_names)

    print("Section 4: threshold, calibration, and AUPRC")
    run_threshold_calibration(y_true, probabilities, label_names)

    print("Section 5: lightweight feature ablation")
    df, text_col, emotion_cols, topic_cols = load_analysis_data(prefer_cleaned=True)
    df = df.reset_index(drop=True)
    split_map = split_indices(df[emotion_cols].to_numpy(dtype=int))
    run_feature_ablation(df, text_col, emotion_cols, topic_cols, label_names, split_map)

    print("Comparative statistics: Lemotif and Stories")
    untrained_metrics, untrained_predictions = evaluate_untrained_lemotif(
        df,
        text_col,
        emotion_cols,
        label_names,
        split_map,
    )
    run_lemotif_comparisons(
        fixed_metrics,
        fixed_predictions,
        untrained_metrics,
        untrained_predictions,
        label_names,
        iterations=args.bootstrap_iterations,
    )
    run_stories_comparisons()

    if args.skip_lime:
        (LIME_DIR / "LIME_SKIPPED.txt").write_text(
            "LIME was skipped because --skip-lime was passed.\n",
            encoding="utf-8",
        )
    else:
        print("Section 6: LIME explanations")
        fixed_predictions_with_position = fixed_predictions.copy()
        fixed_predictions_with_position["_position"] = np.arange(len(fixed_predictions_with_position))
        run_lime_explanations(
            fixed_predictions_with_position,
            probabilities,
            label_names,
            limit=args.lime_examples,
            num_samples=args.lime_num_samples,
        )

    write_summary(args)
    print(f"Saved BERT Optuna diagnostics to: {DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    main()
