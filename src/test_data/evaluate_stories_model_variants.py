from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name, safe_name
from test_data.common import STORIES_CSV_PATH, TEST_DATA_OUTPUT_DIR, load_stories_data
from train_bert import (
    BATCH_SIZE,
    LOCAL_FILES_ONLY,
    MAX_LENGTH,
    SEED,
    SENTIMENT_VALENCE,
    LemotifEmotionDataset,
    derive_sentiment,
    format_active_labels,
    run_epoch,
    set_seed,
)

OUTPUT_DIR = TEST_DATA_OUTPUT_DIR / "stories_model_variants"
FIGURES_DIR = OUTPUT_DIR / "figures"
SUMMARY_CSV_PATH = OUTPUT_DIR / "stories_model_variant_summary.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "stories_model_variant_report.md"

BASELINE_DIR = PROJECT_ROOT / "output" / "bert_emotion_model"
FIXED_DIR = PROJECT_ROOT / "output" / "BERT-Hyperparamter-Fixed"
OPTUNA_DIR = PROJECT_ROOT / "output" / "bert_emotion_model_optuna"
EXTERNAL_EVAL_DIR = TEST_DATA_OUTPUT_DIR / "external_evaluation"
UNTRAINED_DIR = TEST_DATA_OUTPUT_DIR / "untrained_bert"
GEMMA_SENTIMENT_DIR = TEST_DATA_OUTPUT_DIR / "gemma_sentiment" / "google_gemma-3n-E2B-it"

VARIANTS = [
    {
        "key": "baseline_checkpoint",
        "display": "BERT baseline",
        "source": BASELINE_DIR / "model",
        "metrics_source": BASELINE_DIR / "metrics.json",
        "threshold": None,
        "min_labels": 0,
        "note": "Saved baseline Lemotif split checkpoint; 18-label emotion probabilities; sentiment derived afterward.",
    },
    {
        "key": "fixed_inference",
        "display": "BERT fixed inference",
        "source": BASELINE_DIR / "model",
        "metrics_source": FIXED_DIR / "metrics.json",
        "threshold": 0.59,
        "min_labels": 1,
        "note": "Same saved baseline BERT weights with threshold .59 and minimum one-label fallback.",
    },
    {
        "key": "optuna_checkpoint",
        "display": "BERT Optuna",
        "source": OPTUNA_DIR / "model",
        "metrics_source": OPTUNA_DIR / "metrics.json",
        "threshold": None,
        "min_labels": 0,
        "note": "Saved Optuna-tuned Lemotif split checkpoint; 18-label emotion probabilities; sentiment derived afterward.",
    },
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_threshold(metrics_source: Path, override: float | None) -> tuple[float, str]:
    if override is not None:
        return float(override), "explicit fixed-inference threshold"
    payload = load_json(metrics_source)
    if "best_threshold" in payload:
        return float(payload["best_threshold"]), f"best_threshold from {metrics_source}"
    if "threshold" in payload:
        return float(payload["threshold"]), f"threshold from {metrics_source}"
    return 0.55, "fallback default threshold"


def apply_threshold_with_min_labels(
    probabilities: np.ndarray,
    threshold: float,
    min_labels: int,
) -> np.ndarray:
    y_pred = (probabilities >= threshold).astype(int)
    if min_labels <= 0:
        return y_pred

    for row_index in range(y_pred.shape[0]):
        active_count = int(y_pred[row_index].sum())
        if active_count >= min_labels:
            continue
        for label_index in np.argsort(probabilities[row_index])[::-1]:
            if y_pred[row_index, label_index] == 0:
                y_pred[row_index, label_index] = 1
                active_count += 1
            if active_count >= min_labels:
                break
    return y_pred


def compute_emotion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, object]:
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    sample_precision, sample_recall, sample_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="samples",
        zero_division=0,
    )
    label_precision, label_recall, label_f1, label_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    surplus = y_pred.sum(axis=1) - y_true.sum(axis=1)
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())

    return {
        "subset_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(np.not_equal(y_true, y_pred).mean()),
        "avg_true_labels_per_entry": float(y_true.sum(axis=1).mean()),
        "avg_predicted_labels_per_entry": float(y_pred.sum(axis=1).mean()),
        "zero_predicted_rows": int((y_pred.sum(axis=1) == 0).sum()),
        "overpredicted_any_rows": int((surplus > 0).sum()),
        "underpredicted_any_rows": int((surplus < 0).sum()),
        "same_cardinality_rows": int((surplus == 0).sum()),
        "overpredicted_by_at_least_2_rows": int((surplus >= 2).sum()),
        "underpredicted_by_at_least_2_rows": int((surplus <= -2).sum()),
        "false_positive_labels": false_positives,
        "false_negative_labels": false_negatives,
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
            "precision": float(sample_precision),
            "recall": float(sample_recall),
            "f1": float(sample_f1),
        },
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


def compute_sentiment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, object]:
    true_sentiments = []
    pred_sentiments = []
    true_scores = []
    pred_scores = []

    for true_row, pred_row in zip(y_true, y_pred):
        true_sentiment, true_score = derive_sentiment(true_row, label_names)
        pred_sentiment, pred_score = derive_sentiment(pred_row, label_names)
        true_sentiments.append(true_sentiment)
        pred_sentiments.append(pred_sentiment)
        true_scores.append(true_score)
        pred_scores.append(pred_score)

    ordered_labels = ["negative", "neutral", "positive"]
    precision, recall, f1, support = precision_recall_fscore_support(
        true_sentiments,
        pred_sentiments,
        labels=ordered_labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_sentiments,
        pred_sentiments,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(true_sentiments, pred_sentiments)),
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
                ordered_labels,
                precision,
                recall,
                f1,
                support,
            )
        ],
        "true_sentiments": true_sentiments,
        "predicted_sentiments": pred_sentiments,
        "true_sentiment_scores": true_scores,
        "predicted_sentiment_scores": pred_scores,
    }


def build_predictions_frame(
    df: pd.DataFrame,
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
        pred_sentiment, pred_score = derive_sentiment(y_pred[position], label_names)
        row = {
            "row_id": int(row_id),
            "text": df.iloc[row_id][text_col],
            "true_emotions": format_active_labels(y_true[position], label_names),
            "predicted_emotions": format_active_labels(y_pred[position], label_names),
            "true_label_count": int(y_true[position].sum()),
            "predicted_label_count": int(y_pred[position].sum()),
            "prediction_surplus": int(y_pred[position].sum() - y_true[position].sum()),
            "true_sentiment": true_sentiment,
            "predicted_sentiment": pred_sentiment,
            "true_sentiment_score": true_score,
            "predicted_sentiment_score": pred_score,
        }
        for label_idx, label_name in enumerate(label_names):
            row[f"prob_{safe_name(label_name.lower())}"] = float(y_prob[position, label_idx])
            row[f"pred_{safe_name(label_name.lower())}"] = int(y_pred[position, label_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_variant(
    variant: dict[str, object],
    stories_df: pd.DataFrame,
    text_col: str,
    label_names: list[str],
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    source = Path(variant["source"])  # type: ignore[arg-type]
    threshold, threshold_source = resolve_threshold(
        Path(variant["metrics_source"]),  # type: ignore[arg-type]
        variant["threshold"],  # type: ignore[arg-type]
    )
    min_labels = int(variant["min_labels"])
    variant_dir = OUTPUT_DIR / str(variant["key"])
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {variant['display']} on Stories.")
    print(f"  Source: {source}")
    print(f"  Threshold: {threshold:.2f}; min labels: {min_labels}")

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        local_files_only=LOCAL_FILES_ONLY,
        use_fast=True,
    )
    dataset = LemotifEmotionDataset(
        texts=stories_df[text_col].fillna("").astype(str).tolist(),
        labels=labels,
        row_ids=list(range(len(stories_df))),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    label_id_map = {idx: label for idx, label in enumerate(label_names)}
    reverse_label_id_map = {label: idx for idx, label in label_id_map.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label=label_id_map,
        label2id=reverse_label_id_map,
        ignore_mismatched_sizes=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    test_loss, y_true, y_prob, row_ids = run_epoch(
        model=model,
        data_loader=data_loader,
        loss_fn=loss_fn,
        device=device,
    )
    y_pred = apply_threshold_with_min_labels(y_prob, threshold, min_labels)

    emotion_metrics = compute_emotion_metrics(y_true, y_pred, label_names)
    sentiment_metrics = compute_sentiment_metrics(y_true, y_pred, label_names)
    sentiment_export = {
        key: value
        for key, value in sentiment_metrics.items()
        if key
        not in {
            "true_sentiments",
            "predicted_sentiments",
            "true_sentiment_scores",
            "predicted_sentiment_scores",
        }
    }

    predictions = build_predictions_frame(
        df=stories_df,
        row_ids=row_ids,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        label_names=label_names,
        text_col=text_col,
    )
    predictions.to_csv(variant_dir / "stories_predictions.csv", index=False)
    pd.DataFrame(emotion_metrics["label_metrics"]).to_csv(
        variant_dir / "per_emotion_metrics.csv",
        index=False,
    )
    pd.DataFrame(sentiment_export["label_metrics"]).to_csv(
        variant_dir / "per_sentiment_metrics.csv",
        index=False,
    )

    payload = {
        "stories_test_dataset_path": str(STORIES_CSV_PATH),
        "variant_key": variant["key"],
        "variant_display": variant["display"],
        "source_model_dir": str(source),
        "source_metrics_path": str(variant["metrics_source"]),
        "note": variant["note"],
        "local_files_only": LOCAL_FILES_ONLY,
        "seed": SEED,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "minimum_predicted_labels": min_labels,
        "test_loss": float(test_loss),
        "split_sizes": {"test": len(stories_df)},
        "label_names": label_names,
        "test_metrics": emotion_metrics,
        "test_sentiment_metrics": sentiment_export,
    }
    write_json(variant_dir / "metrics.json", payload)

    return {
        "variant_key": variant["key"],
        "model": variant["display"],
        "source_model_dir": str(source),
        "threshold": threshold,
        "minimum_predicted_labels": min_labels,
        "emotion_micro_f1": emotion_metrics["micro"]["f1"],
        "emotion_macro_f1": emotion_metrics["macro"]["f1"],
        "emotion_subset_accuracy": emotion_metrics["subset_accuracy"],
        "emotion_hamming_loss": emotion_metrics["hamming_loss"],
        "emotion_micro_precision": emotion_metrics["micro"]["precision"],
        "emotion_micro_recall": emotion_metrics["micro"]["recall"],
        "sentiment_accuracy": sentiment_export["accuracy"],
        "sentiment_macro_f1": sentiment_export["macro"]["f1"],
        "sentiment_macro_precision": sentiment_export["macro"]["precision"],
        "sentiment_macro_recall": sentiment_export["macro"]["recall"],
        "avg_true_labels": emotion_metrics["avg_true_labels_per_entry"],
        "avg_predicted_labels": emotion_metrics["avg_predicted_labels_per_entry"],
        "zero_predicted_rows": emotion_metrics["zero_predicted_rows"],
        "overpredicted_rows": emotion_metrics["overpredicted_any_rows"],
        "underpredicted_rows": emotion_metrics["underpredicted_any_rows"],
        "same_cardinality_rows": emotion_metrics["same_cardinality_rows"],
        "false_positive_labels": emotion_metrics["false_positive_labels"],
        "false_negative_labels": emotion_metrics["false_negative_labels"],
    }


def add_existing_comparator_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, display, path in [
        ("full_lemotif_external", "BERT full-Lemotif external", EXTERNAL_EVAL_DIR / "metrics.json"),
        ("untrained_bert", "Untrained BERT", UNTRAINED_DIR / "metrics.json"),
    ]:
        if not path.exists():
            continue
        payload = load_json(path)
        rows.append(
            {
                "variant_key": key,
                "model": display,
                "source_model_dir": payload.get("source_model_dir", payload.get("initialization_source", "")),
                "threshold": payload.get("threshold", payload.get("best_threshold", np.nan)),
                "minimum_predicted_labels": payload.get("minimum_predicted_labels", 0),
                "emotion_micro_f1": payload["test_metrics"]["micro"]["f1"],
                "emotion_macro_f1": payload["test_metrics"]["macro"]["f1"],
                "emotion_subset_accuracy": payload["test_metrics"]["subset_accuracy"],
                "emotion_hamming_loss": payload["test_metrics"]["hamming_loss"],
                "emotion_micro_precision": payload["test_metrics"]["micro"]["precision"],
                "emotion_micro_recall": payload["test_metrics"]["micro"]["recall"],
                "sentiment_accuracy": payload["test_sentiment_metrics"]["accuracy"],
                "sentiment_macro_f1": payload["test_sentiment_metrics"]["macro"]["f1"],
                "sentiment_macro_precision": payload["test_sentiment_metrics"]["macro"]["precision"],
                "sentiment_macro_recall": payload["test_sentiment_metrics"]["macro"]["recall"],
                "avg_true_labels": payload["test_metrics"].get("avg_true_labels_per_entry", np.nan),
                "avg_predicted_labels": payload["test_metrics"].get("avg_predicted_labels_per_entry", np.nan),
                "zero_predicted_rows": payload["test_metrics"].get("zero_predicted_rows", np.nan),
                "overpredicted_rows": payload["test_metrics"].get("overpredicted_any_rows", np.nan),
                "underpredicted_rows": payload["test_metrics"].get("underpredicted_any_rows", np.nan),
                "same_cardinality_rows": np.nan,
                "false_positive_labels": np.nan,
                "false_negative_labels": np.nan,
            }
        )

    if rows:
        summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)

    gemma_path = GEMMA_SENTIMENT_DIR / "metrics.json"
    if gemma_path.exists():
        payload = load_json(gemma_path)
        gemma_metrics = payload.get("test_sentiment_metrics", payload.get("metrics", {}))
        gemma_row = {
            "variant_key": "gemma_sentiment",
            "model": "Gemma sentiment",
            "source_model_dir": payload.get("model_id", ""),
            "threshold": np.nan,
            "minimum_predicted_labels": np.nan,
            "emotion_micro_f1": np.nan,
            "emotion_macro_f1": np.nan,
            "emotion_subset_accuracy": np.nan,
            "emotion_hamming_loss": np.nan,
            "emotion_micro_precision": np.nan,
            "emotion_micro_recall": np.nan,
            "sentiment_accuracy": gemma_metrics["accuracy"],
            "sentiment_macro_f1": gemma_metrics["macro"]["f1"],
            "sentiment_macro_precision": gemma_metrics["macro"]["precision"],
            "sentiment_macro_recall": gemma_metrics["macro"]["recall"],
            "avg_true_labels": np.nan,
            "avg_predicted_labels": np.nan,
            "zero_predicted_rows": np.nan,
            "overpredicted_rows": np.nan,
            "underpredicted_rows": np.nan,
            "same_cardinality_rows": np.nan,
            "false_positive_labels": np.nan,
            "false_negative_labels": np.nan,
        }
        summary = pd.concat([summary, pd.DataFrame([gemma_row])], ignore_index=True)
    return summary


def metric_from_payload(path: Path, metric: str) -> float:
    payload = load_json(path)
    if metric == "emotion_micro_f1":
        return float(payload["test_metrics"]["micro"]["f1"])
    if metric == "emotion_macro_f1":
        return float(payload["test_metrics"]["macro"]["f1"])
    if metric == "sentiment_accuracy":
        return float(payload["test_sentiment_metrics"]["accuracy"])
    if metric == "sentiment_macro_f1":
        return float(payload["test_sentiment_metrics"]["macro"]["f1"])
    raise KeyError(metric)


def build_in_domain_vs_stories_frame(summary: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("baseline_checkpoint", "BERT baseline", BASELINE_DIR / "metrics.json"),
        ("fixed_inference", "BERT fixed inference", FIXED_DIR / "metrics.json"),
        ("optuna_checkpoint", "BERT Optuna", OPTUNA_DIR / "metrics.json"),
    ]
    metrics = [
        ("emotion_micro_f1", "Emotion micro-F1"),
        ("emotion_macro_f1", "Emotion macro-F1"),
        ("sentiment_accuracy", "Sentiment accuracy"),
        ("sentiment_macro_f1", "Sentiment macro-F1"),
    ]
    rows = []
    for key, display, in_domain_path in specs:
        stories_row = summary[summary["variant_key"] == key].iloc[0]
        for metric_key, metric_label in metrics:
            rows.append(
                {
                    "model": display,
                    "metric": metric_label,
                    "dataset": "Lemotif held-out",
                    "score": metric_from_payload(in_domain_path, metric_key),
                }
            )
            rows.append(
                {
                    "model": display,
                    "metric": metric_label,
                    "dataset": "Stories external",
                    "score": float(stories_row[metric_key]),
                }
            )
    return pd.DataFrame(rows)


def plot_overall_metrics(summary: pd.DataFrame) -> Path:
    model_order = [
        "BERT baseline",
        "BERT fixed inference",
        "BERT Optuna",
        "BERT full-Lemotif external",
    ]
    frame = summary[summary["model"].isin(model_order)].copy()
    frame["model"] = pd.Categorical(frame["model"], categories=model_order, ordered=True)
    long = frame.melt(
        id_vars=["model"],
        value_vars=[
            "emotion_micro_f1",
            "emotion_macro_f1",
            "sentiment_accuracy",
            "sentiment_macro_f1",
        ],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(
        {
            "emotion_micro_f1": "Emotion micro-F1",
            "emotion_macro_f1": "Emotion macro-F1",
            "sentiment_accuracy": "Sentiment accuracy",
            "sentiment_macro_f1": "Sentiment macro-F1",
        }
    )
    path = FIGURES_DIR / "fig_stories_bert_variant_metrics.png"
    plt.figure(figsize=(11.5, 6.2))
    ax = sns.barplot(data=long, x="metric", y="score", hue="model", palette="deep")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Stories external performance by BERT variant")
    for patch in ax.patches:
        height = patch.get_height()
        if np.isfinite(height) and height > 0.001:
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
    ax.legend(title="", loc="upper right")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_in_domain_vs_stories(frame: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "fig_in_domain_vs_stories_by_variant.png"
    g = sns.catplot(
        data=frame,
        kind="bar",
        x="metric",
        y="score",
        hue="dataset",
        col="model",
        palette=["#3f5f95", "#73a2cc"],
        height=4.1,
        aspect=0.9,
        sharey=True,
    )
    g.set_axis_labels("", "Score")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1.0))
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=28)
        for patch in ax.patches:
            height = patch.get_height()
            if np.isfinite(height) and height > 0.001:
                ax.annotate(
                    f"{height:.3f}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle("In-domain Lemotif vs. Stories external transfer")
    g.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    return path


def plot_cardinality() -> Path:
    prediction_distributions = []
    true_distribution = None
    max_label_count = 0
    for variant in VARIANTS:
        predictions = pd.read_csv(OUTPUT_DIR / str(variant["key"]) / "stories_predictions.csv")
        max_label_count = max(max_label_count, int(predictions["predicted_label_count"].max()))
        if true_distribution is None:
            true_distribution = predictions["true_label_count"].value_counts()
            max_label_count = max(max_label_count, int(predictions["true_label_count"].max()))
        counts = predictions["predicted_label_count"].value_counts()
        prediction_distributions.append((str(variant["display"]), counts))

    label_counts = np.arange(0, max_label_count + 1)
    width = 0.22
    offsets = np.linspace(
        -width * (len(prediction_distributions) - 1) / 2,
        width * (len(prediction_distributions) - 1) / 2,
        len(prediction_distributions),
    )
    path = FIGURES_DIR / "fig_stories_variant_cardinality.png"
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    palette = sns.color_palette("deep", n_colors=len(prediction_distributions))
    true_values = [int(true_distribution.get(count, 0)) for count in label_counts]
    ax.plot(
        label_counts,
        true_values,
        marker="o",
        color="#222222",
        linewidth=2,
        label="True label count",
        zorder=4,
    )
    for offset, (display, counts), color in zip(offsets, prediction_distributions, palette):
        values = [int(counts.get(count, 0)) for count in label_counts]
        ax.bar(
            label_counts + offset,
            values,
            width=width,
            alpha=0.78,
            color=color,
            label=display,
            zorder=3,
        )
    ax.set_xticks(label_counts)
    ax.set_xlabel("Emotion labels per reflection")
    ax.set_ylabel("Rows")
    ax.set_title("Stories prediction-cardinality comparison by BERT variant")
    ax.legend(title="", loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_per_emotion_f1() -> Path:
    frames = []
    for variant in VARIANTS:
        frame = pd.read_csv(OUTPUT_DIR / str(variant["key"]) / "per_emotion_metrics.csv")
        frame["model"] = variant["display"]
        frames.append(frame[["label", "f1", "support", "model"]])
    combined = pd.concat(frames, ignore_index=True)
    supported_labels = (
        combined.groupby("label")["support"].max().loc[lambda series: series > 0].index.tolist()
    )
    combined = combined[combined["label"].isin(supported_labels)]
    order = (
        combined[combined["model"] == "BERT Optuna"]
        .sort_values("f1", ascending=False)["label"]
        .tolist()
    )
    path = FIGURES_DIR / "fig_stories_variant_per_emotion_f1.png"
    plt.figure(figsize=(10.8, 8.0))
    ax = sns.barplot(
        data=combined,
        y="label",
        x="f1",
        hue="model",
        order=order,
        palette="deep",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_title("Stories per-emotion F1 by BERT variant")
    ax.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_sentiment_comparison(summary: pd.DataFrame) -> Path:
    model_order = [
        "BERT baseline",
        "BERT fixed inference",
        "BERT Optuna",
        "Untrained BERT",
        "Gemma sentiment",
    ]
    frame = summary[summary["model"].isin(model_order)].copy()
    frame["model"] = pd.Categorical(frame["model"], categories=model_order, ordered=True)
    long = frame.melt(
        id_vars=["model"],
        value_vars=[
            "sentiment_accuracy",
            "sentiment_macro_precision",
            "sentiment_macro_recall",
            "sentiment_macro_f1",
        ],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(
        {
            "sentiment_accuracy": "Accuracy",
            "sentiment_macro_precision": "Macro precision",
            "sentiment_macro_recall": "Macro recall",
            "sentiment_macro_f1": "Macro F1",
        }
    )
    path = FIGURES_DIR / "fig_stories_sentiment_variant_comparison.png"
    plt.figure(figsize=(11.2, 6.2))
    ax = sns.barplot(data=long, x="metric", y="score", hue="model", palette="deep")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Stories sentiment comparison")
    for patch in ax.patches:
        height = patch.get_height()
        if np.isfinite(height) and height > 0.001:
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
    ax.legend(title="", loc="upper right")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def write_report(summary: pd.DataFrame, figure_paths: list[Path]) -> None:
    fixed = summary[summary["variant_key"] == "fixed_inference"].iloc[0]
    optuna = summary[summary["variant_key"] == "optuna_checkpoint"].iloc[0]
    lines = [
        "# Stories BERT Variant Evaluation",
        "",
        "This evaluation scores the saved Lemotif-trained BERT variants on the same Stories gold-label dataset. The BERT models output the full 18-label Lemotif emotion taxonomy; sentiment metrics are derived from those predicted emotion labels using the existing valence mapping.",
        "",
        "## Key Results",
        "",
        f"- BERT fixed inference on Stories: emotion micro-F1 = {fixed['emotion_micro_f1']:.3f}, emotion macro-F1 = {fixed['emotion_macro_f1']:.3f}, sentiment accuracy = {fixed['sentiment_accuracy']:.3f}, sentiment macro-F1 = {fixed['sentiment_macro_f1']:.3f}, average predicted labels = {fixed['avg_predicted_labels']:.3f}.",
        f"- BERT Optuna on Stories: emotion micro-F1 = {optuna['emotion_micro_f1']:.3f}, emotion macro-F1 = {optuna['emotion_macro_f1']:.3f}, sentiment accuracy = {optuna['sentiment_accuracy']:.3f}, sentiment macro-F1 = {optuna['sentiment_macro_f1']:.3f}, average predicted labels = {optuna['avg_predicted_labels']:.3f}.",
        f"- Fixed inference overreported label count in {int(fixed['overpredicted_rows'])}/114 rows and underreported it in {int(fixed['underpredicted_rows'])}/114 rows.",
        f"- Optuna overreported label count in {int(optuna['overpredicted_rows'])}/114 rows and underreported it in {int(optuna['underpredicted_rows'])}/114 rows.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in figure_paths)
    lines.append("")
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    set_seed(SEED)

    stories_df, text_col, emotion_cols, _ = load_stories_data()
    stories_df = stories_df.reset_index(drop=True)
    label_names = [display_name(col) for col in emotion_cols]
    labels = stories_df[emotion_cols].to_numpy(dtype=float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stories rows: {len(stories_df)}")
    print(f"Emotion labels: {len(label_names)}")
    print(f"Device: {device}")

    rows = []
    for variant in VARIANTS:
        rows.append(
            evaluate_variant(
                variant=variant,
                stories_df=stories_df,
                text_col=text_col,
                label_names=label_names,
                labels=labels,
                device=device,
            )
        )

    summary = pd.DataFrame(rows)
    summary = add_existing_comparator_rows(summary)
    summary.to_csv(SUMMARY_CSV_PATH, index=False)

    in_domain_vs_stories = build_in_domain_vs_stories_frame(summary)
    in_domain_vs_stories.to_csv(OUTPUT_DIR / "in_domain_vs_stories_by_variant.csv", index=False)

    figure_paths = [
        plot_overall_metrics(summary),
        plot_in_domain_vs_stories(in_domain_vs_stories),
        plot_cardinality(),
        plot_per_emotion_f1(),
        plot_sentiment_comparison(summary),
    ]
    write_report(summary, figure_paths)

    print()
    print(summary[[
        "model",
        "emotion_micro_f1",
        "emotion_macro_f1",
        "sentiment_accuracy",
        "sentiment_macro_f1",
        "avg_predicted_labels",
        "overpredicted_rows",
        "underpredicted_rows",
    ]].to_string(index=False))
    print()
    print(f"Saved summary to: {SUMMARY_CSV_PATH}")
    print(f"Saved report to: {SUMMARY_MD_PATH}")
    for path in figure_paths:
        print(f"Saved figure to: {path}")


if __name__ == "__main__":
    main()
