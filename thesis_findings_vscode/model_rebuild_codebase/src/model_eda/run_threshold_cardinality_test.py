from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BERT_OUTPUT_DIR


THRESHOLD = float(os.getenv("LEMOTIF_CARDINALITY_TEST_THRESHOLD", "0.59"))
MIN_LABELS = int(os.getenv("LEMOTIF_CARDINALITY_TEST_MIN_LABELS", "1"))
TARGET_OVERPREDICTED_MIN = int(os.getenv("LEMOTIF_CARDINALITY_TARGET_MIN", "20"))
TARGET_OVERPREDICTED_MAX = int(os.getenv("LEMOTIF_CARDINALITY_TARGET_MAX", "30"))

INPUT_PREDICTIONS_PATH = BERT_OUTPUT_DIR / "test_predictions.csv"
INPUT_METRICS_PATH = BERT_OUTPUT_DIR / "metrics.json"
OUTPUT_DIR = BERT_OUTPUT_DIR / "threshold_cardinality_test"
ADJUSTED_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions_threshold_0_59_min1.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.txt"
SWEEP_PATH = OUTPUT_DIR / "threshold_sweep.csv"

SENTIMENT_VALENCE = {
    "afraid": -1.0,
    "angry": -1.0,
    "anxious": -1.0,
    "ashamed": -1.0,
    "awkward": -1.0,
    "bored": -1.0,
    "calm": 1.0,
    "confused": 0.0,
    "disgusted": -1.0,
    "excited": 1.0,
    "frustrated": -1.0,
    "happy": 1.0,
    "jealous": -1.0,
    "nostalgic": 0.0,
    "proud": 1.0,
    "sad": -1.0,
    "satisfied": 1.0,
    "surprised": 0.0,
}


def parse_label_set(value: str) -> set[str]:
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "none":
        return set()
    return {part.strip() for part in cleaned.split(",") if part.strip()}


def format_active_labels(binary_labels: np.ndarray, label_names: list[str]) -> str:
    labels = [
        label
        for label, is_active in zip(label_names, binary_labels)
        if int(is_active) == 1
    ]
    return ", ".join(labels) if labels else "none"


def labels_to_matrix(values: pd.Series, label_names: list[str]) -> np.ndarray:
    rows = []
    for value in values:
        active = parse_label_set(value)
        rows.append([1 if label in active else 0 for label in label_names])
    return np.array(rows, dtype=int)


def apply_threshold_with_min_labels(
    probabilities: np.ndarray,
    threshold: float,
    min_labels: int,
) -> np.ndarray:
    predictions = (probabilities >= threshold).astype(int)
    if min_labels <= 0:
        return predictions

    for row_index in range(predictions.shape[0]):
        active_count = int(predictions[row_index].sum())
        if active_count >= min_labels:
            continue

        needed = min_labels - active_count
        ranked_label_indices = np.argsort(probabilities[row_index])[::-1]
        for label_index in ranked_label_indices:
            if predictions[row_index, label_index] == 0:
                predictions[row_index, label_index] = 1
                needed -= 1
            if needed == 0:
                break

    return predictions


def derive_sentiment(binary_labels: np.ndarray, label_names: list[str]) -> tuple[str, float]:
    scores = [
        SENTIMENT_VALENCE[label.lower()]
        for label, is_active in zip(label_names, binary_labels)
        if int(is_active) == 1
    ]
    if not scores:
        return "neutral", 0.0

    score = float(np.mean(scores))
    if score > 0.25:
        return "positive", score
    if score < -0.25:
        return "negative", score
    return "neutral", score


def compute_multilabel_metrics(
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
    label_precision, label_recall, label_f1, label_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    label_metrics = []
    for label, precision, recall, f1, support in zip(
        label_names,
        label_precision,
        label_recall,
        label_f1,
        label_support,
    ):
        label_metrics.append(
            {
                "label": label,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )

    surplus = y_pred.sum(axis=1) - y_true.sum(axis=1)
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
        "subset_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(np.not_equal(y_true, y_pred).mean()),
        "avg_true_labels_per_entry": float(y_true.sum(axis=1).mean()),
        "avg_predicted_labels_per_entry": float(y_pred.sum(axis=1).mean()),
        "zero_predicted_rows": int((y_pred.sum(axis=1) == 0).sum()),
        "overpredicted_by_exactly_2_rows": int((surplus == 2).sum()),
        "overpredicted_by_at_least_2_rows": int((surplus >= 2).sum()),
        "overpredicted_any_rows": int((surplus > 0).sum()),
        "underpredicted_any_rows": int((surplus < 0).sum()),
        "label_metrics": label_metrics,
    }


def compute_sentiment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, object]:
    true_sentiments = []
    predicted_sentiments = []

    for true_row, pred_row in zip(y_true, y_pred):
        true_sentiment, _ = derive_sentiment(true_row, label_names)
        predicted_sentiment, _ = derive_sentiment(pred_row, label_names)
        true_sentiments.append(true_sentiment)
        predicted_sentiments.append(predicted_sentiment)

    ordered_labels = ["negative", "neutral", "positive"]
    precision, recall, f1, support = precision_recall_fscore_support(
        true_sentiments,
        predicted_sentiments,
        labels=ordered_labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_sentiments,
        predicted_sentiments,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(true_sentiments, predicted_sentiments)),
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
    }


def build_adjusted_predictions(
    predictions: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    adjusted = predictions.copy()
    adjusted["predicted_emotions_original"] = adjusted["predicted_emotions"]
    adjusted["predicted_sentiment_original"] = adjusted["predicted_sentiment"]
    adjusted["predicted_emotions"] = [
        format_active_labels(row, label_names) for row in y_pred
    ]

    sentiments = [derive_sentiment(row, label_names) for row in y_pred]
    adjusted["predicted_sentiment"] = [sentiment for sentiment, _ in sentiments]
    adjusted["predicted_sentiment_score"] = [score for _, score in sentiments]
    adjusted["true_label_count"] = y_true.sum(axis=1)
    adjusted["predicted_label_count"] = y_pred.sum(axis=1)
    adjusted["prediction_surplus"] = (
        adjusted["predicted_label_count"] - adjusted["true_label_count"]
    )
    adjusted["hyperparameter_testing_change"] = (
        f"threshold={THRESHOLD:.2f}; min_predicted_labels={MIN_LABELS}"
    )
    return adjusted


def build_threshold_sweep(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.50, 0.76, 0.01):
        y_pred = apply_threshold_with_min_labels(probabilities, float(threshold), MIN_LABELS)
        metrics = compute_multilabel_metrics(y_true, y_pred, label_names)
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "min_predicted_labels": MIN_LABELS,
                "emotion_micro_f1": metrics["micro"]["f1"],
                "emotion_macro_f1": metrics["macro"]["f1"],
                "avg_predicted_labels_per_entry": metrics["avg_predicted_labels_per_entry"],
                "zero_predicted_rows": metrics["zero_predicted_rows"],
                "overpredicted_by_at_least_2_rows": metrics[
                    "overpredicted_by_at_least_2_rows"
                ],
                "overpredicted_by_exactly_2_rows": metrics[
                    "overpredicted_by_exactly_2_rows"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_summary_text(
    baseline_metrics: dict,
    adjusted_metrics: dict,
    adjusted_sentiment_metrics: dict,
) -> str:
    target_met = (
        TARGET_OVERPREDICTED_MIN
        <= adjusted_metrics["overpredicted_by_at_least_2_rows"]
        <= TARGET_OVERPREDICTED_MAX
    )
    return (
        "Hyperparameter testing change: threshold/cardinality adjustment\n"
        "==============================================================\n"
        f"Input predictions: {INPUT_PREDICTIONS_PATH}\n"
        f"Original validation-selected threshold: {baseline_metrics.get('best_threshold')}\n"
        f"Tested threshold: {THRESHOLD:.2f}\n"
        f"Minimum predicted labels per row: {MIN_LABELS}\n"
        f"Target rows with >=2 extra predicted emotions: "
        f"{TARGET_OVERPREDICTED_MIN}-{TARGET_OVERPREDICTED_MAX}\n"
        f"Target met: {target_met}\n\n"
        "Adjusted emotion metrics\n"
        "------------------------\n"
        f"Micro F1: {adjusted_metrics['micro']['f1']:.4f}\n"
        f"Macro F1: {adjusted_metrics['macro']['f1']:.4f}\n"
        f"Average predicted labels per entry: "
        f"{adjusted_metrics['avg_predicted_labels_per_entry']:.4f}\n"
        f"Rows with zero predicted emotions: {adjusted_metrics['zero_predicted_rows']}\n"
        f"Rows with exactly 2 extra predicted emotions: "
        f"{adjusted_metrics['overpredicted_by_exactly_2_rows']}\n"
        f"Rows with at least 2 extra predicted emotions: "
        f"{adjusted_metrics['overpredicted_by_at_least_2_rows']}\n"
        f"Rows with any overprediction: {adjusted_metrics['overpredicted_any_rows']}\n"
        f"Rows with any underprediction: {adjusted_metrics['underpredicted_any_rows']}\n\n"
        "Adjusted derived sentiment metrics\n"
        "----------------------------------\n"
        f"Accuracy: {adjusted_sentiment_metrics['accuracy']:.4f}\n"
        f"Macro F1: {adjusted_sentiment_metrics['macro']['f1']:.4f}\n\n"
        "Annotation\n"
        "----------\n"
        "This is a hyperparameter-testing change applied after model training. "
        "It changes the inference threshold and cardinality fallback, but not the "
        "fine-tuned BERT weights.\n"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(INPUT_PREDICTIONS_PATH).fillna("")
    with open(INPUT_METRICS_PATH, "r", encoding="utf-8") as handle:
        baseline_metrics = json.load(handle)

    label_names = baseline_metrics["label_names"]
    probability_columns = [column for column in predictions.columns if column.startswith("prob_")]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    y_true = labels_to_matrix(predictions["true_emotions"], label_names)
    y_pred = apply_threshold_with_min_labels(probabilities, THRESHOLD, MIN_LABELS)

    adjusted_metrics = compute_multilabel_metrics(y_true, y_pred, label_names)
    adjusted_sentiment_metrics = compute_sentiment_metrics(y_true, y_pred, label_names)
    adjusted_predictions = build_adjusted_predictions(predictions, y_true, y_pred, label_names)
    threshold_sweep = build_threshold_sweep(probabilities, y_true, label_names)

    metrics_payload = {
        "annotation": "Hyperparameter testing change",
        "model_weights": "unchanged fine-tuned BERT",
        "input_predictions_path": str(INPUT_PREDICTIONS_PATH),
        "baseline_threshold": baseline_metrics.get("best_threshold"),
        "tested_threshold": THRESHOLD,
        "minimum_predicted_labels": MIN_LABELS,
        "target_overpredicted_by_at_least_2_rows": [
            TARGET_OVERPREDICTED_MIN,
            TARGET_OVERPREDICTED_MAX,
        ],
        "test_metrics": adjusted_metrics,
        "test_sentiment_metrics": adjusted_sentiment_metrics,
    }

    adjusted_predictions.to_csv(ADJUSTED_PREDICTIONS_PATH, index=False)
    threshold_sweep.to_csv(SWEEP_PATH, index=False)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    SUMMARY_PATH.write_text(
        build_summary_text(baseline_metrics, adjusted_metrics, adjusted_sentiment_metrics),
        encoding="utf-8",
    )

    print(SUMMARY_PATH.read_text(encoding="utf-8"))
    print(f"Saved adjusted predictions to: {ADJUSTED_PREDICTIONS_PATH}")
    print(f"Saved threshold sweep to: {SWEEP_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
