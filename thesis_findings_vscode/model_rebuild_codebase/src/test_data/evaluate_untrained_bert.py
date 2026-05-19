from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name
from test_data.common import TEST_DATA_OUTPUT_DIR, STORIES_CSV_PATH, load_stories_data
from train_bert import (
    BATCH_SIZE,
    LOCAL_FILES_ONLY,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    LemotifEmotionDataset,
    build_predictions_frame,
    compute_multilabel_metrics,
    compute_sentiment_metrics,
    create_label_distribution,
    run_epoch,
    set_seed,
)

UNTRAINED_BERT_DIR = TEST_DATA_OUTPUT_DIR / "untrained_bert"
METRICS_PATH = UNTRAINED_BERT_DIR / "metrics.json"
RUN_SUMMARY_PATH = UNTRAINED_BERT_DIR / "run_summary.txt"
TEST_PREDICTIONS_PATH = UNTRAINED_BERT_DIR / "stories_test_predictions.csv"
STORIES_LABEL_DISTRIBUTION_PATH = UNTRAINED_BERT_DIR / "stories_label_distribution.csv"
CONFIG_SNAPSHOT_PATH = UNTRAINED_BERT_DIR / "config_snapshot.json"

TRAINED_EXTERNAL_METRICS_PATH = PROJECT_ROOT / "test data" / "output" / "external_evaluation" / "metrics.json"
TRAINED_EXTERNAL_MODEL_DIR = PROJECT_ROOT / "test data" / "output" / "external_evaluation" / "model"
TRAINED_BASELINE_METRICS_PATH = PROJECT_ROOT / "output" / "bert_emotion_model" / "metrics.json"
TRAINED_BASELINE_MODEL_DIR = PROJECT_ROOT / "output" / "bert_emotion_model" / "model"


def resolve_threshold() -> tuple[float, str]:
    if TRAINED_EXTERNAL_METRICS_PATH.exists():
        with open(TRAINED_EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return float(metrics.get("threshold", 0.55)), "reused from trained Stories external evaluation"

    if TRAINED_BASELINE_METRICS_PATH.exists():
        with open(TRAINED_BASELINE_METRICS_PATH, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return float(metrics.get("best_threshold", 0.55)), "reused from Lemotif validation baseline"

    return 0.55, "default fixed threshold"


def resolve_local_source() -> tuple[str, str]:
    if TRAINED_BASELINE_MODEL_DIR.exists():
        return str(TRAINED_BASELINE_MODEL_DIR), "local trained baseline model directory"
    if TRAINED_EXTERNAL_MODEL_DIR.exists():
        return str(TRAINED_EXTERNAL_MODEL_DIR), "local trained Stories model directory"
    return MODEL_NAME, "model name fallback"


def build_summary_text(
    stories_path: str,
    label_names: list[str],
    config_source: str,
    source_description: str,
    threshold: float,
    threshold_source: str,
    test_rows: int,
    device: torch.device,
    test_metrics: dict[str, object],
    test_sentiment_metrics: dict[str, object],
) -> str:
    return (
        "Untrained BERT Stories baseline\n"
        "===============================\n"
        f"Stories test data: {stories_path}\n"
        f"Base architecture: {MODEL_NAME}\n"
        f"Tokenizer/config source: {config_source}\n"
        f"Source description: {source_description}\n"
        "Weight initialization: random from config (no Lemotif or Stories training)\n"
        f"Local files only: {LOCAL_FILES_ONLY}\n"
        f"Device: {device}\n"
        f"Emotion labels: {len(label_names)}\n"
        f"Stories test rows: {test_rows}\n"
        f"Threshold: {threshold:.2f} ({threshold_source})\n\n"
        "Stories test emotion metrics\n"
        "---------------------------\n"
        f"Micro F1: {test_metrics['micro']['f1']:.4f}\n"
        f"Macro F1: {test_metrics['macro']['f1']:.4f}\n"
        f"Subset accuracy: {test_metrics['subset_accuracy']:.4f}\n"
        f"Hamming loss: {test_metrics['hamming_loss']:.4f}\n"
        f"Average predicted labels per entry: {test_metrics['avg_predicted_labels_per_entry']:.4f}\n\n"
        "Stories test sentiment metrics\n"
        "------------------------------\n"
        f"Accuracy: {test_sentiment_metrics['accuracy']:.4f}\n"
        f"Macro F1: {test_sentiment_metrics['macro']['f1']:.4f}\n"
    )


def main() -> None:
    UNTRAINED_BERT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    threshold, threshold_source = resolve_threshold()
    config_source, source_description = resolve_local_source()

    stories_df, stories_text_col, emotion_cols, _ = load_stories_data()
    stories_df = stories_df.reset_index(drop=True)

    label_names = [display_name(col) for col in emotion_cols]
    label_distribution = create_label_distribution(stories_df, emotion_cols)
    label_distribution.to_csv(STORIES_LABEL_DISTRIBUTION_PATH, index=False)

    stories_texts = stories_df[stories_text_col].fillna("").astype(str).tolist()
    stories_labels = stories_df[emotion_cols].to_numpy(dtype=float)

    print("Evaluating an untrained BERT model on Stories test data.")
    print(f"Stories test rows: {len(stories_df)}")
    print(f"Tokenizer/config source: {config_source}")
    print("Weight initialization: random from config")
    print(f"Threshold: {threshold:.2f} ({threshold_source})")

    tokenizer = AutoTokenizer.from_pretrained(
        config_source,
        local_files_only=LOCAL_FILES_ONLY,
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        config_source,
        local_files_only=LOCAL_FILES_ONLY,
    )

    label_id_map = {idx: label for idx, label in enumerate(label_names)}
    reverse_label_id_map = {label: idx for idx, label in label_id_map.items()}
    config.num_labels = len(label_names)
    config.problem_type = "multi_label_classification"
    config.id2label = label_id_map
    config.label2id = reverse_label_id_map

    with open(CONFIG_SNAPSHOT_PATH, "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    dataset = LemotifEmotionDataset(
        texts=stories_texts,
        labels=stories_labels,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    _, test_true, test_prob, test_row_ids = run_epoch(
        model=model,
        data_loader=data_loader,
        loss_fn=loss_fn,
        device=device,
    )

    test_metrics = compute_multilabel_metrics(test_true, test_prob, threshold, label_names)
    test_sentiment_metrics = compute_sentiment_metrics(test_true, test_prob, threshold, label_names)

    predictions = build_predictions_frame(
        df=stories_df,
        row_ids=test_row_ids,
        y_true=test_true,
        y_prob=test_prob,
        threshold=threshold,
        label_names=label_names,
        text_col=stories_text_col,
    )
    predictions["predicted_label_count"] = (
        predictions["predicted_emotions"].fillna("none").astype(str).str.split(", ").map(
            lambda labels: 0 if labels == ["none"] else len(labels)
        )
    )
    predictions["mean_probability"] = pd.DataFrame(test_prob).mean(axis=1).to_numpy()
    predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metrics_payload = {
        "stories_test_dataset_path": str(STORIES_CSV_PATH),
        "model_name": MODEL_NAME,
        "evaluation_mode": "untrained_random_init",
        "config_source": config_source,
        "source_description": source_description,
        "local_files_only": LOCAL_FILES_ONLY,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "split_sizes": {"test": len(stories_df)},
        "label_names": label_names,
        "test_metrics": test_metrics,
        "test_sentiment_metrics": test_sentiment_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    summary_text = build_summary_text(
        stories_path=str(STORIES_CSV_PATH),
        label_names=label_names,
        config_source=config_source,
        source_description=source_description,
        threshold=threshold,
        threshold_source=threshold_source,
        test_rows=len(stories_df),
        device=device,
        test_metrics=test_metrics,
        test_sentiment_metrics=test_sentiment_metrics,
    )
    RUN_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print()
    print(summary_text)
    print(f"Saved config snapshot to: {CONFIG_SNAPSHOT_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved Stories test predictions to: {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
