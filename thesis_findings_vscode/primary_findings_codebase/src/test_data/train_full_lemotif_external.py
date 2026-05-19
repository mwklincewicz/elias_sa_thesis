from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name, load_analysis_data, resolve_dataset_path
from test_data.common import EXTERNAL_EVAL_DIR, STORIES_CSV_PATH, load_stories_data
from train_bert import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_FILES_ONLY,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    LemotifEmotionDataset,
    build_pos_weight,
    build_predictions_frame,
    compute_multilabel_metrics,
    compute_sentiment_metrics,
    create_label_distribution,
    run_epoch,
    set_seed,
)

MODEL_DIR = EXTERNAL_EVAL_DIR / "model"
METRICS_PATH = EXTERNAL_EVAL_DIR / "metrics.json"
RUN_SUMMARY_PATH = EXTERNAL_EVAL_DIR / "run_summary.txt"
HISTORY_PATH = EXTERNAL_EVAL_DIR / "training_history.csv"
TEST_PREDICTIONS_PATH = EXTERNAL_EVAL_DIR / "stories_test_predictions.csv"
LEMOTIF_LABEL_DISTRIBUTION_PATH = EXTERNAL_EVAL_DIR / "lemotif_label_distribution.csv"
STORIES_LABEL_DISTRIBUTION_PATH = EXTERNAL_EVAL_DIR / "stories_label_distribution.csv"
BASELINE_METRICS_PATH = CURRENT_FILE.parents[2] / "output" / "bert_emotion_model" / "metrics.json"
BASELINE_MODEL_DIR = CURRENT_FILE.parents[2] / "output" / "bert_emotion_model" / "model"
FULL_TRAIN_EPOCHS = int(os.getenv("LEMOTIF_FULL_DATA_EPOCHS", "1"))


def resolve_threshold() -> tuple[float, str]:
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return float(metrics.get("best_threshold", 0.55)), "reused from prior Lemotif validation run"
    return 0.55, "default fixed threshold"


def build_summary_text(
    lemotif_path: str,
    stories_path: str,
    label_names: list[str],
    initialization_source: str,
    threshold: float,
    threshold_source: str,
    train_rows: int,
    test_rows: int,
    device: torch.device,
    test_metrics: dict[str, object],
    test_sentiment_metrics: dict[str, object],
) -> str:
    return (
        "Full Lemotif training with Stories external test\n"
        "=============================================\n"
        f"Lemotif training data: {lemotif_path}\n"
        f"Stories external test data: {stories_path}\n"
        f"Base model: {MODEL_NAME}\n"
        f"Initialization source: {initialization_source}\n"
        f"Local files only: {LOCAL_FILES_ONLY}\n"
        f"Device: {device}\n"
        f"Emotion labels: {len(label_names)}\n"
        f"Train rows: {train_rows}\n"
        f"External test rows: {test_rows}\n"
        f"Full-data epochs: {FULL_TRAIN_EPOCHS}\n"
        f"Threshold: {threshold:.2f} ({threshold_source})\n\n"
        "External Stories test emotion metrics\n"
        "------------------------------------\n"
        f"Micro F1: {test_metrics['micro']['f1']:.4f}\n"
        f"Macro F1: {test_metrics['macro']['f1']:.4f}\n"
        f"Subset accuracy: {test_metrics['subset_accuracy']:.4f}\n"
        f"Hamming loss: {test_metrics['hamming_loss']:.4f}\n\n"
        "External Stories test sentiment metrics\n"
        "--------------------------------------\n"
        f"Accuracy: {test_sentiment_metrics['accuracy']:.4f}\n"
        f"Macro F1: {test_sentiment_metrics['macro']['f1']:.4f}\n"
    )


def main() -> None:
    EXTERNAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    threshold, threshold_source = resolve_threshold()
    initialization_source = str(BASELINE_MODEL_DIR) if BASELINE_MODEL_DIR.exists() else MODEL_NAME

    lemotif_df, text_col, emotion_cols, _ = load_analysis_data(prefer_cleaned=True)
    lemotif_df = lemotif_df.reset_index(drop=True)
    lemotif_path = resolve_dataset_path(prefer_cleaned=True)

    stories_df, stories_text_col, _, _ = load_stories_data()
    stories_df = stories_df.reset_index(drop=True)

    label_names = [display_name(col) for col in emotion_cols]
    lemotif_label_distribution = create_label_distribution(lemotif_df, emotion_cols)
    stories_label_distribution = create_label_distribution(stories_df, emotion_cols)
    lemotif_label_distribution.to_csv(LEMOTIF_LABEL_DISTRIBUTION_PATH, index=False)
    stories_label_distribution.to_csv(STORIES_LABEL_DISTRIBUTION_PATH, index=False)

    lemotif_texts = lemotif_df[text_col].fillna("").astype(str).tolist()
    lemotif_labels = lemotif_df[emotion_cols].to_numpy(dtype=float)
    stories_texts = stories_df[stories_text_col].fillna("").astype(str).tolist()
    stories_labels = stories_df[emotion_cols].to_numpy(dtype=float)

    print("Training BERT on the full Lemotif dataset and evaluating on Stories external test data.")
    print(f"Lemotif training rows: {len(lemotif_df)}")
    print(f"Stories external test rows: {len(stories_df)}")
    print(f"Initialization source: {initialization_source}")
    print(f"Full-data epochs: {FULL_TRAIN_EPOCHS}")
    print(f"Threshold: {threshold:.2f} ({threshold_source})")

    tokenizer = AutoTokenizer.from_pretrained(
        initialization_source,
        local_files_only=LOCAL_FILES_ONLY,
        use_fast=True,
    )

    train_dataset = LemotifEmotionDataset(
        texts=lemotif_texts,
        labels=lemotif_labels,
        row_ids=list(range(len(lemotif_df))),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    test_dataset = LemotifEmotionDataset(
        texts=stories_texts,
        labels=stories_labels,
        row_ids=list(range(len(stories_df))),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    label_id_map = {idx: label for idx, label in enumerate(label_names)}
    reverse_label_id_map = {label: idx for idx, label in label_id_map.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        initialization_source,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label=label_id_map,
        label2id=reverse_label_id_map,
        ignore_mismatched_sizes=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model.to(device)

    pos_weight = build_pos_weight(lemotif_labels).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_training_steps = len(train_loader) * FULL_TRAIN_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history_rows = []
    for epoch in range(1, FULL_TRAIN_EPOCHS + 1):
        train_loss, train_true, train_prob, _ = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        train_metrics = compute_multilabel_metrics(train_true, train_prob, threshold, label_names)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_micro_f1": train_metrics["micro"]["f1"],
                "train_macro_f1": train_metrics["macro"]["f1"],
                "threshold": threshold,
            }
        )
        print(
            f"Epoch {epoch}/{FULL_TRAIN_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_micro_f1={train_metrics['micro']['f1']:.4f}"
        )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(HISTORY_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    _, test_true, test_prob, test_row_ids = run_epoch(
        model=model,
        data_loader=test_loader,
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
    predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metrics_payload = {
        "training_dataset_path": str(lemotif_path),
        "stories_test_dataset_path": str(STORIES_CSV_PATH),
        "model_name": MODEL_NAME,
        "initialization_source": initialization_source,
        "local_files_only": LOCAL_FILES_ONLY,
        "epochs": FULL_TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "split_sizes": {"train": len(lemotif_df), "test": len(stories_df)},
        "label_names": label_names,
        "test_metrics": test_metrics,
        "test_sentiment_metrics": test_sentiment_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    summary_text = build_summary_text(
        lemotif_path=str(lemotif_path),
        stories_path=str(STORIES_CSV_PATH),
        label_names=label_names,
        initialization_source=initialization_source,
        threshold=threshold,
        threshold_source=threshold_source,
        train_rows=len(lemotif_df),
        test_rows=len(stories_df),
        device=device,
        test_metrics=test_metrics,
        test_sentiment_metrics=test_sentiment_metrics,
    )
    RUN_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print()
    print(summary_text)
    print(f"Saved external evaluation model to: {MODEL_DIR}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved Stories test predictions to: {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
