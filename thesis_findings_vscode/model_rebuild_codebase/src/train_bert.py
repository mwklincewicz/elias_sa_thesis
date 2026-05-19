from __future__ import annotations

import copy
import json
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import BERT_OUTPUT_DIR
from data_loader import display_name, load_analysis_data, resolve_dataset_path, safe_name

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("LEMOTIF_BERT_MODEL", "bert-base-uncased")
LOCAL_FILES_ONLY = os.getenv("LEMOTIF_LOCAL_FILES_ONLY", "0") == "1"

SEED = int(os.getenv("LEMOTIF_SEED", "42"))
TRAIN_SIZE = float(os.getenv("LEMOTIF_TRAIN_SIZE", "0.70"))
VAL_SIZE = float(os.getenv("LEMOTIF_VAL_SIZE", "0.15"))
TEST_SIZE = float(os.getenv("LEMOTIF_TEST_SIZE", "0.15"))

MAX_LENGTH = int(os.getenv("LEMOTIF_MAX_LENGTH", "192"))
BATCH_SIZE = int(os.getenv("LEMOTIF_BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("LEMOTIF_EPOCHS", "4"))
LEARNING_RATE = float(os.getenv("LEMOTIF_LEARNING_RATE", "2e-5"))
WEIGHT_DECAY = float(os.getenv("LEMOTIF_WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("LEMOTIF_WARMUP_RATIO", "0.10"))
MAX_POS_WEIGHT = float(os.getenv("LEMOTIF_MAX_POS_WEIGHT", "20.0"))

MODEL_DIR = BERT_OUTPUT_DIR / "model"
METRICS_PATH = BERT_OUTPUT_DIR / "metrics.json"
RUN_SUMMARY_PATH = BERT_OUTPUT_DIR / "run_summary.txt"
HISTORY_PATH = BERT_OUTPUT_DIR / "training_history.csv"
LABEL_DISTRIBUTION_PATH = BERT_OUTPUT_DIR / "label_distribution.csv"
THRESHOLD_SCAN_PATH = BERT_OUTPUT_DIR / "validation_threshold_scan.csv"
TEST_PREDICTIONS_PATH = BERT_OUTPUT_DIR / "test_predictions.csv"

# Sentiment is derived from the emotion taxonomy, matching the thesis framing:
# the model predicts emotions, and sentiment is summarized afterward.
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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


class LemotifEmotionDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        row_ids: list[int],
        tokenizer,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels.astype(np.float32)
        self.row_ids = row_ids

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        item["row_id"] = torch.tensor(self.row_ids[index], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_stratify_keys(label_matrix: np.ndarray) -> list[str] | None:
    label_counts = label_matrix.sum(axis=0)
    keys: list[str] = []

    for row in label_matrix:
        active_labels = np.flatnonzero(row > 0)
        if active_labels.size == 0:
            keys.append("no_label")
            continue

        rarest_label = active_labels[np.argmin(label_counts[active_labels])]
        keys.append(f"label_{int(rarest_label)}")

    if len(keys) < 2:
        return None

    key_counts = Counter(keys)
    if min(key_counts.values()) < 2:
        return None

    return keys


def split_indices(label_matrix: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.arange(len(label_matrix))

    first_stage_keys = build_stratify_keys(label_matrix)
    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=VAL_SIZE + TEST_SIZE,
            random_state=SEED,
            shuffle=True,
            stratify=first_stage_keys,
        )
    except ValueError:
        print("Falling back to an unstratified train/validation/test split.")
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=VAL_SIZE + TEST_SIZE,
            random_state=SEED,
            shuffle=True,
            stratify=None,
        )

    second_stage_keys = build_stratify_keys(label_matrix[temp_idx])
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_size,
            random_state=SEED,
            shuffle=True,
            stratify=second_stage_keys,
        )
    except ValueError:
        print("Falling back to an unstratified validation/test split.")
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_size,
            random_state=SEED,
            shuffle=True,
            stratify=None,
        )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def create_label_distribution(
    df: pd.DataFrame,
    emotion_cols: list[str],
) -> pd.DataFrame:
    total_rows = len(df)
    rows = []

    for col in emotion_cols:
        positive_count = int(df[col].sum())
        rows.append(
            {
                "column": col,
                "label": display_name(col),
                "positive_rows": positive_count,
                "prevalence": positive_count / total_rows,
            }
        )

    label_distribution = pd.DataFrame(rows).sort_values(
        by="positive_rows",
        ascending=False,
    )
    return label_distribution


# ---------------------------------------------------------------------------
# Metrics and sentiment helpers
# ---------------------------------------------------------------------------


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    scan_rows = []
    best_threshold = 0.50
    best_micro_f1 = -1.0

    for threshold in np.arange(0.10, 0.95, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        _, _, micro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        scan_rows.append(
            {
                "threshold": float(threshold),
                "micro_f1": float(micro_f1),
                "macro_f1": float(macro_f1),
            }
        )

        if micro_f1 > best_micro_f1:
            best_micro_f1 = float(micro_f1)
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame(scan_rows)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label_names: list[str],
) -> dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)

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
    samples_precision, samples_recall, samples_f1, _ = precision_recall_fscore_support(
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

    label_metrics = []
    for name, precision, recall, f1, support in zip(
        label_names,
        label_precision,
        label_recall,
        label_f1,
        label_support,
    ):
        label_metrics.append(
            {
                "label": name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )

    return {
        "threshold": float(threshold),
        "subset_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(np.not_equal(y_true, y_pred).mean()),
        "avg_true_labels_per_entry": float(y_true.sum(axis=1).mean()),
        "avg_predicted_labels_per_entry": float(y_pred.sum(axis=1).mean()),
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
        "label_metrics": label_metrics,
    }


def derive_sentiment(binary_labels: np.ndarray, label_names: list[str]) -> tuple[str, float]:
    active_scores = [
        SENTIMENT_VALENCE[label_name.lower()]
        for label_name, is_active in zip(label_names, binary_labels)
        if int(is_active) == 1
    ]

    if not active_scores:
        return "neutral", 0.0

    score = float(np.mean(active_scores))
    if score > 0.25:
        return "positive", score
    if score < -0.25:
        return "negative", score
    return "neutral", score


def compute_sentiment_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label_names: list[str],
) -> dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)

    true_sentiments = []
    pred_sentiments = []

    for true_row, pred_row in zip(y_true, y_pred):
        true_sentiment, _ = derive_sentiment(true_row, label_names)
        pred_sentiment, _ = derive_sentiment(pred_row, label_names)
        true_sentiments.append(true_sentiment)
        pred_sentiments.append(pred_sentiment)

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

    label_metrics = []
    for label, label_precision, label_recall, label_f1, label_support in zip(
        ordered_labels,
        precision,
        recall,
        f1,
        support,
    ):
        label_metrics.append(
            {
                "label": label,
                "precision": float(label_precision),
                "recall": float(label_recall),
                "f1": float(label_f1),
                "support": int(label_support),
            }
        )

    return {
        "accuracy": float(accuracy_score(true_sentiments, pred_sentiments)),
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "label_metrics": label_metrics,
    }


def format_active_labels(binary_labels: np.ndarray, label_names: list[str]) -> str:
    active_labels = [
        label_name
        for label_name, is_active in zip(label_names, binary_labels)
        if int(is_active) == 1
    ]
    if not active_labels:
        return "none"
    return ", ".join(active_labels)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def build_pos_weight(train_labels: np.ndarray) -> torch.Tensor:
    positive = train_labels.sum(axis=0)
    negative = len(train_labels) - positive

    raw_weights = negative / np.clip(positive, a_min=1.0, a_max=None)
    clipped_weights = np.clip(raw_weights, a_min=1.0, a_max=MAX_POS_WEIGHT)
    return torch.tensor(clipped_weights, dtype=torch.float32)


def run_epoch(
    model,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scheduler=None,
) -> tuple[float, np.ndarray, np.ndarray, list[int]]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_row_ids: list[int] = []

    context_manager = torch.enable_grad() if is_training else torch.no_grad()
    with context_manager:
        for batch in data_loader:
            labels = batch.pop("labels").to(device)
            row_ids = batch.pop("row_id").tolist()
            inputs = {key: value.to(device) for key, value in batch.items()}

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_row_ids.extend(int(row_id) for row_id in row_ids)

    average_loss = total_loss / len(data_loader.dataset)
    return average_loss, np.vstack(all_labels), np.vstack(all_probs), all_row_ids


def build_predictions_frame(
    df: pd.DataFrame,
    row_ids: list[int],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label_names: list[str],
    text_col: str,
) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    rows = []

    for position, row_id in enumerate(row_ids):
        true_sentiment, true_sentiment_score = derive_sentiment(y_true[position], label_names)
        pred_sentiment, pred_sentiment_score = derive_sentiment(y_pred[position], label_names)

        row = {
            "row_id": int(row_id),
            "text": df.iloc[row_id][text_col],
            "true_emotions": format_active_labels(y_true[position], label_names),
            "predicted_emotions": format_active_labels(y_pred[position], label_names),
            "true_sentiment": true_sentiment,
            "predicted_sentiment": pred_sentiment,
            "true_sentiment_score": true_sentiment_score,
            "predicted_sentiment_score": pred_sentiment_score,
        }

        for label_idx, label_name in enumerate(label_names):
            row[f"prob_{safe_name(label_name.lower())}"] = float(y_prob[position, label_idx])

        rows.append(row)

    return pd.DataFrame(rows)


def build_run_summary(
    dataset_path: str,
    label_names: list[str],
    split_sizes: dict[str, int],
    device: torch.device,
    best_epoch: int,
    best_threshold: float,
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
    test_sentiment_metrics: dict[str, object],
) -> str:
    return (
        "Local Lemotif BERT baseline\n"
        "===========================\n"
        f"Dataset used: {dataset_path}\n"
        f"Base model: {MODEL_NAME}\n"
        f"Local files only: {LOCAL_FILES_ONLY}\n"
        f"Device: {device}\n"
        f"Emotion labels: {len(label_names)}\n"
        f"Split sizes: train={split_sizes['train']}, val={split_sizes['val']}, test={split_sizes['test']}\n"
        f"Best epoch: {best_epoch}\n"
        f"Validation-selected threshold: {best_threshold:.2f}\n\n"
        "Validation emotion metrics\n"
        "--------------------------\n"
        f"Micro F1: {validation_metrics['micro']['f1']:.4f}\n"
        f"Macro F1: {validation_metrics['macro']['f1']:.4f}\n"
        f"Subset accuracy: {validation_metrics['subset_accuracy']:.4f}\n\n"
        "Test emotion metrics\n"
        "--------------------\n"
        f"Micro F1: {test_metrics['micro']['f1']:.4f}\n"
        f"Macro F1: {test_metrics['macro']['f1']:.4f}\n"
        f"Subset accuracy: {test_metrics['subset_accuracy']:.4f}\n"
        f"Hamming loss: {test_metrics['hamming_loss']:.4f}\n\n"
        "Test sentiment metrics (derived from predicted emotions)\n"
        "-------------------------------------------------------\n"
        f"Accuracy: {test_sentiment_metrics['accuracy']:.4f}\n"
        f"Macro F1: {test_sentiment_metrics['macro']['f1']:.4f}\n"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1.0):
        raise ValueError("TRAIN_SIZE + VAL_SIZE + TEST_SIZE must sum to 1.0.")

    BERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    df, text_col, emotion_cols, _ = load_analysis_data(prefer_cleaned=True)
    df = df.reset_index(drop=True)
    dataset_path = resolve_dataset_path(prefer_cleaned=True)

    label_names = [display_name(col) for col in emotion_cols]
    label_distribution = create_label_distribution(df, emotion_cols)
    label_distribution.to_csv(LABEL_DISTRIBUTION_PATH, index=False)

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[emotion_cols].to_numpy(dtype=np.float32)
    split_map = split_indices(labels)
    split_sizes = {name: len(indices) for name, indices in split_map.items()}

    print("Training a local single-entry BERT baseline for Lemotif emotion recognition.")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {MODEL_NAME}")
    print(
        "Split sizes: "
        f"train={split_sizes['train']} | val={split_sizes['val']} | test={split_sizes['test']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
        use_fast=True,
    )

    datasets = {}
    for split_name, row_ids in split_map.items():
        datasets[split_name] = LemotifEmotionDataset(
            texts=[texts[idx] for idx in row_ids],
            labels=labels[row_ids],
            row_ids=row_ids.tolist(),
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    label_id_map = {idx: label for idx, label in enumerate(label_names)}
    reverse_label_id_map = {label: idx for idx, label in label_id_map.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label=label_id_map,
        label2id=reverse_label_id_map,
        ignore_mismatched_sizes=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model.to(device)

    pos_weight = build_pos_weight(labels[split_map["train"]]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_training_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_epoch = 0
    best_threshold = 0.50
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_micro_f1 = -1.0
    best_threshold_scan = pd.DataFrame()
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_true, train_prob, _ = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_loss, val_true, val_prob, _ = run_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_threshold, threshold_scan = find_best_threshold(val_true, val_prob)
        train_metrics = compute_multilabel_metrics(
            train_true,
            train_prob,
            epoch_threshold,
            label_names,
        )
        val_metrics = compute_multilabel_metrics(
            val_true,
            val_prob,
            epoch_threshold,
            label_names,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "threshold": epoch_threshold,
                "train_micro_f1": train_metrics["micro"]["f1"],
                "train_macro_f1": train_metrics["macro"]["f1"],
                "val_micro_f1": val_metrics["micro"]["f1"],
                "val_macro_f1": val_metrics["macro"]["f1"],
            }
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_micro_f1={val_metrics['micro']['f1']:.4f} | "
            f"threshold={epoch_threshold:.2f}"
        )

        if val_metrics["micro"]["f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro"]["f1"]
            best_epoch = epoch
            best_threshold = epoch_threshold
            best_model_state = copy.deepcopy(model.state_dict())
            best_threshold_scan = threshold_scan.copy()

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(HISTORY_PATH, index=False)
    best_threshold_scan.to_csv(THRESHOLD_SCAN_PATH, index=False)

    model.load_state_dict(best_model_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    _, final_val_true, final_val_prob, final_val_row_ids = run_epoch(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )
    _, test_true, test_prob, test_row_ids = run_epoch(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    validation_metrics = compute_multilabel_metrics(
        final_val_true,
        final_val_prob,
        best_threshold,
        label_names,
    )
    test_metrics = compute_multilabel_metrics(
        test_true,
        test_prob,
        best_threshold,
        label_names,
    )
    validation_sentiment_metrics = compute_sentiment_metrics(
        final_val_true,
        final_val_prob,
        best_threshold,
        label_names,
    )
    test_sentiment_metrics = compute_sentiment_metrics(
        test_true,
        test_prob,
        best_threshold,
        label_names,
    )

    test_predictions = build_predictions_frame(
        df=df,
        row_ids=test_row_ids,
        y_true=test_true,
        y_prob=test_prob,
        threshold=best_threshold,
        label_names=label_names,
        text_col=text_col,
    )
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metrics_payload = {
        "dataset_path": str(dataset_path),
        "model_name": MODEL_NAME,
        "local_files_only": LOCAL_FILES_ONLY,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "label_names": label_names,
        "split_sizes": split_sizes,
        "validation_metrics": validation_metrics,
        "validation_sentiment_metrics": validation_sentiment_metrics,
        "test_metrics": test_metrics,
        "test_sentiment_metrics": test_sentiment_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)

    summary_text = build_run_summary(
        dataset_path=str(dataset_path),
        label_names=label_names,
        split_sizes=split_sizes,
        device=device,
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_sentiment_metrics=test_sentiment_metrics,
    )
    RUN_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print()
    print(summary_text)
    print(f"Saved model to: {MODEL_DIR}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved test predictions to: {TEST_PREDICTIONS_PATH}")
    print(f"Saved label distribution to: {LABEL_DISTRIBUTION_PATH}")
    print(f"Saved training history to: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
