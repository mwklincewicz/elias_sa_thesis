from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3nForCausalLM, Gemma3nTextConfig

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name, load_analysis_data, resolve_dataset_path, safe_name
from train_bert import (
    compute_multilabel_metrics,
    compute_sentiment_metrics,
    create_label_distribution,
    derive_sentiment,
    format_active_labels,
    set_seed,
    split_indices,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = CURRENT_FILE.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "gemma_2b_emotion_model"
ZERO_SHOT_DIR = OUTPUT_DIR / "zero_shot"
FINETUNED_DIR = OUTPUT_DIR / "finetuned"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = FINETUNED_DIR / "adapter"

for path in [OUTPUT_DIR, ZERO_SHOT_DIR, FINETUNED_DIR, COMPARISON_DIR, FIGURES_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)


MODEL_ID = os.getenv("GEMMA_EMOTION_MODEL_ID", "google/gemma-3n-E2B-it")
LOCAL_FILES_ONLY = os.getenv("GEMMA_LOCAL_FILES_ONLY", "1") == "1"
SEED = int(os.getenv("GEMMA_EMOTION_SEED", "42"))
MAX_LENGTH = int(os.getenv("GEMMA_EMOTION_MAX_LENGTH", "192"))
MAX_NEW_TOKENS = int(os.getenv("GEMMA_EMOTION_MAX_NEW_TOKENS", "32"))
TRAIN_BATCH_SIZE = int(os.getenv("GEMMA_EMOTION_TRAIN_BATCH_SIZE", "2"))
TRAIN_GRAD_ACCUM_STEPS = int(os.getenv("GEMMA_EMOTION_GRAD_ACCUM_STEPS", "4"))
EVAL_BATCH_SIZE = int(os.getenv("GEMMA_EMOTION_EVAL_BATCH_SIZE", "2"))
EPOCHS = int(os.getenv("GEMMA_EMOTION_EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("GEMMA_EMOTION_LR", "2e-4"))
WEIGHT_DECAY = float(os.getenv("GEMMA_EMOTION_WEIGHT_DECAY", "0.0"))
LORA_R = int(os.getenv("GEMMA_EMOTION_LORA_R", "8"))
LORA_ALPHA = int(os.getenv("GEMMA_EMOTION_LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("GEMMA_EMOTION_LORA_DROPOUT", "0.05"))
GPU_MAX_MEMORY_GB = float(os.getenv("GEMMA_GPU_MAX_MEMORY_GB", "3.5"))
MAX_ROWS_PER_SPLIT = int(os.getenv("GEMMA_EMOTION_MAX_ROWS", "0"))
SKIP_ZERO_SHOT = os.getenv("GEMMA_EMOTION_SKIP_ZERO_SHOT", "0") == "1"
SKIP_FINETUNE = os.getenv("GEMMA_EMOTION_SKIP_FINETUNE", "0") == "1"
USE_GRADIENT_CHECKPOINTING = os.getenv("GEMMA_EMOTION_GRADIENT_CHECKPOINTING", "0") == "1"

BERT_BASELINE_METRICS_PATH = PROJECT_ROOT / "output" / "bert_emotion_model" / "metrics.json"
SENTIMENT_ORDER = ["negative", "neutral", "positive"]

sns.set_theme(style="whitegrid")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_limit_split(df: pd.DataFrame) -> pd.DataFrame:
    if MAX_ROWS_PER_SPLIT <= 0:
        return df
    return df.head(MAX_ROWS_PER_SPLIT).reset_index(drop=True)


def model_snapshot_path() -> str:
    return snapshot_download(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)


def load_text_config(snapshot_path: str) -> Gemma3nTextConfig:
    config_path = Path(snapshot_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        full_config = json.load(handle)
    return Gemma3nTextConfig.from_dict(full_config["text_config"])


def get_device_config() -> dict[str, object]:
    if torch.cuda.is_available():
        return {
            "label": f"cuda:{torch.cuda.current_device()}",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            "device_map": {"": torch.cuda.current_device()},
            "max_memory": None,
            "dtype": torch.float16,
        }

    return {
        "label": "cpu",
        "quantization_config": None,
        "device_map": "cpu",
        "max_memory": None,
        "dtype": torch.float32,
    }


def emotion_label_names(emotion_cols: list[str]) -> list[str]:
    return [display_name(col) for col in emotion_cols]


def sorted_label_string(binary_row: np.ndarray, label_names: list[str]) -> str:
    active = [label for label, flag in zip(label_names, binary_row) if int(flag) == 1]
    return ", ".join(active) if active else "none"


def build_emotion_prompt(text: str, label_names: list[str]) -> str:
    label_block = ", ".join(label_names)
    return (
        "Label the diary reflection with every matching emotion from this list only:\n"
        f"{label_block}\n"
        "Return only the exact label names as a comma-separated list, or none.\n\n"
        f"Reflection: {text.strip()}\n"
    )


def tokenize_generation_prompt(tokenizer, prompt: str):
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt")


def build_generation_batch(tokenizer, prompts: list[str]) -> dict[str, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        rendered_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        return tokenizer(
            rendered_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )

    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )


def normalize_generated_labels(raw_text: str, label_names: list[str]) -> list[str]:
    found = []
    lowered = raw_text.lower()
    label_patterns = sorted(label_names, key=len, reverse=True)

    for label in label_patterns:
        pattern = rf"(?<![a-z]){re.escape(label.lower())}(?![a-z])"
        if re.search(pattern, lowered):
            found.append(label)

    ordered = [label for label in label_names if label in found]
    return ordered


def labels_to_binary(labels: list[str], label_names: list[str]) -> np.ndarray:
    label_set = set(labels)
    return np.array([1 if label in label_set else 0 for label in label_names], dtype=int)


def build_prediction_rows(
    df: pd.DataFrame,
    row_ids: list[int],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    raw_outputs: list[str],
    latencies: list[float],
    text_col: str,
    label_names: list[str],
) -> pd.DataFrame:
    rows = []
    for position, row_id in enumerate(row_ids):
        true_sentiment, true_score = derive_sentiment(true_labels[position], label_names)
        pred_sentiment, pred_score = derive_sentiment(predicted_labels[position], label_names)
        rows.append(
            {
                "row_id": int(row_id),
                "text": df.iloc[row_id][text_col],
                "true_emotions": format_active_labels(true_labels[position], label_names),
                "predicted_emotions": format_active_labels(predicted_labels[position], label_names),
                "true_sentiment": true_sentiment,
                "predicted_sentiment": pred_sentiment,
                "true_sentiment_score": true_score,
                "predicted_sentiment_score": pred_score,
                "raw_model_output": raw_outputs[position],
                "latency_seconds": latencies[position],
            }
        )
    return pd.DataFrame(rows)


@dataclass
class GenerationResult:
    row_ids: list[int]
    y_true: np.ndarray
    y_pred: np.ndarray
    raw_outputs: list[str]
    latencies: list[float]


class GemmaSFTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        emotion_cols: list[str],
        tokenizer,
        label_names: list[str],
        max_length: int,
    ) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        for row_id, row in df.reset_index(drop=True).iterrows():
            reflection = str(row[text_col]).strip()
            target = sorted_label_string(row[emotion_cols].to_numpy(dtype=int), label_names)
            prompt = build_emotion_prompt(reflection, label_names)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            tokenized = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.copy()
            prompt_length = min(len(prompt_ids), max_length)

            for idx in range(prompt_length):
                labels[idx] = -100
            for idx, mask in enumerate(attention_mask):
                if mask == 0:
                    labels[idx] = -100

            self.items.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "row_id": torch.tensor(row_id, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def load_base_model_for_training(snapshot_path: str, device_config: dict[str, object]):
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_config = load_text_config(snapshot_path)

    model = Gemma3nForCausalLM.from_pretrained(
        snapshot_path,
        config=text_config,
        local_files_only=True,
        quantization_config=device_config["quantization_config"],
        device_map=device_config["device_map"],
        max_memory=device_config["max_memory"],
        dtype=device_config["dtype"],
        offload_buffers=True,
    )
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "altup_coef_clip"):
        model.config.text_config.altup_coef_clip = None
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "altup_coef_clip"):
            module.config.altup_coef_clip = None
    model.config.use_cache = False
    if USE_GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    )
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return tokenizer, model


def load_model_for_generation(snapshot_path: str, adapter_dir: Path | None, device_config: dict[str, object]):
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_config = load_text_config(snapshot_path)

    model = Gemma3nForCausalLM.from_pretrained(
        snapshot_path,
        config=text_config,
        local_files_only=True,
        quantization_config=device_config["quantization_config"],
        device_map=device_config["device_map"],
        max_memory=device_config["max_memory"],
        dtype=device_config["dtype"],
        low_cpu_mem_usage=True,
        offload_buffers=True,
    )
    if adapter_dir is not None and adapter_dir.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)

    model.config.use_cache = True
    return tokenizer, model


def generate_predictions(
    model,
    tokenizer,
    df: pd.DataFrame,
    text_col: str,
    emotion_cols: list[str],
    label_names: list[str],
) -> GenerationResult:
    row_ids: list[int] = []
    true_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    raw_outputs: list[str] = []
    latencies: list[float] = []

    model.eval()
    df_local = df.reset_index(drop=True)
    for start in range(0, len(df_local), EVAL_BATCH_SIZE):
        stop = min(start + EVAL_BATCH_SIZE, len(df_local))
        batch_df = df_local.iloc[start:stop]
        prompts = [
            build_emotion_prompt(str(row[text_col]).strip(), label_names)
            for _, row in batch_df.iterrows()
        ]
        generation_inputs = build_generation_batch(tokenizer, prompts)
        generation_inputs = {key: value.to(model.device) for key, value in generation_inputs.items()}
        prompt_token_counts = generation_inputs["attention_mask"].sum(dim=1).tolist()

        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **generation_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                logits_to_keep=1,
            )
        batch_latency = time.perf_counter() - started
        per_example_latency = batch_latency / len(batch_df)

        for local_offset, (_, row) in enumerate(batch_df.iterrows()):
            prompt_tokens = int(prompt_token_counts[local_offset])
            generated_tokens = generated[local_offset][prompt_tokens:]
            raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            parsed_labels = normalize_generated_labels(raw_output, label_names)

            row_ids.append(int(start + local_offset))
            true_rows.append(row[emotion_cols].to_numpy(dtype=int))
            pred_rows.append(labels_to_binary(parsed_labels, label_names))
            raw_outputs.append(raw_output)
            latencies.append(per_example_latency)

            print(
                f"{start + local_offset + 1}/{len(df_local)} | "
                f"true={format_active_labels(true_rows[-1], label_names)} | "
                f"pred={format_active_labels(pred_rows[-1], label_names)} | "
                f"latency={per_example_latency:.2f}s"
            )

    return GenerationResult(
        row_ids=row_ids,
        y_true=np.vstack(true_rows),
        y_pred=np.vstack(pred_rows),
        raw_outputs=raw_outputs,
        latencies=latencies,
    )


def run_training_epoch(model, data_loader: DataLoader, optimizer, grad_accum_steps: int) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(data_loader, start=1):
        row_ids = batch.pop("row_id")
        del row_ids
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        batch_size = batch["input_ids"].size(0)
        total_loss += float(outputs.loss.detach().cpu()) * batch_size
        total_examples += batch_size

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if total_examples == 0:
        return math.nan

    if len(data_loader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / total_examples


def save_metrics_bundle(
    output_dir: Path,
    metrics_payload: dict,
    predictions: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    if history_df is not None:
        history_df.to_csv(output_dir / "training_history.csv", index=False)


def flatten_metrics_for_comparison(
    run_name: str,
    metrics_payload: dict,
) -> list[dict[str, object]]:
    test_metrics = metrics_payload["test_metrics"]
    sentiment_metrics = metrics_payload["test_sentiment_metrics"]
    rows = [
        {"run": run_name, "metric": "Emotion micro precision", "value": test_metrics["micro"]["precision"]},
        {"run": run_name, "metric": "Emotion micro recall", "value": test_metrics["micro"]["recall"]},
        {"run": run_name, "metric": "Emotion micro F1", "value": test_metrics["micro"]["f1"]},
        {"run": run_name, "metric": "Emotion macro precision", "value": test_metrics["macro"]["precision"]},
        {"run": run_name, "metric": "Emotion macro recall", "value": test_metrics["macro"]["recall"]},
        {"run": run_name, "metric": "Emotion macro F1", "value": test_metrics["macro"]["f1"]},
        {"run": run_name, "metric": "Emotion samples F1", "value": test_metrics["samples"]["f1"]},
        {"run": run_name, "metric": "Subset accuracy", "value": test_metrics["subset_accuracy"]},
        {"run": run_name, "metric": "Hamming loss", "value": test_metrics["hamming_loss"]},
        {
            "run": run_name,
            "metric": "Avg true labels per entry",
            "value": test_metrics["avg_true_labels_per_entry"],
        },
        {
            "run": run_name,
            "metric": "Avg predicted labels per entry",
            "value": test_metrics["avg_predicted_labels_per_entry"],
        },
        {"run": run_name, "metric": "Sentiment accuracy", "value": sentiment_metrics["accuracy"]},
        {"run": run_name, "metric": "Sentiment macro precision", "value": sentiment_metrics["macro"]["precision"]},
        {"run": run_name, "metric": "Sentiment macro recall", "value": sentiment_metrics["macro"]["recall"]},
        {"run": run_name, "metric": "Sentiment macro F1", "value": sentiment_metrics["macro"]["f1"]},
    ]
    return rows


def build_run_summary(title: str, metrics: dict, average_latency: float) -> str:
    test_metrics = metrics["test_metrics"]
    test_sentiment = metrics["test_sentiment_metrics"]
    return (
        f"{title}\n"
        f"{'=' * len(title)}\n"
        f"Model: {metrics['model_name']}\n"
        f"Dataset: {metrics['dataset_path']}\n"
        f"Split sizes: {metrics['split_sizes']}\n"
        f"Device: {metrics['device']}\n"
        f"Average latency (s): {average_latency:.2f}\n\n"
        "Emotion metrics\n"
        "---------------\n"
        f"Micro F1: {test_metrics['micro']['f1']:.4f}\n"
        f"Macro F1: {test_metrics['macro']['f1']:.4f}\n"
        f"Subset accuracy: {test_metrics['subset_accuracy']:.4f}\n"
        f"Hamming loss: {test_metrics['hamming_loss']:.4f}\n\n"
        "Sentiment metrics\n"
        "-----------------\n"
        f"Accuracy: {test_sentiment['accuracy']:.4f}\n"
        f"Macro F1: {test_sentiment['macro']['f1']:.4f}\n"
    )


def plot_comparison_figure(comparison_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    emotion_rows = comparison_df[
        comparison_df["metric"].isin(
            ["Emotion micro F1", "Emotion macro F1", "Subset accuracy", "Hamming loss"]
        )
    ]
    sentiment_rows = comparison_df[
        comparison_df["metric"].isin(["Sentiment accuracy", "Sentiment macro F1"])
    ]

    sns.barplot(data=emotion_rows, x="metric", y="value", hue="run", ax=axes[0])
    axes[0].set_title("Emotion metrics")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=18)

    sns.barplot(data=sentiment_rows, x="metric", y="value", hue="run", ax=axes[1])
    axes[1].set_title("Sentiment metrics")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=18)

    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "fig_compare_runs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_figure(comparison_df: pd.DataFrame) -> None:
    selected = comparison_df[
        comparison_df["metric"].isin(
            [
                "Emotion micro precision",
                "Emotion micro recall",
                "Emotion macro precision",
                "Emotion macro recall",
                "Sentiment macro precision",
                "Sentiment macro recall",
            ]
        )
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=selected, x="metric", y="value", hue="run", ax=ax)
    ax.set_title("Precision and recall comparison across runs")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "fig_compare_precision_recall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latency_figure(zero_shot_latency: list[float], finetuned_latency: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "latency_seconds": zero_shot_latency + finetuned_latency,
            "run": ["Zero-shot Gemma 2B"] * len(zero_shot_latency)
            + ["Fine-tuned Gemma 2B"] * len(finetuned_latency),
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.boxplot(data=frame, x="run", y="latency_seconds", ax=ax)
    ax.set_title("Gemma 2B latency before and after fine-tuning")
    ax.set_xlabel("")
    ax.set_ylabel("Seconds per example")
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "fig_compare_latency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)

    dataset_path = resolve_dataset_path(prefer_cleaned=True)
    df, text_col, emotion_cols, _ = load_analysis_data(prefer_cleaned=True)
    df = df.reset_index(drop=True)
    label_names = emotion_label_names(emotion_cols)
    split_map = split_indices(df[emotion_cols].to_numpy(dtype=float))

    train_df = df.iloc[split_map["train"]].reset_index(drop=True)
    val_df = df.iloc[split_map["val"]].reset_index(drop=True)
    test_df = df.iloc[split_map["test"]].reset_index(drop=True)
    train_df = maybe_limit_split(train_df)
    val_df = maybe_limit_split(val_df)
    test_df = maybe_limit_split(test_df)

    label_distribution = create_label_distribution(df, emotion_cols)
    label_distribution.to_csv(OUTPUT_DIR / "label_distribution.csv", index=False)

    snapshot_path = model_snapshot_path()
    device_config = get_device_config()

    zero_payload: dict[str, object] | None = None
    finetuned_payload: dict[str, object] | None = None
    zero_latencies: list[float] = []
    finetuned_latencies: list[float] = []

    # Zero-shot test evaluation.
    if not SKIP_ZERO_SHOT:
        zero_tokenizer, zero_model = load_model_for_generation(snapshot_path, None, device_config)
        zero_result = generate_predictions(
            model=zero_model,
            tokenizer=zero_tokenizer,
            df=test_df,
            text_col=text_col,
            emotion_cols=emotion_cols,
            label_names=label_names,
        )
        zero_metrics = compute_multilabel_metrics(
            zero_result.y_true,
            zero_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        zero_sentiment_metrics = compute_sentiment_metrics(
            zero_result.y_true,
            zero_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        zero_predictions = build_prediction_rows(
            df=test_df,
            row_ids=zero_result.row_ids,
            true_labels=zero_result.y_true,
            predicted_labels=zero_result.y_pred,
            raw_outputs=zero_result.raw_outputs,
            latencies=zero_result.latencies,
            text_col=text_col,
            label_names=label_names,
        )
        zero_payload = {
            "dataset_path": str(dataset_path),
            "model_name": MODEL_ID,
            "evaluation_mode": "zero_shot_generation",
            "device": device_config["label"],
            "split_sizes": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "label_names": label_names,
            "max_rows_per_split": MAX_ROWS_PER_SPLIT,
            "test_metrics": zero_metrics,
            "test_sentiment_metrics": zero_sentiment_metrics,
            "average_latency_seconds": float(np.mean(zero_result.latencies)),
        }
        zero_latencies = zero_result.latencies
        save_metrics_bundle(ZERO_SHOT_DIR, zero_payload, zero_predictions)
        (ZERO_SHOT_DIR / "run_summary.txt").write_text(
            build_run_summary(
                "Gemma 2B zero-shot emotion evaluation",
                zero_payload,
                float(np.mean(zero_result.latencies)),
            ),
            encoding="utf-8",
        )
        del zero_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Fine-tuning.
    if not SKIP_FINETUNE:
        train_tokenizer, train_model = load_base_model_for_training(snapshot_path, device_config)
        train_dataset = GemmaSFTDataset(
            df=train_df,
            text_col=text_col,
            emotion_cols=emotion_cols,
            tokenizer=train_tokenizer,
            label_names=label_names,
            max_length=MAX_LENGTH,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        optimizer = AdamW(
            [param for param in train_model.parameters() if param.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        history_rows = []
        for epoch in range(1, EPOCHS + 1):
            train_loss = run_training_epoch(train_model, train_loader, optimizer, TRAIN_GRAD_ACCUM_STEPS)
            history_rows.append({"epoch": epoch, "train_loss": train_loss})
            print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f}")

        train_model.save_pretrained(MODEL_DIR)
        train_tokenizer.save_pretrained(MODEL_DIR)
        history_df = pd.DataFrame(history_rows)

        del train_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_tokenizer, eval_model = load_model_for_generation(snapshot_path, MODEL_DIR, device_config)
        val_result = generate_predictions(
            model=eval_model,
            tokenizer=eval_tokenizer,
            df=val_df,
            text_col=text_col,
            emotion_cols=emotion_cols,
            label_names=label_names,
        )
        test_result = generate_predictions(
            model=eval_model,
            tokenizer=eval_tokenizer,
            df=test_df,
            text_col=text_col,
            emotion_cols=emotion_cols,
            label_names=label_names,
        )
        val_metrics = compute_multilabel_metrics(
            val_result.y_true,
            val_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        val_sentiment_metrics = compute_sentiment_metrics(
            val_result.y_true,
            val_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        test_metrics = compute_multilabel_metrics(
            test_result.y_true,
            test_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        test_sentiment_metrics = compute_sentiment_metrics(
            test_result.y_true,
            test_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        test_predictions = build_prediction_rows(
            df=test_df,
            row_ids=test_result.row_ids,
            true_labels=test_result.y_true,
            predicted_labels=test_result.y_pred,
            raw_outputs=test_result.raw_outputs,
            latencies=test_result.latencies,
            text_col=text_col,
            label_names=label_names,
        )
        finetuned_payload = {
            "dataset_path": str(dataset_path),
            "model_name": MODEL_ID,
            "evaluation_mode": "lora_finetuned_generation",
            "device": device_config["label"],
            "split_sizes": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "label_names": label_names,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": TRAIN_GRAD_ACCUM_STEPS,
            "max_rows_per_split": MAX_ROWS_PER_SPLIT,
            "validation_metrics": val_metrics,
            "validation_sentiment_metrics": val_sentiment_metrics,
            "test_metrics": test_metrics,
            "test_sentiment_metrics": test_sentiment_metrics,
            "average_latency_seconds": float(np.mean(test_result.latencies)),
        }
        finetuned_latencies = test_result.latencies
        save_metrics_bundle(FINETUNED_DIR, finetuned_payload, test_predictions, history_df=history_df)
        (FINETUNED_DIR / "run_summary.txt").write_text(
            build_run_summary(
                "Gemma 2B fine-tuned emotion evaluation",
                finetuned_payload,
                float(np.mean(test_result.latencies)),
            ),
            encoding="utf-8",
        )

    # Comparisons.
    bert_metrics = load_json(BERT_BASELINE_METRICS_PATH)
    comparison_rows = flatten_metrics_for_comparison("BERT baseline", bert_metrics)
    if zero_payload is not None:
        comparison_rows.extend(flatten_metrics_for_comparison("Gemma 2B zero-shot", zero_payload))
    if finetuned_payload is not None:
        comparison_rows.extend(flatten_metrics_for_comparison("Gemma 2B fine-tuned", finetuned_payload))
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(COMPARISON_DIR / "metrics_comparison.csv", index=False)
    plot_comparison_figure(comparison_df)
    plot_precision_recall_figure(comparison_df)
    if zero_latencies and finetuned_latencies:
        plot_latency_figure(zero_latencies, finetuned_latencies)

    summary = {
        "zero_shot_test_metrics": None if zero_payload is None else zero_payload["test_metrics"],
        "zero_shot_test_sentiment_metrics": None
        if zero_payload is None
        else zero_payload["test_sentiment_metrics"],
        "finetuned_test_metrics": None if finetuned_payload is None else finetuned_payload["test_metrics"],
        "finetuned_test_sentiment_metrics": None
        if finetuned_payload is None
        else finetuned_payload["test_sentiment_metrics"],
        "bert_baseline_test_metrics": bert_metrics["test_metrics"],
        "bert_baseline_test_sentiment_metrics": bert_metrics["test_sentiment_metrics"],
    }
    with open(COMPARISON_DIR / "comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved zero-shot outputs to: {ZERO_SHOT_DIR}")
    print(f"Saved fine-tuned outputs to: {FINETUNED_DIR}")
    print(f"Saved comparisons to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
