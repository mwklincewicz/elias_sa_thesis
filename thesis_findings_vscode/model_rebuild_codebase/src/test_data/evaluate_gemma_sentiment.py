from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name, safe_name
from test_data.common import GEMMA_SENTIMENT_DIR, STORIES_CSV_PATH, load_stories_data
from train_bert import derive_sentiment, format_active_labels


DEFAULT_MODEL_IDS = [
    "google/gemma-3n-E2B-it",
    "google/gemma-3n-E4B-it",
]
MODEL_IDS = [
    item.strip()
    for item in os.getenv("GEMMA_MODEL_IDS", ",".join(DEFAULT_MODEL_IDS)).split(",")
    if item.strip()
]
LOCAL_FILES_ONLY = os.getenv("GEMMA_LOCAL_FILES_ONLY", "0") == "1"
MAX_NEW_TOKENS = int(os.getenv("GEMMA_MAX_NEW_TOKENS", "6"))
GPU_MAX_MEMORY_GB = float(os.getenv("GEMMA_GPU_MAX_MEMORY_GB", "3.0"))
PROMPT_VERSION = "stories-sa-v1"


def output_dir_for_model(model_id: str) -> Path:
    return GEMMA_SENTIMENT_DIR / safe_name(model_id.replace("/", "_"))


def select_device_config() -> dict[str, object]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {
            "device_label": f"cuda:{torch.cuda.current_device()}",
            "torch_dtype": dtype,
            "device_map": "auto",
            "max_memory": {
                0: f"{GPU_MAX_MEMORY_GB:.1f}GiB",
                "cpu": "48GiB",
            },
        }

    return {
        "device_label": "cpu",
        "torch_dtype": torch.float32,
        "device_map": "cpu",
        "max_memory": None,
    }


def gpu_visible_to_system() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def build_prompt(text: str) -> str:
    return (
        "You are labeling the sentiment of a short personal reflection.\n"
        "Choose exactly one label from: positive, negative, neutral.\n"
        "Return only one lowercase word and nothing else.\n\n"
        f"Reflection:\n{text.strip()}\n"
    )


def tokenize_prompt(tokenizer, prompt: str):
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt")


def parse_sentiment(raw_text: str) -> tuple[str, str]:
    cleaned = raw_text.strip().lower()
    for label in ("positive", "negative", "neutral"):
        if label in cleaned:
            return label, "parsed"

    first_token = cleaned.replace(".", " ").replace(",", " ").split()
    if first_token:
        token = first_token[0]
        if token in {"positive", "negative", "neutral"}:
            return token, "parsed"

    return "neutral", "fallback_neutral"


def compute_sentiment_metrics_from_labels(
    true_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, object]:
    ordered_labels = ["negative", "neutral", "positive"]
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=ordered_labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
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


def build_summary_text(
    model_id: str,
    output_dir: Path,
    stories_path: Path,
    device_label: str,
    metrics: dict[str, object],
    parse_failures: int,
    average_latency_seconds: float,
    gpu_detected_by_system: bool,
) -> str:
    sentiment_metrics = metrics["test_sentiment_metrics"]
    return (
        "Gemma Stories sentiment evaluation\n"
        "=================================\n"
        f"Stories test data: {stories_path}\n"
        f"Model: {model_id}\n"
        f"Output folder: {output_dir}\n"
        f"Local files only: {LOCAL_FILES_ONLY}\n"
        f"Evaluation mode: zero-shot instruction following\n"
        f"Device: {device_label}\n"
        f"GPU detected by system tools: {gpu_detected_by_system}\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Parse fallbacks: {parse_failures}\n"
        f"Average latency per example (s): {average_latency_seconds:.2f}\n\n"
        "Stories sentiment metrics\n"
        "-------------------------\n"
        f"Accuracy: {sentiment_metrics['accuracy']:.4f}\n"
        f"Macro precision: {sentiment_metrics['macro']['precision']:.4f}\n"
        f"Macro recall: {sentiment_metrics['macro']['recall']:.4f}\n"
        f"Macro F1: {sentiment_metrics['macro']['f1']:.4f}\n"
    )


def load_model_bundle(model_id: str, device_config: dict[str, object]):
    try:
        snapshot_path = snapshot_download(
            model_id,
            local_files_only=LOCAL_FILES_ONLY,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=LOCAL_FILES_ONLY,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            local_files_only=LOCAL_FILES_ONLY,
            dtype=device_config["torch_dtype"],
            device_map=device_config["device_map"],
            max_memory=device_config["max_memory"],
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
    except OSError as exc:
        lowered = str(exc).lower()
        if "gated repo" in lowered or "401" in lowered or "access to model" in lowered:
            raise RuntimeError(
                "Gemma download is blocked because the official Hugging Face repo is gated. "
                "Accept the Gemma license for the model on Hugging Face and log in locally "
                "with `huggingface-cli login`, or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            ) from exc
        raise

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def evaluate_model(
    model_id: str,
    stories_df: pd.DataFrame,
    text_col: str,
    emotion_cols: list[str],
) -> dict[str, object]:
    label_names = [display_name(col) for col in emotion_cols]
    device_config = select_device_config()
    system_gpu_visible = gpu_visible_to_system()
    if device_config["device_label"] == "cpu" and system_gpu_visible:
        print(
            "Warning: NVIDIA GPU is visible to the system, but PyTorch cannot use CUDA in this "
            "environment. The script will run on CPU unless a CUDA-enabled PyTorch build is installed."
        )
    tokenizer, model = load_model_bundle(model_id, device_config)

    true_emotions = stories_df[emotion_cols].to_numpy(dtype=int)
    true_sentiments = []
    predictions = []
    latencies = []
    parse_failures = 0

    for row_id, (_, row) in enumerate(stories_df.iterrows()):
        text = str(row[text_col]).strip()
        true_sentiment, true_score = derive_sentiment(true_emotions[row_id], label_names)
        true_sentiments.append(true_sentiment)

        prompt = build_prompt(text)
        model_inputs = tokenize_prompt(tokenizer, prompt)
        if hasattr(model_inputs, "to") and device_config["device_label"] != "cpu":
            model_inputs = model_inputs.to(model.device)
        elif isinstance(model_inputs, torch.Tensor) and device_config["device_label"] != "cpu":
            model_inputs = model_inputs.to(model.device)

        if isinstance(model_inputs, torch.Tensor):
            generation_inputs = {"input_ids": model_inputs}
            prompt_token_count = model_inputs.shape[-1]
        else:
            generation_inputs = dict(model_inputs)
            prompt_token_count = generation_inputs["input_ids"].shape[-1]

        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **generation_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        generated_tokens = generated[0][prompt_token_count:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predicted_sentiment, parse_status = parse_sentiment(raw_output)
        if parse_status != "parsed":
            parse_failures += 1

        predictions.append(
            {
                "row_id": row_id,
                "text": text,
                "true_emotions": format_active_labels(true_emotions[row_id], label_names),
                "true_sentiment": true_sentiment,
                "predicted_sentiment": predicted_sentiment,
                "true_sentiment_score": true_score,
                "raw_model_output": raw_output,
                "parse_status": parse_status,
                "latency_seconds": elapsed,
            }
        )

        print(
            f"[{model_id}] {row_id + 1}/{len(stories_df)} | "
            f"true={true_sentiment} | pred={predicted_sentiment} | "
            f"latency={elapsed:.2f}s"
        )

    predicted_sentiments = [row["predicted_sentiment"] for row in predictions]
    sentiment_metrics = compute_sentiment_metrics_from_labels(true_sentiments, predicted_sentiments)

    return {
        "stories_path": str(STORIES_CSV_PATH),
        "model_name": model_id,
        "task": "sentiment_analysis",
        "evaluation_mode": "zero_shot_instruction",
        "local_files_only": LOCAL_FILES_ONLY,
        "device": device_config["device_label"],
        "gpu_detected_by_system": system_gpu_visible,
        "prompt_version": PROMPT_VERSION,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        },
        "split_sizes": {"test": len(stories_df)},
        "parse_fallbacks": parse_failures,
        "average_latency_seconds": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "test_sentiment_metrics": sentiment_metrics,
        "predictions": predictions,
    }


def save_model_outputs(result: dict[str, object]) -> None:
    model_id = str(result["model_name"])
    output_dir = output_dir_for_model(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(result["predictions"])
    predictions_path = output_dir / "stories_test_predictions.csv"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "run_summary.txt"

    metrics_payload = {key: value for key, value in result.items() if key != "predictions"}
    predictions_df.to_csv(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    summary_text = build_summary_text(
        model_id=model_id,
        output_dir=output_dir,
        stories_path=Path(str(result["stories_path"])),
        device_label=str(result["device"]),
        metrics=metrics_payload,
        parse_failures=int(result["parse_fallbacks"]),
        average_latency_seconds=float(result["average_latency_seconds"]),
        gpu_detected_by_system=bool(result["gpu_detected_by_system"]),
    )
    summary_path.write_text(summary_text, encoding="utf-8")


def main() -> None:
    GEMMA_SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    stories_df, text_col, emotion_cols, _ = load_stories_data()
    stories_df = stories_df.reset_index(drop=True)

    comparison_rows = []
    for model_id in MODEL_IDS:
        result = evaluate_model(
            model_id=model_id,
            stories_df=stories_df,
            text_col=text_col,
            emotion_cols=emotion_cols,
        )
        save_model_outputs(result)
        metrics = result["test_sentiment_metrics"]
        comparison_rows.append(
            {
                "model_name": model_id,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro"]["precision"],
                "macro_recall": metrics["macro"]["recall"],
                "macro_f1": metrics["macro"]["f1"],
                "parse_fallbacks": result["parse_fallbacks"],
                "average_latency_seconds": result["average_latency_seconds"],
                "device": result["device"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(GEMMA_SENTIMENT_DIR / "comparison_summary.csv", index=False)
    print(f"Saved comparison summary to: {GEMMA_SENTIMENT_DIR / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
