from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dataclasses import dataclass

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import display_name
from gemma_2b_emotion_pipeline import (
    EVAL_BATCH_SIZE,
    FINETUNED_DIR,
    MAX_NEW_TOKENS,
    MODEL_ID,
    build_emotion_prompt,
    build_generation_batch,
    build_prediction_rows,
    generate_predictions,
    get_device_config,
    labels_to_binary,
    load_model_for_generation,
    model_snapshot_path,
    normalize_generated_labels,
)
from test_data.common import STORIES_CSV_PATH, TEST_DATA_OUTPUT_DIR, load_stories_data
from train_bert import (
    compute_multilabel_metrics,
    compute_sentiment_metrics,
    derive_sentiment,
    format_active_labels,
    set_seed,
)

OUTPUT_DIR = TEST_DATA_OUTPUT_DIR / "gemma_emotion_stories"
ZERO_SHOT_DIR = OUTPUT_DIR / "zero_shot"
FINETUNED_STORIES_DIR = OUTPUT_DIR / "finetuned"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
FIGURES_DIR = OUTPUT_DIR / "figures"
ADAPTER_DIR = FINETUNED_DIR / "adapter"
SEED = 42
RUN_SELECTION = {
    item.strip().lower()
    for item in os.getenv("GEMMA_STORIES_RUNS", "finetuned,zero_shot").split(",")
    if item.strip()
}

for path in [OUTPUT_DIR, ZERO_SHOT_DIR, FINETUNED_STORIES_DIR, COMPARISON_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


@dataclass
class IncrementalResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    predictions: pd.DataFrame
    latencies: list[float]


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_label_string(value: str, label_names: list[str]) -> np.ndarray:
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "none":
        active = set()
    else:
        active = {part.strip() for part in cleaned.split(",") if part.strip()}
    return np.array([1 if label in active else 0 for label in label_names], dtype=int)


def append_prediction_row(path: Path, row: dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def generate_predictions_incremental(
    model,
    tokenizer,
    df: pd.DataFrame,
    text_col: str,
    emotion_cols: list[str],
    label_names: list[str],
    output_dir: Path,
    run_label: str,
) -> IncrementalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "stories_predictions_partial.csv"

    completed: set[int] = set()
    if partial_path.exists():
        existing = pd.read_csv(partial_path)
        if "row_id" in existing.columns:
            completed = {int(row_id) for row_id in existing["row_id"].tolist()}
            print(f"Resuming {run_label}: {len(completed)} completed rows found.", flush=True)

    model.eval()
    df_local = df.reset_index(drop=True)
    for row_id, row in df_local.iterrows():
        if int(row_id) in completed:
            continue

        prompt = build_emotion_prompt(str(row[text_col]).strip(), label_names)
        generation_inputs = build_generation_batch(tokenizer, [prompt])
        generation_inputs = {key: value.to(model.device) for key, value in generation_inputs.items()}
        prompt_tokens = int(generation_inputs["attention_mask"].sum(dim=1).tolist()[0])

        started = pd.Timestamp.utcnow().timestamp()
        timer = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_timer = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if timer is not None and end_timer is not None:
            timer.record()
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
        if timer is not None and end_timer is not None:
            end_timer.record()
            torch.cuda.synchronize()
            latency = float(timer.elapsed_time(end_timer) / 1000.0)
        else:
            latency = float(pd.Timestamp.utcnow().timestamp() - started)

        generated_tokens = generated[0][prompt_tokens:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        parsed_labels = normalize_generated_labels(raw_output, label_names)
        y_true = row[emotion_cols].to_numpy(dtype=int)
        y_pred = labels_to_binary(parsed_labels, label_names)
        true_sentiment, true_score = derive_sentiment(y_true, label_names)
        pred_sentiment, pred_score = derive_sentiment(y_pred, label_names)

        prediction_row = {
            "row_id": int(row_id),
            "text": row[text_col],
            "true_emotions": format_active_labels(y_true, label_names),
            "predicted_emotions": format_active_labels(y_pred, label_names),
            "true_label_count": int(y_true.sum()),
            "predicted_label_count": int(y_pred.sum()),
            "prediction_surplus": int(y_pred.sum() - y_true.sum()),
            "true_sentiment": true_sentiment,
            "predicted_sentiment": pred_sentiment,
            "true_sentiment_score": true_score,
            "predicted_sentiment_score": pred_score,
            "raw_model_output": raw_output,
            "latency_seconds": latency,
        }
        append_prediction_row(partial_path, prediction_row)
        print(
            f"{run_label} {row_id + 1}/{len(df_local)} | "
            f"true={prediction_row['true_emotions']} | "
            f"pred={prediction_row['predicted_emotions']} | "
            f"latency={latency:.2f}s",
            flush=True,
        )

    predictions = pd.read_csv(partial_path).sort_values("row_id").reset_index(drop=True)
    y_true_matrix = np.vstack(
        [parse_label_string(value, label_names) for value in predictions["true_emotions"]]
    )
    y_pred_matrix = np.vstack(
        [parse_label_string(value, label_names) for value in predictions["predicted_emotions"]]
    )
    predictions.to_csv(output_dir / "stories_predictions.csv", index=False)
    return IncrementalResult(
        y_true=y_true_matrix,
        y_pred=y_pred_matrix,
        predictions=predictions,
        latencies=predictions["latency_seconds"].astype(float).tolist(),
    )


def save_run_bundle(
    output_dir: Path,
    run_name: str,
    result,
    stories_df: pd.DataFrame,
    text_col: str,
    label_names: list[str],
    model_source: str,
    device_label: str,
    emotion_metrics: dict,
    sentiment_metrics: dict,
) -> dict:
    predictions = build_prediction_rows(
        df=stories_df,
        row_ids=result.row_ids,
        true_labels=result.y_true,
        predicted_labels=result.y_pred,
        raw_outputs=result.raw_outputs,
        latencies=result.latencies,
        text_col=text_col,
        label_names=label_names,
    )
    predictions["true_label_count"] = result.y_true.sum(axis=1)
    predictions["predicted_label_count"] = result.y_pred.sum(axis=1)
    predictions["prediction_surplus"] = (
        predictions["predicted_label_count"] - predictions["true_label_count"]
    )

    payload = {
        "stories_test_dataset_path": str(STORIES_CSV_PATH),
        "model_name": MODEL_ID,
        "evaluation_mode": run_name,
        "model_source": model_source,
        "device": device_label,
        "split_sizes": {"test": len(stories_df)},
        "label_names": label_names,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "test_metrics": emotion_metrics,
        "test_sentiment_metrics": sentiment_metrics,
        "average_latency_seconds": float(np.mean(result.latencies)),
        "total_runtime_seconds": float(np.sum(result.latencies)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "stories_predictions.csv", index=False)
    pd.DataFrame(emotion_metrics["label_metrics"]).to_csv(
        output_dir / "per_emotion_metrics.csv",
        index=False,
    )
    pd.DataFrame(sentiment_metrics["label_metrics"]).to_csv(
        output_dir / "per_sentiment_metrics.csv",
        index=False,
    )
    write_json(output_dir / "metrics.json", payload)
    (output_dir / "run_summary.txt").write_text(
        build_summary_text(payload),
        encoding="utf-8",
    )
    return payload


def save_incremental_run_bundle(
    output_dir: Path,
    run_name: str,
    result: IncrementalResult,
    label_names: list[str],
    model_source: str,
    device_label: str,
    emotion_metrics: dict,
    sentiment_metrics: dict,
) -> dict:
    result.predictions.to_csv(output_dir / "stories_predictions.csv", index=False)
    pd.DataFrame(emotion_metrics["label_metrics"]).to_csv(
        output_dir / "per_emotion_metrics.csv",
        index=False,
    )
    pd.DataFrame(sentiment_metrics["label_metrics"]).to_csv(
        output_dir / "per_sentiment_metrics.csv",
        index=False,
    )
    payload = {
        "stories_test_dataset_path": str(STORIES_CSV_PATH),
        "model_name": MODEL_ID,
        "evaluation_mode": run_name,
        "model_source": model_source,
        "device": device_label,
        "split_sizes": {"test": len(result.predictions)},
        "label_names": label_names,
        "eval_batch_size": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "test_metrics": emotion_metrics,
        "test_sentiment_metrics": sentiment_metrics,
        "average_latency_seconds": float(np.mean(result.latencies)),
        "total_runtime_seconds": float(np.sum(result.latencies)),
    }
    write_json(output_dir / "metrics.json", payload)
    (output_dir / "run_summary.txt").write_text(build_summary_text(payload), encoding="utf-8")
    return payload


def build_summary_text(payload: dict) -> str:
    emotion = payload["test_metrics"]
    sentiment = payload["test_sentiment_metrics"]
    return (
        f"Gemma Stories 18-label emotion evaluation: {payload['evaluation_mode']}\n"
        "=======================================================\n"
        f"Stories evaluation data: {payload['stories_test_dataset_path']}\n"
        f"Model: {payload['model_name']}\n"
        f"Source: {payload['model_source']}\n"
        f"Device: {payload['device']}\n"
        f"Rows: {payload['split_sizes']['test']}\n"
        f"Average latency: {payload['average_latency_seconds']:.2f}s/example\n\n"
        "Emotion metrics\n"
        "---------------\n"
        f"Micro F1: {emotion['micro']['f1']:.4f}\n"
        f"Macro F1: {emotion['macro']['f1']:.4f}\n"
        f"Subset accuracy: {emotion['subset_accuracy']:.4f}\n"
        f"Hamming loss: {emotion['hamming_loss']:.4f}\n"
        f"Average true labels: {emotion['avg_true_labels_per_entry']:.4f}\n"
        f"Average predicted labels: {emotion['avg_predicted_labels_per_entry']:.4f}\n\n"
        "Derived sentiment metrics\n"
        "-------------------------\n"
        f"Accuracy: {sentiment['accuracy']:.4f}\n"
        f"Macro F1: {sentiment['macro']['f1']:.4f}\n"
    )


def flatten_summary(run_label: str, payload: dict) -> dict[str, object]:
    emotion = payload["test_metrics"]
    sentiment = payload["test_sentiment_metrics"]
    return {
        "run": run_label,
        "emotion_micro_f1": emotion["micro"]["f1"],
        "emotion_macro_f1": emotion["macro"]["f1"],
        "emotion_subset_accuracy": emotion["subset_accuracy"],
        "emotion_hamming_loss": emotion["hamming_loss"],
        "emotion_micro_precision": emotion["micro"]["precision"],
        "emotion_micro_recall": emotion["micro"]["recall"],
        "sentiment_accuracy": sentiment["accuracy"],
        "sentiment_macro_f1": sentiment["macro"]["f1"],
        "sentiment_macro_precision": sentiment["macro"]["precision"],
        "sentiment_macro_recall": sentiment["macro"]["recall"],
        "avg_true_labels": emotion["avg_true_labels_per_entry"],
        "avg_predicted_labels": emotion["avg_predicted_labels_per_entry"],
        "average_latency_seconds": payload["average_latency_seconds"],
    }


def plot_metric_comparison(summary: pd.DataFrame) -> Path:
    long = summary.melt(
        id_vars=["run"],
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
            "sentiment_accuracy": "Derived sentiment accuracy",
            "sentiment_macro_f1": "Derived sentiment macro-F1",
        }
    )
    path = FIGURES_DIR / "fig_gemma_stories_emotion_sentiment_metrics.png"
    plt.figure(figsize=(10.5, 5.8))
    ax = sns.barplot(data=long, x="metric", y="score", hue="run", palette="deep")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Gemma 18-label emotion evaluation on Stories")
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
                fontsize=9,
            )
    ax.legend(title="")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_per_emotion_f1() -> Path:
    frames = []
    for run_label, run_dir in [
        ("Gemma zero-shot", ZERO_SHOT_DIR),
        ("Gemma fine-tuned", FINETUNED_STORIES_DIR),
    ]:
        frame = pd.read_csv(run_dir / "per_emotion_metrics.csv")
        frame["run"] = run_label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    supported = combined.groupby("label")["support"].max().loc[lambda series: series > 0].index
    combined = combined[combined["label"].isin(supported)]
    order = (
        combined[combined["run"] == "Gemma fine-tuned"]
        .sort_values("f1", ascending=False)["label"]
        .tolist()
    )
    path = FIGURES_DIR / "fig_gemma_stories_per_emotion_f1.png"
    plt.figure(figsize=(10.8, 7.0))
    ax = sns.barplot(data=combined, y="label", x="f1", hue="run", order=order, palette="deep")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_title("Gemma Stories per-emotion F1")
    ax.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_cardinality() -> Path:
    rows = []
    true_counts_written = False
    for run_label, run_dir in [
        ("Gemma zero-shot", ZERO_SHOT_DIR),
        ("Gemma fine-tuned", FINETUNED_STORIES_DIR),
    ]:
        predictions = pd.read_csv(run_dir / "stories_predictions.csv")
        if not true_counts_written:
            for count, rows_count in predictions["true_label_count"].value_counts().items():
                rows.append(
                    {"label_count": int(count), "series": "True label count", "rows": int(rows_count)}
                )
            true_counts_written = True
        for count, rows_count in predictions["predicted_label_count"].value_counts().items():
            rows.append({"label_count": int(count), "series": run_label, "rows": int(rows_count)})

    frame = pd.DataFrame(rows)
    path = FIGURES_DIR / "fig_gemma_stories_cardinality.png"
    plt.figure(figsize=(10.0, 5.8))
    ax = sns.lineplot(
        data=frame[frame["series"] == "True label count"],
        x="label_count",
        y="rows",
        marker="o",
        color="#222222",
        linewidth=2,
        label="True label count",
    )
    sns.barplot(
        data=frame[frame["series"] != "True label count"],
        x="label_count",
        y="rows",
        hue="series",
        alpha=0.75,
        ax=ax,
    )
    ax.set_xlabel("Emotion labels per reflection")
    ax.set_ylabel("Rows")
    ax.set_title("Gemma Stories prediction cardinality")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def write_report(summary: pd.DataFrame, figure_paths: list[Path]) -> None:
    zero = summary[summary["run"] == "Gemma zero-shot"].iloc[0]
    tuned = summary[summary["run"] == "Gemma fine-tuned"].iloc[0]
    lines = [
        "# Gemma 18-Label Emotion Evaluation on Stories",
        "",
        "This run evaluates both Gemma variants on Stories using the full 18-label Lemotif emotion taxonomy. Sentiment metrics are derived from the predicted emotion labels, matching the BERT evaluation logic.",
        "",
        "## Summary",
        "",
        f"- Gemma zero-shot: emotion micro-F1 = {zero['emotion_micro_f1']:.3f}, emotion macro-F1 = {zero['emotion_macro_f1']:.3f}, derived sentiment accuracy = {zero['sentiment_accuracy']:.3f}, derived sentiment macro-F1 = {zero['sentiment_macro_f1']:.3f}.",
        f"- Gemma fine-tuned on Lemotif: emotion micro-F1 = {tuned['emotion_micro_f1']:.3f}, emotion macro-F1 = {tuned['emotion_macro_f1']:.3f}, derived sentiment accuracy = {tuned['sentiment_accuracy']:.3f}, derived sentiment macro-F1 = {tuned['sentiment_macro_f1']:.3f}.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in figure_paths)
    lines.append("")
    (OUTPUT_DIR / "gemma_stories_emotion_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    stories_df, text_col, emotion_cols, _ = load_stories_data()
    stories_df = stories_df.reset_index(drop=True)
    label_names = [display_name(col) for col in emotion_cols]

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned Gemma adapter not found: {ADAPTER_DIR}")

    snapshot_path = model_snapshot_path()
    device_config = get_device_config()

    print("Gemma Stories 18-label emotion evaluation")
    print(f"Rows: {len(stories_df)}")
    print(f"Labels: {len(label_names)}")
    print(f"Snapshot: {snapshot_path}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"Device: {device_config['label']}")

    tuned_payload = None
    zero_payload = None

    # Run the fine-tuned adapter first by default because it is much faster and
    # is the key comparison against BERT. The zero-shot run can resume row-by-row.
    if "finetuned" in RUN_SELECTION or "fine_tuned" in RUN_SELECTION:
        tuned_tokenizer, tuned_model = load_model_for_generation(snapshot_path, ADAPTER_DIR, device_config)
        tuned_result = generate_predictions_incremental(
            model=tuned_model,
            tokenizer=tuned_tokenizer,
            df=stories_df,
            text_col=text_col,
            emotion_cols=emotion_cols,
            label_names=label_names,
            output_dir=FINETUNED_STORIES_DIR,
            run_label="Gemma fine-tuned",
        )
        tuned_emotion = compute_multilabel_metrics(
            tuned_result.y_true,
            tuned_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        tuned_sentiment = compute_sentiment_metrics(
            tuned_result.y_true,
            tuned_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        tuned_payload = save_incremental_run_bundle(
            FINETUNED_STORIES_DIR,
            "lemotif_lora_finetuned_18_label_generation",
            tuned_result,
            label_names,
            model_source=str(ADAPTER_DIR),
            device_label=str(device_config["label"]),
            emotion_metrics=tuned_emotion,
            sentiment_metrics=tuned_sentiment,
        )
        del tuned_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "zero_shot" in RUN_SELECTION or "zeroshot" in RUN_SELECTION:
        zero_tokenizer, zero_model = load_model_for_generation(snapshot_path, None, device_config)
        zero_result = generate_predictions_incremental(
            model=zero_model,
            tokenizer=zero_tokenizer,
            df=stories_df,
            text_col=text_col,
            emotion_cols=emotion_cols,
            label_names=label_names,
            output_dir=ZERO_SHOT_DIR,
            run_label="Gemma zero-shot",
        )
        zero_emotion = compute_multilabel_metrics(
            zero_result.y_true,
            zero_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        zero_sentiment = compute_sentiment_metrics(
            zero_result.y_true,
            zero_result.y_pred.astype(float),
            0.5,
            label_names,
        )
        zero_payload = save_incremental_run_bundle(
            ZERO_SHOT_DIR,
            "zero_shot_18_label_generation",
            zero_result,
            label_names,
            model_source=snapshot_path,
            device_label=str(device_config["label"]),
            emotion_metrics=zero_emotion,
            sentiment_metrics=zero_sentiment,
        )

    summary_rows = []
    if zero_payload is None and (ZERO_SHOT_DIR / "metrics.json").exists():
        with open(ZERO_SHOT_DIR / "metrics.json", "r", encoding="utf-8") as handle:
            zero_payload = json.load(handle)
    if tuned_payload is None and (FINETUNED_STORIES_DIR / "metrics.json").exists():
        with open(FINETUNED_STORIES_DIR / "metrics.json", "r", encoding="utf-8") as handle:
            tuned_payload = json.load(handle)
    if zero_payload is not None:
        summary_rows.append(flatten_summary("Gemma zero-shot", zero_payload))
    if tuned_payload is not None:
        summary_rows.append(flatten_summary("Gemma fine-tuned", tuned_payload))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(COMPARISON_DIR / "gemma_stories_emotion_summary.csv", index=False)
    write_json(
        COMPARISON_DIR / "comparison_summary.json",
        {
            "zero_shot": zero_payload,
            "fine_tuned": tuned_payload,
        },
    )

    figure_paths = []
    if len(summary) >= 2:
        figure_paths = [
            plot_metric_comparison(summary),
            plot_per_emotion_f1(),
            plot_cardinality(),
        ]
        write_report(summary, figure_paths)

    print()
    print(summary.to_string(index=False))
    print(f"Saved outputs to: {OUTPUT_DIR}")
    for path in figure_paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
