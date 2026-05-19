from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_loader import display_name, load_analysis_data, resolve_dataset_path
from train_bert import (
    LOCAL_FILES_ONLY,
    MODEL_NAME,
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    MAX_POS_WEIGHT,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    LemotifEmotionDataset,
    build_predictions_frame,
    compute_multilabel_metrics,
    compute_sentiment_metrics,
    create_label_distribution,
    find_best_threshold,
    run_epoch,
    set_seed,
    split_indices,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "bert_emotion_model_optuna"

OPTUNA_TRIALS = int(os.getenv("LEMOTIF_OPTUNA_TRIALS", "10"))
OPTUNA_SEED = int(os.getenv("LEMOTIF_OPTUNA_SEED", str(SEED)))
OPTUNA_STUDY_NAME = os.getenv("LEMOTIF_OPTUNA_STUDY_NAME", "lemotif_bert_optuna")
OPTUNA_OUTPUT_DIR_ENV = os.getenv("LEMOTIF_OPTUNA_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
OBJECTIVE_METRIC = "validation_micro_f1"


def resolve_output_dir(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


OUTPUT_DIR = resolve_output_dir(OPTUNA_OUTPUT_DIR_ENV)
MODEL_DIR = OUTPUT_DIR / "model"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.txt"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
LABEL_DISTRIBUTION_PATH = OUTPUT_DIR / "label_distribution.csv"
THRESHOLD_SCAN_PATH = OUTPUT_DIR / "validation_threshold_scan.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
TRIALS_CSV_PATH = OUTPUT_DIR / "study_trials.csv"
TRIALS_JSON_PATH = OUTPUT_DIR / "study_trials.json"
STUDY_SUMMARY_PATH = OUTPUT_DIR / "study_summary.json"
PARAM_IMPORTANCE_PATH = OUTPUT_DIR / "parameter_importances.csv"

SEARCH_SPACE = {
    "learning_rate": {"type": "float_log", "low": 1e-5, "high": 8e-5},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.15},
    "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.25},
    "epochs": {"type": "int", "low": 3, "high": 6},
    "batch_size": {"type": "categorical", "choices": [4, 8, 12]},
    "max_length": {"type": "categorical", "choices": [128, 160, 192, 224, 256]},
    "max_pos_weight": {"type": "float", "low": 8.0, "high": 30.0},
}
DEFAULT_HYPERPARAMETERS = {
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "max_pos_weight": MAX_POS_WEIGHT,
}


def resolve_model_source() -> tuple[str, str]:
    explicit_path = Path(MODEL_NAME)
    if explicit_path.exists():
        return str(explicit_path.resolve()), "explicit local model path"

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{MODEL_NAME.replace('/', '--')}" / "snapshots"
    if cache_root.exists():
        snapshots = sorted(
            [path for path in cache_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0].resolve()), "local Hugging Face cache snapshot"

    return MODEL_NAME, "model id"


MODEL_SOURCE, MODEL_SOURCE_DESCRIPTION = resolve_model_source()
MODEL_SOURCE_IS_LOCAL = Path(MODEL_SOURCE).exists()


def build_pos_weight_with_cap(train_labels: np.ndarray, max_pos_weight: float) -> torch.Tensor:
    positive = train_labels.sum(axis=0)
    negative = len(train_labels) - positive
    raw_weights = negative / np.clip(positive, a_min=1.0, a_max=None)
    clipped_weights = np.clip(raw_weights, a_min=1.0, a_max=max_pos_weight)
    return torch.tensor(clipped_weights, dtype=torch.float32)


def suggest_hyperparameters(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.25),
        "epochs": trial.suggest_int("epochs", 3, 6),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 12]),
        "max_length": trial.suggest_categorical("max_length", [128, 160, 192, 224, 256]),
        "max_pos_weight": trial.suggest_float("max_pos_weight", 8.0, 30.0),
    }


def build_run_summary(
    dataset_path: str,
    label_names: list[str],
    split_sizes: dict[str, int],
    device: torch.device,
    best_trial_number: int,
    best_epoch: int,
    best_threshold: float,
    selected_hyperparameters: dict[str, float | int],
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
    test_sentiment_metrics: dict[str, object],
) -> str:
    param_lines = "\n".join(
        f"- {name}: {value}"
        for name, value in selected_hyperparameters.items()
    )
    return (
        "Optuna-tuned Lemotif BERT baseline\n"
        "=================================\n"
        f"Dataset used: {dataset_path}\n"
        f"Base model: {MODEL_NAME}\n"
        f"Initialization source: {MODEL_SOURCE}\n"
        f"Initialization source type: {MODEL_SOURCE_DESCRIPTION}\n"
        f"Objective metric: {OBJECTIVE_METRIC}\n"
        f"Study name: {OPTUNA_STUDY_NAME}\n"
        f"Trials requested: {OPTUNA_TRIALS}\n"
        f"Local files only: {LOCAL_FILES_ONLY}\n"
        f"Device: {device}\n"
        f"Emotion labels: {len(label_names)}\n"
        f"Split sizes: train={split_sizes['train']}, val={split_sizes['val']}, test={split_sizes['test']}\n"
        f"Best trial: {best_trial_number}\n"
        f"Best epoch: {best_epoch}\n"
        f"Validation-selected threshold: {best_threshold:.2f}\n\n"
        "Selected hyperparameters\n"
        "------------------------\n"
        f"{param_lines}\n\n"
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


def save_best_run_artifacts(
    *,
    dataset_path: str,
    label_names: list[str],
    split_sizes: dict[str, int],
    device: torch.device,
    trial_number: int,
    selected_hyperparameters: dict[str, float | int],
    best_epoch: int,
    best_threshold: float,
    best_threshold_scan: pd.DataFrame,
    history_df: pd.DataFrame,
    validation_metrics: dict[str, object],
    validation_sentiment_metrics: dict[str, object],
    test_metrics: dict[str, object],
    test_sentiment_metrics: dict[str, object],
    test_predictions: pd.DataFrame,
    model,
    tokenizer,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    history_df.to_csv(HISTORY_PATH, index=False)
    best_threshold_scan.to_csv(THRESHOLD_SCAN_PATH, index=False)
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metrics_payload = {
        "dataset_path": dataset_path,
        "model_name": MODEL_NAME,
        "model_source": MODEL_SOURCE,
        "model_source_description": MODEL_SOURCE_DESCRIPTION,
        "local_files_only": LOCAL_FILES_ONLY,
        "seed": OPTUNA_SEED,
        "study_name": OPTUNA_STUDY_NAME,
        "objective_metric": OBJECTIVE_METRIC,
        "trials_requested": OPTUNA_TRIALS,
        "best_trial_number": trial_number,
        "default_hyperparameters": DEFAULT_HYPERPARAMETERS,
        "selected_hyperparameters": selected_hyperparameters,
        "search_space": SEARCH_SPACE,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "label_names": label_names,
        "split_sizes": split_sizes,
        "validation_metrics": validation_metrics,
        "validation_sentiment_metrics": validation_sentiment_metrics,
        "test_metrics": test_metrics,
        "test_sentiment_metrics": test_sentiment_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    summary_text = build_run_summary(
        dataset_path=dataset_path,
        label_names=label_names,
        split_sizes=split_sizes,
        device=device,
        best_trial_number=trial_number,
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        selected_hyperparameters=selected_hyperparameters,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_sentiment_metrics=test_sentiment_metrics,
    )
    RUN_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(OPTUNA_SEED)

    df, text_col, emotion_cols, _ = load_analysis_data(prefer_cleaned=True)
    df = df.reset_index(drop=True)
    dataset_path = str(resolve_dataset_path(prefer_cleaned=True))

    label_names = [display_name(col) for col in emotion_cols]
    label_distribution = create_label_distribution(df, emotion_cols)
    label_distribution.to_csv(LABEL_DISTRIBUTION_PATH, index=False)

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[emotion_cols].to_numpy(dtype=np.float32)
    split_map = split_indices(labels)
    split_sizes = {name: int(len(indices)) for name, indices in split_map.items()}
    train_labels = labels[split_map["train"]]

    print("Running Optuna hyperparameter tuning for the Lemotif BERT baseline.")
    print(f"Dataset: {dataset_path}")
    print(f"Study name: {OPTUNA_STUDY_NAME}")
    print(f"Trials: {OPTUNA_TRIALS}")
    print(f"Objective: {OBJECTIVE_METRIC}")
    print(f"Initialization source: {MODEL_SOURCE} ({MODEL_SOURCE_DESCRIPTION})")
    print(
        "Split sizes: "
        f"train={split_sizes['train']} | val={split_sizes['val']} | test={split_sizes['test']}"
    )

    best_tracker = {"value": float("-inf"), "trial_number": -1}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_id_map = {idx: label for idx, label in enumerate(label_names)}
    reverse_label_id_map = {label: idx for idx, label in label_id_map.items()}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)
        set_seed(OPTUNA_SEED)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SOURCE,
            local_files_only=LOCAL_FILES_ONLY or MODEL_SOURCE_IS_LOCAL,
            use_fast=True,
        )

        datasets = {}
        for split_name, row_ids in split_map.items():
            datasets[split_name] = LemotifEmotionDataset(
                texts=[texts[idx] for idx in row_ids],
                labels=labels[row_ids],
                row_ids=row_ids.tolist(),
                tokenizer=tokenizer,
                max_length=int(params["max_length"]),
            )

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            datasets["train"],
            batch_size=int(params["batch_size"]),
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            datasets["val"],
            batch_size=int(params["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            datasets["test"],
            batch_size=int(params["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SOURCE,
            num_labels=len(label_names),
            problem_type="multi_label_classification",
            id2label=label_id_map,
            label2id=reverse_label_id_map,
            ignore_mismatched_sizes=True,
            local_files_only=LOCAL_FILES_ONLY or MODEL_SOURCE_IS_LOCAL,
        )
        model.to(device)

        pos_weight = build_pos_weight_with_cap(train_labels, float(params["max_pos_weight"])).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )

        total_training_steps = len(train_loader) * int(params["epochs"])
        warmup_steps = int(total_training_steps * float(params["warmup_ratio"]))
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        best_epoch = 0
        best_threshold = 0.50
        best_model_state = copy.deepcopy(model.state_dict())
        best_val_micro_f1 = float("-inf")
        best_threshold_scan = pd.DataFrame()
        history_rows = []

        for epoch in range(1, int(params["epochs"]) + 1):
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
            train_metrics = compute_multilabel_metrics(train_true, train_prob, epoch_threshold, label_names)
            val_metrics = compute_multilabel_metrics(val_true, val_prob, epoch_threshold, label_names)

            history_rows.append(
                {
                    "trial": trial.number,
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
                f"Trial {trial.number} | Epoch {epoch}/{int(params['epochs'])} | "
                f"val_micro_f1={val_metrics['micro']['f1']:.4f} | "
                f"threshold={epoch_threshold:.2f}"
            )

            trial.report(val_metrics["micro"]["f1"], step=epoch)

            if val_metrics["micro"]["f1"] > best_val_micro_f1:
                best_val_micro_f1 = float(val_metrics["micro"]["f1"])
                best_epoch = epoch
                best_threshold = float(epoch_threshold)
                best_model_state = copy.deepcopy(model.state_dict())
                best_threshold_scan = threshold_scan.copy()

        history_df = pd.DataFrame(history_rows)
        model.load_state_dict(best_model_state)

        _, final_val_true, final_val_prob, _ = run_epoch(
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

        validation_metrics = compute_multilabel_metrics(final_val_true, final_val_prob, best_threshold, label_names)
        test_metrics = compute_multilabel_metrics(test_true, test_prob, best_threshold, label_names)
        validation_sentiment_metrics = compute_sentiment_metrics(
            final_val_true,
            final_val_prob,
            best_threshold,
            label_names,
        )
        test_sentiment_metrics = compute_sentiment_metrics(test_true, test_prob, best_threshold, label_names)
        test_predictions = build_predictions_frame(
            df=df,
            row_ids=test_row_ids,
            y_true=test_true,
            y_prob=test_prob,
            threshold=best_threshold,
            label_names=label_names,
            text_col=text_col,
        )

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_threshold", best_threshold)
        trial.set_user_attr("validation_micro_f1", validation_metrics["micro"]["f1"])
        trial.set_user_attr("validation_macro_f1", validation_metrics["macro"]["f1"])
        trial.set_user_attr("test_micro_f1", test_metrics["micro"]["f1"])
        trial.set_user_attr("test_macro_f1", test_metrics["macro"]["f1"])
        trial.set_user_attr("test_sentiment_accuracy", test_sentiment_metrics["accuracy"])

        if best_val_micro_f1 > best_tracker["value"]:
            best_tracker["value"] = best_val_micro_f1
            best_tracker["trial_number"] = trial.number
            save_best_run_artifacts(
                dataset_path=dataset_path,
                label_names=label_names,
                split_sizes=split_sizes,
                device=device,
                trial_number=trial.number,
                selected_hyperparameters=params,
                best_epoch=best_epoch,
                best_threshold=best_threshold,
                best_threshold_scan=best_threshold_scan,
                history_df=history_df,
                validation_metrics=validation_metrics,
                validation_sentiment_metrics=validation_sentiment_metrics,
                test_metrics=test_metrics,
                test_sentiment_metrics=test_sentiment_metrics,
                test_predictions=test_predictions,
                model=model,
                tokenizer=tokenizer,
            )

        del train_loader, val_loader, test_loader, model, optimizer, scheduler, loss_fn, pos_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_micro_f1

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("Optuna did not complete any trials.")

    trial_records = []
    for trial in study.trials:
        trial_records.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "objective_value": trial.value,
                **trial.params,
                **{f"user_{key}": value for key, value in trial.user_attrs.items()},
            }
        )

    trial_frame = pd.DataFrame(trial_records)
    trial_frame.to_csv(TRIALS_CSV_PATH, index=False)
    with open(TRIALS_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(trial_records, handle, indent=2)

    try:
        importances = optuna.importance.get_param_importances(study)
    except ValueError:
        importances = {}
    importance_frame = pd.DataFrame(
        [{"parameter": name, "importance": value} for name, value in importances.items()]
    )
    if not importance_frame.empty:
        importance_frame.to_csv(PARAM_IMPORTANCE_PATH, index=False)

    summary_payload = {
        "study_name": OPTUNA_STUDY_NAME,
        "objective_metric": OBJECTIVE_METRIC,
        "seed": OPTUNA_SEED,
        "trials_requested": OPTUNA_TRIALS,
        "trials_completed": len(completed_trials),
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "output_dir": str(OUTPUT_DIR),
    }
    with open(STUDY_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print()
    print(f"Saved best Optuna model to: {MODEL_DIR}")
    print(f"Saved best-run metrics to: {METRICS_PATH}")
    print(f"Saved study trials to: {TRIALS_CSV_PATH}")
    if PARAM_IMPORTANCE_PATH.exists():
        print(f"Saved parameter importances to: {PARAM_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
