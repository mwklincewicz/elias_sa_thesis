from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelTask:
    name: str
    description: str
    scripts: tuple[str, ...]
    outputs: tuple[str, ...]
    prerequisites: str
    notes: str = ""


TASKS: dict[str, ModelTask] = {
    "prepare-data": ModelTask(
        name="prepare-data",
        description="Clean Lemotif and format Stories into the Lemotif label schema.",
        scripts=("src/clean_data.py", "src/test_data/prepare_stories_data.py"),
        outputs=("output/lemotif_cleaned.csv", "test data/stories_data.csv"),
        prerequisites="Raw files in data/: Lemotif thesis data.csv and stories_source.xlsx.",
    ),
    "bert-baseline": ModelTask(
        name="bert-baseline",
        description="Train the main fine-tuned BERT multi-label emotion baseline on Lemotif.",
        scripts=("src/train_bert.py",),
        outputs=("output/bert_emotion_model/",),
        prerequisites="Run prepare-data first. Requires transformers/torch and BERT model access or local cache.",
        notes="Set LEMOTIF_LOCAL_FILES_ONLY=1 to force cached/local Hugging Face files.",
    ),
    "bert-baseline-figures": ModelTask(
        name="bert-baseline-figures",
        description="Recreate BERT held-out-test figures from the baseline artifacts.",
        scripts=("src/model_eda/run_model_eda.py",),
        outputs=("output/bert_emotion_model/figures/",),
        prerequisites="Run bert-baseline first.",
    ),
    "bert-threshold-cardinality": ModelTask(
        name="bert-threshold-cardinality",
        description="Run the threshold/cardinality hyperparameter test for the fine-tuned BERT predictions.",
        scripts=("src/model_eda/run_threshold_cardinality_test.py",),
        outputs=("output/bert_emotion_model/threshold_cardinality_test/",),
        prerequisites="Run bert-baseline first.",
        notes="Default threshold is 0.59 and minimum predicted labels is 1.",
    ),
    "bert-hyperparameter-fixed": ModelTask(
        name="bert-hyperparameter-fixed",
        description=(
            "Build the BERT-Hyperparamter-Fixed segment with standard figures, "
            "comparison tables, and comparison graphs."
        ),
        scripts=(
            "src/model_eda/run_threshold_cardinality_test.py",
            "src/model_eda/build_bert_hyperparameter_fixed.py",
        ),
        outputs=("output/BERT-Hyperparamter-Fixed/",),
        prerequisites="Run bert-baseline first.",
        notes="Uses unchanged fine-tuned BERT weights with threshold 0.59 and minimum predicted labels 1.",
    ),
    "bert-optuna": ModelTask(
        name="bert-optuna",
        description="Run Optuna hyperparameter tuning for BERT.",
        scripts=("src/train_bert_optuna.py",),
        outputs=("output/bert_emotion_model_optuna/",),
        prerequisites="Run prepare-data first. Requires baseline dependencies.",
        notes="Control runtime with LEMOTIF_OPTUNA_TRIALS. Default is 10 trials.",
    ),
    "bert-optuna-comparison": ModelTask(
        name="bert-optuna-comparison",
        description="Compare manual-default BERT against Optuna-tuned BERT.",
        scripts=("src/model_eda/run_optuna_comparison.py",),
        outputs=("output/bert_emotion_model_optuna/comparisons/",),
        prerequisites="Run bert-baseline and bert-optuna first.",
    ),
    "stories-external-bert": ModelTask(
        name="stories-external-bert",
        description="Train on full Lemotif and evaluate on the Stories external test set.",
        scripts=("src/test_data/train_full_lemotif_external.py",),
        outputs=("test data/output/external_evaluation/",),
        prerequisites="Run prepare-data and preferably bert-baseline first.",
        notes="Uses the prior BERT threshold/model directory when available.",
    ),
    "untrained-bert": ModelTask(
        name="untrained-bert",
        description="Evaluate random-weight BERT with the same config/tokenizer as a control baseline.",
        scripts=("src/test_data/evaluate_untrained_bert.py",),
        outputs=("test data/output/untrained_bert/",),
        prerequisites="Run prepare-data and preferably stories-external-bert or bert-baseline first.",
    ),
    "untrained-bert-comparison": ModelTask(
        name="untrained-bert-comparison",
        description="Compare trained vs untrained BERT on Stories.",
        scripts=("src/test_data/run_untrained_bert_comparisons.py",),
        outputs=("test data/output/untrained_bert/comparisons/",),
        prerequisites="Run stories-external-bert and untrained-bert first.",
    ),
    "gemma-lemotif": ModelTask(
        name="gemma-lemotif",
        description="Run the Gemma 2B Lemotif zero-shot/fine-tuned emotion pipeline.",
        scripts=("src/gemma_2b_emotion_pipeline.py",),
        outputs=("output/gemma_2b_emotion_model/",),
        prerequisites="Run prepare-data and bert-baseline first.",
        notes="Requires Gemma model access, a suitable transformers build, and substantial compute.",
    ),
    "gemma-stories-sentiment": ModelTask(
        name="gemma-stories-sentiment",
        description="Evaluate Gemma on Stories derived sentiment.",
        scripts=("src/test_data/evaluate_gemma_sentiment.py",),
        outputs=("test data/output/gemma_sentiment/",),
        prerequisites="Run prepare-data first.",
        notes="Requires accepted Gemma license and Hugging Face login/token unless cached locally.",
    ),
    "gemma-stories-figures": ModelTask(
        name="gemma-stories-figures",
        description="Create Gemma Stories sentiment figures and BERT-vs-Gemma sentiment comparisons.",
        scripts=("src/test_data/run_gemma_visuals.py",),
        outputs=("test data/output/gemma_sentiment/**/figures/", "test data/output/gemma_sentiment/comparisons/"),
        prerequisites="Run stories-external-bert and gemma-stories-sentiment first.",
    ),
    "dataset-comparisons": ModelTask(
        name="dataset-comparisons",
        description="Recreate Lemotif-vs-Stories emotion and sentiment distribution comparisons.",
        scripts=(
            "src/presentation/plot_emotion_dataset_comparison.py",
            "src/presentation/plot_sentiment_dataset_comparison.py",
        ),
        outputs=("output/dataset_comparison/",),
        prerequisites="Run prepare-data first.",
    ),
}


RECOMMENDED_ORDER = [
    "prepare-data",
    "bert-baseline",
    "bert-baseline-figures",
    "bert-threshold-cardinality",
    "bert-hyperparameter-fixed",
    "bert-optuna",
    "bert-optuna-comparison",
    "stories-external-bert",
    "untrained-bert",
    "untrained-bert-comparison",
    "dataset-comparisons",
    "gemma-lemotif",
    "gemma-stories-sentiment",
    "gemma-stories-figures",
]


def print_task(task: ModelTask) -> None:
    print(f"{task.name}")
    print(f"  {task.description}")
    print(f"  Prerequisites: {task.prerequisites}")
    print(f"  Scripts: {', '.join(task.scripts)}")
    print(f"  Outputs: {', '.join(task.outputs)}")
    if task.notes:
        print(f"  Notes: {task.notes}")
    print()


def parse_env_overrides(raw_values: list[str]) -> dict[str, str]:
    overrides = {}
    for raw in raw_values:
        if "=" not in raw:
            raise SystemExit(f"Environment override must be KEY=VALUE, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Environment override has an empty key: {raw}")
        overrides[key] = value
    return overrides


def run_script(script: str, env: dict[str, str], dry_run: bool) -> None:
    script_path = ROOT / script
    command = [sys.executable, str(script_path)]
    print(f"Running: {' '.join(command)}")
    if dry_run:
        return

    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"Failed while running {script}")


def run_task(task_name: str, env: dict[str, str], dry_run: bool) -> None:
    task = TASKS[task_name]
    print_task(task)
    for script in task.scripts:
        run_script(script, env=env, dry_run=dry_run)


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(parse_env_overrides(args.env or []))
    if args.local_files_only:
        env["LEMOTIF_LOCAL_FILES_ONLY"] = "1"
        env["GEMMA_LOCAL_FILES_ONLY"] = "1"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run thesis model-rebuild tasks one by one."
    )
    parser.add_argument(
        "task",
        nargs="?",
        choices=["list", "recommended-order", *TASKS.keys()],
        default="list",
        help="Task to run. Use `list` to inspect available model rebuilds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force local/cached model loading where supported.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Set an environment override, e.g. --env LEMOTIF_EPOCHS=1.",
    )
    args = parser.parse_args()

    if args.task == "list":
        print("Available model rebuild tasks:\n")
        for name in RECOMMENDED_ORDER:
            print_task(TASKS[name])
        return

    if args.task == "recommended-order":
        print("Recommended one-by-one run order:\n")
        for index, name in enumerate(RECOMMENDED_ORDER, start=1):
            print(f"{index}. {name}")
        print()
        print("Run one task with:")
        print("  python run_models.py <task-name>")
        return

    run_task(args.task, env=build_env(args), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
