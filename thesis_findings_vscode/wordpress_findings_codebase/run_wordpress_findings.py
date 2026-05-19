from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    scripts: tuple[str, ...]
    outputs: tuple[str, ...]
    notes: str = ""


TASKS: dict[str, Task] = {
    "smoke": Task(
        name="smoke",
        description="Run the deterministic no-network smoke test with a stub analyzer.",
        scripts=("tests/smoke_test.py",),
        outputs=("output_smoke/journal_sentiment_results.csv",),
        notes="Use this first to confirm the WordPress pipeline code imports and writes outputs.",
    ),
    "train-theme-model": Task(
        name="train-theme-model",
        description="Train the local Lemotif topic/theme BERT model used for public journal theme labels.",
        scripts=("train_theme_bert.py",),
        outputs=("output/bert_theme_model/model/", "output/bert_theme_model/metrics.json"),
        notes=(
            "Requires a cleaned Lemotif file from one of the sibling thesis codebases. "
            "Override with LEMOTIF_THEME_DATASET_PATH if needed."
        ),
    ),
    "selected-sites": Task(
        name="selected-sites",
        description="Scrape the selected WordPress site list and run emotion/theme inference.",
        scripts=("run_selected_site_pipeline.py",),
        outputs=("output/selected_site_analysis/public_reflection_predictions.csv",),
        notes=(
            "Requires internet access, the emotion BERT model, and the trained theme model. "
            "Override model paths with WORDPRESS_EMOTION_MODEL_PATH and WORDPRESS_THEME_MODEL_PATH."
        ),
    ),
    "report": Task(
        name="report",
        description="Regenerate aggregate WordPress report figures and tables from selected-site results.",
        scripts=(
            "generate_selected_site_report.py",
            "generate_public_journal_emotion_profile.py",
            "generate_public_journal_emotion_network.py",
        ),
        outputs=("output/selected_site_analysis/report/",),
        notes="Requires output/selected_site_analysis/public_reflection_predictions.csv from the selected-sites step.",
    ),
    "thesis-tables": Task(
        name="thesis-tables",
        description="Regenerate thesis input tables and collection metadata.",
        scripts=("generate_thesis_input_tables.py",),
        outputs=("output/selected_site_analysis/thesis_input_tables/",),
        notes="The full input dataset table contains public post text and is intentionally not bundled by default.",
    ),
}

RECOMMENDED_ORDER = [
    "smoke",
    "train-theme-model",
    "selected-sites",
    "report",
    "thesis-tables",
]


def print_task(task: Task) -> None:
    print(task.name)
    print(f"  {task.description}")
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


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(parse_env_overrides(args.env or []))
    return env


def run_script(script: str, env: dict[str, str], dry_run: bool) -> None:
    command = [sys.executable, str(ROOT / script)]
    print(f"Running: {' '.join(command)}")
    if dry_run:
        return
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"Failed while running {script}")


def run_task(name: str, env: dict[str, str], dry_run: bool) -> None:
    task = TASKS[name]
    print_task(task)
    for script in task.scripts:
        run_script(script, env=env, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the WordPress public-reflection reproducibility steps."
    )
    parser.add_argument(
        "task",
        nargs="?",
        choices=["list", "recommended-order", "all", *TASKS.keys()],
        default="list",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Set an environment override, e.g. --env WORDPRESS_EMOTION_MODEL_PATH=path.",
    )
    args = parser.parse_args()

    if args.task == "list":
        print("Available WordPress reproducibility tasks:\n")
        for task_name in RECOMMENDED_ORDER:
            print_task(TASKS[task_name])
        return

    if args.task == "recommended-order":
        print("Recommended WordPress run order:\n")
        for index, task_name in enumerate(RECOMMENDED_ORDER, start=1):
            print(f"{index}. {task_name}")
        print()
        print("Run one task with:")
        print("  python run_wordpress_findings.py <task-name>")
        return

    env = build_env(args)
    if args.task == "all":
        for task_name in RECOMMENDED_ORDER:
            run_task(task_name, env=env, dry_run=args.dry_run)
        return

    run_task(args.task, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
