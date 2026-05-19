from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

LIGHTWEIGHT_SCRIPTS = [
    "src/clean_data.py",
    "src/test_data/prepare_stories_data.py",
    "src/test_data/run_stories_eda.py",
    "src/presentation/plot_emotion_dataset_comparison.py",
    "src/presentation/plot_sentiment_dataset_comparison.py",
]

FULL_TRAINING_SCRIPTS = [
    "src/train_bert.py",
    "src/model_eda/run_model_eda.py",
    "src/test_data/train_full_lemotif_external.py",
    "src/test_data/evaluate_untrained_bert.py",
    "src/test_data/run_untrained_bert_comparisons.py",
]

SUMMARY_SCRIPTS = [
    "src/model_eda/run_threshold_cardinality_test.py",
    "src/model_eda/build_bert_hyperparameter_fixed.py",
    "src/presentation/plot_primary_model_comparison.py",
    "src/findings/01_primary_findings.py",
]


def run_script(script: str) -> None:
    script_path = ROOT / script
    print(f"Running {script_path}...")
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Failed while running {script}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the thesis primary findings artifacts."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Retrain BERT and rerun Stories external evaluations. This can take a long time.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run the heavier BERT-Hyperparamter-Fixed diagnostics suite, including LIME explanations.",
    )
    args = parser.parse_args()

    scripts = list(LIGHTWEIGHT_SCRIPTS)
    if args.full:
        scripts.extend(FULL_TRAINING_SCRIPTS)
    else:
        print("Skipping model retraining. Using bundled metrics/artifacts where available.")
        print("Use --full to rebuild BERT and external-test artifacts from scratch.")

    scripts.extend(SUMMARY_SCRIPTS)
    if args.diagnostics:
        scripts.append("src/model_eda/run_bert_fixed_diagnostics.py")

    for script in scripts:
        run_script(script)

    print()
    print(f"Primary findings written to: {ROOT / 'output' / 'findings'}")


if __name__ == "__main__":
    main()
