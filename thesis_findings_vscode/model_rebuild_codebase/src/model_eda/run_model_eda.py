from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
MODEL_EDA_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_FILE.parents[2]

MODEL_EDA_SCRIPTS = [
    "01_training_progress.py",
    "02_validation_vs_test_metrics.py",
    "03_per_emotion_f1.py",
    "04_precision_recall_scatter.py",
    "05_true_vs_predicted_label_counts.py",
    "06_sentiment_confusion_matrix.py",
    "07_example_prediction_table.py",
]


def main() -> None:
    total_steps = len(MODEL_EDA_SCRIPTS)

    for index, script_name in enumerate(MODEL_EDA_SCRIPTS, start=1):
        script_path = MODEL_EDA_DIR / script_name
        print(f"[{index}/{total_steps}] Running {script_path}...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Failed while running {script_name}")

    print("Model evaluation visuals completed.")


if __name__ == "__main__":
    main()
