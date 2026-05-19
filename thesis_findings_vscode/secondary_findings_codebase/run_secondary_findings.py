from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "src/clean_data.py",
    "src/test_data/prepare_stories_data.py",
    "src/test_data/run_stories_eda.py",
    "src/eda/01_text_length_distribution.py",
    "src/eda/02_emotion_prevalence.py",
    "src/eda/03_topic_prevalence.py",
    "src/eda/04_emotion_correlation_matrix.py",
    "src/eda/05_topic_emotion_heatmap.py",
    "src/eda/06_emotion_labels_per_entry.py",
    "src/eda/07_topic_labels_per_entry.py",
    "src/eda/08_representative_entries.py",
    "src/eda/09_emotion_cooccurrence_network.py",
    "src/eda/10_length_vs_emotion_probability.py",
    "src/eda/11_topic_to_emotion_probability.py",
    "src/grouped_eda/run_grouped_eda.py",
    "src/presentation/plot_emotion_dataset_comparison.py",
    "src/presentation/plot_sentiment_dataset_comparison.py",
    "src/findings/02_secondary_findings.py",
]


def run_script(script: str) -> None:
    script_path = ROOT / script
    print(f"Running {script_path}...")
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Failed while running {script}")


def main() -> None:
    for script in SCRIPTS:
        run_script(script)

    print()
    print(f"Secondary findings written to: {ROOT / 'output' / 'findings'}")


if __name__ == "__main__":
    main()
