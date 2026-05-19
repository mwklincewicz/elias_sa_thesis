from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EDA_SCRIPTS
from test_data.common import STORIES_CSV_PATH, STORIES_EDA_OUTPUT_DIR


def main() -> None:
    if not STORIES_CSV_PATH.exists():
        raise SystemExit(
            f"Stories CSV not found at {STORIES_CSV_PATH}. Run src/test_data/prepare_stories_data.py first."
        )

    env = os.environ.copy()
    env["LEMOTIF_ANALYSIS_DATASET_PATH"] = str(STORIES_CSV_PATH)
    env["LEMOTIF_ANALYSIS_OUTPUT_DIR"] = str(STORIES_EDA_OUTPUT_DIR)

    total_steps = len(EDA_SCRIPTS)
    for index, script in enumerate(EDA_SCRIPTS, start=1):
        script_path = PROJECT_ROOT / script
        print(f"[{index}/{total_steps}] Running {script_path} on Stories evaluation data...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            raise SystemExit(f"Failed while running {script}")

    print(f"Stories EDA completed. Output folder: {STORIES_EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
