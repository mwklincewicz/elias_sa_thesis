from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TEST_DATA_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_FILE.parents[2]

PIPELINE_SCRIPTS = [
    "src/test_data/prepare_stories_data.py",
    "src/test_data/run_stories_eda.py",
    "src/test_data/train_full_lemotif_external.py",
]


def main() -> None:
    total_steps = len(PIPELINE_SCRIPTS)

    for index, script in enumerate(PIPELINE_SCRIPTS, start=1):
        script_path = PROJECT_ROOT / script
        print(f"[{index}/{total_steps}] Running {script_path}...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Failed while running {script}")

    print("Stories test-data workflow completed.")


if __name__ == "__main__":
    main()
