from __future__ import annotations

from pathlib import Path
import subprocess
import sys

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import GROUPED_EDA_SCRIPTS, PROJECT_ROOT


def main() -> None:
    total_steps = len(GROUPED_EDA_SCRIPTS)

    for index, script in enumerate(GROUPED_EDA_SCRIPTS, start=1):
        script_path = PROJECT_ROOT / script
        print(f"[{index}/{total_steps}] Running {script_path}...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Failed while running {script}")

    print("Grouped emotion EDA completed.")


if __name__ == "__main__":
    main()
