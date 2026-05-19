from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FINDING_SCRIPTS = [
    "src/findings/01_primary_findings.py",
    "src/findings/02_secondary_findings.py",
]


def main() -> None:
    for index, script in enumerate(FINDING_SCRIPTS, start=1):
        script_path = ROOT / script
        print(f"[{index}/{len(FINDING_SCRIPTS)}] Running {script_path}...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Failed while running {script}")

    print(f"Saved findings summaries under: {ROOT / 'output' / 'findings'}")


if __name__ == "__main__":
    main()
