import os
import subprocess
import sys

from config import EDA_SCRIPTS, GROUPED_EDA_SCRIPTS, MODEL_SCRIPT, PROJECT_ROOT

PIPELINE_SCRIPTS = ["src/clean_data.py", *EDA_SCRIPTS, *GROUPED_EDA_SCRIPTS]

if os.getenv("LEMOTIF_RUN_MODEL", "0") == "1":
    PIPELINE_SCRIPTS.append(MODEL_SCRIPT)


def main():
    total_steps = len(PIPELINE_SCRIPTS)

    for index, script in enumerate(PIPELINE_SCRIPTS, start=1):
        script_path = PROJECT_ROOT / script
        print(f"[{index}/{total_steps}] Running {script_path}...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Failed while running {script}")

    if MODEL_SCRIPT in PIPELINE_SCRIPTS:
        print("Cleaning, EDA, and modeling pipeline completed.")
    else:
        print("Cleaning and EDA pipeline completed.")


if __name__ == "__main__":
    main()
