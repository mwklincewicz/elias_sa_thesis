from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

CSV_PATH = DATA_DIR / "Lemotif thesis data.csv"
CLEANED_CSV_PATH = OUTPUT_DIR / "lemotif_cleaned.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
