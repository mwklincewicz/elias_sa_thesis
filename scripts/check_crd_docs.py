from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MO_PATH = ROOT / "docs" / "METHODOLOGY_OVERVIEW.md"
README_PATH = ROOT / "README.md"
DATA_README_PATH = ROOT / "data" / "README.md"

MO_REQUIRED_PHRASES = [
    "High-Level Pipeline",
    "Dataset Description and Access",
    "Cleaning, Preprocessing, Transformation, and Splitting",
    "Implemented Machine-Learning Models",
    "Evaluation Strategy and Error Analysis",
    "Reproducibility Commands",
    "FAIR and Ethics Boundary",
    "src/train_bert.py",
    "src/train_bert_optuna.py",
    "src/test_data/run_all.py",
    "wordpress_findings_codebase",
]


def require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required CRD file: {path.relative_to(ROOT)}")


def require_phrases(path: Path, phrases: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [phrase for phrase in phrases if phrase not in text]
    if missing:
        formatted = ", ".join(missing)
        raise SystemExit(f"{path.relative_to(ROOT)} is missing required phrase(s): {formatted}")


def main() -> None:
    require_path(MO_PATH)
    require_path(README_PATH)
    require_path(DATA_README_PATH)
    require_phrases(MO_PATH, MO_REQUIRED_PHRASES)
    require_phrases(README_PATH, ["Methodology Overview", "make check", "Code Repository Deliverable"])
    require_phrases(DATA_README_PATH, ["Lemotif thesis data.csv", "stories_source.xlsx", "restricted"])
    print("CRD documentation checks passed.")


if __name__ == "__main__":
    main()
