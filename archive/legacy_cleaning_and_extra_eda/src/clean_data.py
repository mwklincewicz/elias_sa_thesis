from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CSV_PATH, CLEANED_CSV_PATH, OUTPUT_DIR
from data_loader import load_lemotif_data

def main():
    df, text_col, emotion_cols, topic_cols = load_lemotif_data(CSV_PATH)

    original_n = len(df)

    df[text_col] = df[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df[text_col] = df[text_col].str.replace(r"http\S+|www\.\S+", "", regex=True).str.strip()

    before_short = len(df)
    df = df[df[text_col].str.len() > 10].copy()
    removed_short = before_short - len(df)

    df[emotion_cols] = df[emotion_cols].fillna(0).astype(int)
    df[topic_cols] = df[topic_cols].fillna(0).astype(int)

    before_no_emotion = len(df)
    df = df[df[emotion_cols].sum(axis=1) > 0].copy()
    removed_no_emotion = before_no_emotion - len(df)

    before_dupes = len(df)
    df = df.drop_duplicates(subset=text_col).copy()
    removed_dupes = before_dupes - len(df)

    df["char_count"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()
    df["emotion_count"] = df[emotion_cols].sum(axis=1)
    df["topic_count"] = df[topic_cols].sum(axis=1)

    before_long = len(df)
    df = df[df["word_count"] < 300].copy()
    removed_long = before_long - len(df)

    df.to_csv(CLEANED_CSV_PATH, index=False)

    summary_path = OUTPUT_DIR / "cleaning_summary.txt"
    summary = (
        "Lemotif cleaning summary\n"
        "=======================\n"
        f"Original rows: {original_n}\n"
        f"Removed empty/near-empty rows: {removed_short}\n"
        f"Removed no-emotion-label rows: {removed_no_emotion}\n"
        f"Removed duplicate reflections: {removed_dupes}\n"
        f"Removed long-text outliers (>=300 words): {removed_long}\n"
        f"Final rows: {len(df)}\n\n"
        f"Output file:\n{CLEANED_CSV_PATH}\n"
    )
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Saved cleaned CSV to: {CLEANED_CSV_PATH}")
    print(f"Saved cleaning summary to: {summary_path}")

if __name__ == "__main__":
    main()
