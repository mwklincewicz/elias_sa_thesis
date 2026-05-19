from config import CLEANED_CSV_PATH, CLEANING_SUMMARY_PATH, RAW_CSV_PATH
from data_loader import load_lemotif_data, refresh_derived_columns

MIN_TEXT_LENGTH = 10
MAX_WORD_COUNT = 300


def clean_lemotif_data():
    df, text_col, emotion_cols, topic_cols = load_lemotif_data(RAW_CSV_PATH)
    original_n = len(df)

    cleaned = df.copy()
    cleaned[text_col] = cleaned[text_col].str.replace(r"http\S+|www\.\S+", "", regex=True)
    cleaned[text_col] = cleaned[text_col].str.replace(r"\s+", " ", regex=True).str.strip()

    short_mask = cleaned[text_col].str.len() <= MIN_TEXT_LENGTH
    removed_short = int(short_mask.sum())
    cleaned = cleaned.loc[~short_mask].copy()

    no_emotion_mask = cleaned[emotion_cols].sum(axis=1) == 0
    removed_no_emotion = int(no_emotion_mask.sum())
    cleaned = cleaned.loc[~no_emotion_mask].copy()

    duplicate_mask = cleaned.duplicated(subset=[text_col], keep="first")
    removed_duplicates = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()

    cleaned = refresh_derived_columns(cleaned, text_col, emotion_cols, topic_cols)

    long_mask = cleaned["word_count"] >= MAX_WORD_COUNT
    removed_long = int(long_mask.sum())
    cleaned = cleaned.loc[~long_mask].copy()
    cleaned = refresh_derived_columns(cleaned, text_col, emotion_cols, topic_cols)

    summary = {
        "text_col": text_col,
        "emotion_col_count": len(emotion_cols),
        "topic_col_count": len(topic_cols),
        "original_rows": original_n,
        "removed_short_rows": removed_short,
        "removed_no_emotion_rows": removed_no_emotion,
        "removed_duplicate_rows": removed_duplicates,
        "removed_long_rows": removed_long,
        "final_rows": len(cleaned),
    }
    return cleaned, summary


def build_summary_text(summary: dict[str, int | str]) -> str:
    return (
        "Lemotif cleaning summary\n"
        "=======================\n"
        f"Raw file: {RAW_CSV_PATH}\n"
        f"Cleaned file: {CLEANED_CSV_PATH}\n\n"
        f"Detected text column: {summary['text_col']}\n"
        f"Detected emotion columns: {summary['emotion_col_count']}\n"
        f"Detected topic columns: {summary['topic_col_count']}\n\n"
        f"Original rows: {summary['original_rows']}\n"
        f"Removed empty/near-empty rows (<= {MIN_TEXT_LENGTH} chars): {summary['removed_short_rows']}\n"
        f"Removed rows without emotion labels: {summary['removed_no_emotion_rows']}\n"
        f"Removed duplicate reflections: {summary['removed_duplicate_rows']}\n"
        f"Removed long-text outliers (>= {MAX_WORD_COUNT} words): {summary['removed_long_rows']}\n"
        f"Final rows: {summary['final_rows']}\n"
    )


def main():
    cleaned, summary = clean_lemotif_data()
    cleaned.to_csv(CLEANED_CSV_PATH, index=False)

    summary_text = build_summary_text(summary)
    CLEANING_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"Saved cleaned CSV to: {CLEANED_CSV_PATH}")
    print(f"Saved cleaning summary to: {CLEANING_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
