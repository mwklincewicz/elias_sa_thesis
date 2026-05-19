from __future__ import annotations

from common import (
    FORMATTING_SUMMARY_PATH,
    STORIES_CSV_PATH,
    TEST_DATA_DIR,
    build_formatting_summary_text,
    build_stories_dataframe,
)


def main() -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stories_df, summary = build_stories_dataframe()
    stories_df.to_csv(STORIES_CSV_PATH, index=False)

    summary_text = build_formatting_summary_text(summary)
    FORMATTING_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"Saved formatted Stories CSV to: {STORIES_CSV_PATH}")


if __name__ == "__main__":
    main()
