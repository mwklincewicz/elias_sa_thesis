from __future__ import annotations

from common import GROUP_DEFINITIONS, OUTPUT_DIR, build_group_frame, load_dataset, make_group_summary


def main() -> None:
    df, text_col, emotion_cols, topic_cols = load_dataset()
    group_frame, group_columns = build_group_frame(df, emotion_cols)
    summary = make_group_summary(df, group_frame, group_columns)

    summary_path = OUTPUT_DIR / "group_summary.csv"
    summary.to_csv(summary_path, index=False)

    mapping_rows = []
    for group in GROUP_DEFINITIONS:
        for member in group["members"]:
            mapping_rows.append(
                {
                    "group_key": group["key"],
                    "group_label": group["display"],
                    "group_kind": group["kind"],
                    "emotion_key": member,
                }
            )

    mapping_path = OUTPUT_DIR / "group_definition_table.csv"
    mapping_frame = summary.__class__(mapping_rows)
    mapping_frame.to_csv(mapping_path, index=False)

    summary_text = (
        "Grouped emotion EDA summary\n"
        f"Dataset rows: {len(df)}\n"
        f"Text column: {text_col}\n"
        f"Emotion columns: {len(emotion_cols)}\n"
        f"Topic columns: {len(topic_cols)}\n"
        f"Positive-group prevalence: {group_frame['any_positive_group'].mean():.3f}\n"
        f"Negative-group prevalence: {group_frame['any_negative_group'].mean():.3f}\n"
        f"Mixed-affect prevalence: {group_frame['mixed_affect'].mean():.3f}\n"
        f"Average grouped labels per entry: {group_frame['group_count'].mean():.3f}\n"
    )
    summary_txt_path = OUTPUT_DIR / "grouping_summary.txt"
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {mapping_path}")
    print(f"Saved: {summary_txt_path}")


if __name__ == "__main__":
    main()
