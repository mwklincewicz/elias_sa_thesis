import matplotlib.pyplot as plt

import textwrap

from common import OUTPUT_DIR, display_name, load_dataset

def shorten(text: str, width: int = 58) -> str:
    return textwrap.fill(text, width=width)

def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    sample = (
        df.assign(total_labels=df["emotion_count"] + df["topic_count"])
          .sort_values(["total_labels", "word_count"], ascending=[False, False])
          .head(5)
          .copy()
    )

    rows = []
    for _, row in sample.iterrows():
        emotions = [display_name(c) for c in emotion_cols if bool(row[c])]
        topics = [display_name(c) for c in topic_cols if bool(row[c])]
        excerpt = str(row[text_col])[:220]
        if len(str(row[text_col])) > 220:
            excerpt += "..."

        rows.append([
            int(row["word_count"]),
            ", ".join(emotions[:5]) if emotions else "-",
            ", ".join(topics[:4]) if topics else "-",
            shorten(excerpt, 58),
        ])

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Words", "Emotions", "Topics", "Excerpt"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.0)
    plt.title("Representative labeled entries", pad=18)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig8_representative_entries.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
