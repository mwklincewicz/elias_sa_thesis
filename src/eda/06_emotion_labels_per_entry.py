import matplotlib.pyplot as plt

from common import OUTPUT_DIR, load_dataset

def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    counts = df["emotion_count"].value_counts().sort_index()

    plt.figure(figsize=(7.2, 4.5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Number of emotion labels per entry")
    plt.xlabel("Emotion labels assigned")
    plt.ylabel("Number of entries")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig6_emotion_labels_per_entry.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
