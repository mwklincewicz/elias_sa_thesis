import matplotlib.pyplot as plt

from common import OUTPUT_DIR, load_dataset

def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    counts = df["topic_count"].value_counts().sort_index()

    plt.figure(figsize=(7.2, 4.5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Number of topic labels per entry")
    plt.xlabel("Topic labels assigned")
    plt.ylabel("Number of entries")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig7_topic_labels_per_entry.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
