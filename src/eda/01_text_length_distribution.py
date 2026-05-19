import matplotlib.pyplot as plt

from common import load_dataset, OUTPUT_DIR

def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    plt.figure(figsize=(8, 4.8))
    plt.hist(df["word_count"], bins=30, edgecolor="white")
    plt.title("Distribution of reflection length")
    plt.xlabel("Word count")
    plt.ylabel("Number of entries")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_word_count_distribution.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
