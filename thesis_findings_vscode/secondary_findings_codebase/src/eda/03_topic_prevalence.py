import matplotlib.pyplot as plt

from common import OUTPUT_DIR, display_name, load_dataset

def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    shares = df[topic_cols].mean().sort_values(ascending=False) * 100
    labels = [display_name(c) for c in shares.index]

    plt.figure(figsize=(8.8, max(4.5, 0.38 * len(labels))))
    plt.barh(labels[::-1], shares.values[::-1])
    plt.title("Topic label prevalence")
    plt.xlabel("Share of entries (%)")
    plt.ylabel("")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_topic_prevalence.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
