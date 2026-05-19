import matplotlib.pyplot as plt
import seaborn as sns

from common import OUTPUT_DIR, display_name, load_dataset

def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    corr = df[emotion_cols].corr()
    renamed = [display_name(c) for c in emotion_cols]
    corr.index = renamed
    corr.columns = renamed

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, vmin=-1, vmax=1)
    plt.title("Emotion label correlations")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_emotion_corr.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
