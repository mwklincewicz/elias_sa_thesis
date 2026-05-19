import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import OUTPUT_DIR, display_name, load_dataset


def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()
    top_emotions = df[emotion_cols].mean().sort_values(ascending=False).head(4).index.tolist()

    bins = [0, 10, 20, 40, 60, 100, 150, 300]
    labels = ["1-10", "11-20", "21-40", "41-60", "61-100", "101-150", "151-299"]
    df = df.copy()
    df["length_bin"] = pd.cut(df["word_count"], bins=bins, labels=labels, include_lowest=True, right=True)

    plot_df = (
        df.groupby("length_bin", observed=False)[top_emotions]
        .mean()
        .reset_index()
        .melt(id_vars="length_bin", var_name="emotion", value_name="probability")
    )
    plot_df["emotion"] = plot_df["emotion"].map(display_name)

    plt.figure(figsize=(10, 5.8))
    sns.lineplot(data=plot_df, x="length_bin", y="probability", hue="emotion", marker="o")
    plt.title("Reflection length vs emotion probability")
    plt.xlabel("Reflection length bin (words)")
    plt.ylabel("Probability of emotion label")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig10_length_vs_emotion_probability.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
