import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import OUTPUT_DIR, display_name, load_dataset


def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    rows = []
    for topic in topic_cols:
        topic_mask = df[topic].astype(bool)
        if not topic_mask.any():
            continue

        emotion_probs = df.loc[topic_mask, emotion_cols].mean().sort_values(ascending=False)
        top_emotion = emotion_probs.index[0]
        rows.append(
            {
                "topic": display_name(topic),
                "emotion": display_name(top_emotion),
                "probability": float(emotion_probs.iloc[0]),
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("probability", ascending=True)

    plt.figure(figsize=(9.5, max(4.8, 0.55 * max(1, len(plot_df)))))
    if plot_df.empty:
        plt.text(0.5, 0.5, "No topic-emotion probabilities could be computed.", ha="center", va="center")
        plt.axis("off")
    else:
        ax = sns.scatterplot(
            data=plot_df,
            x="probability",
            y="topic",
            hue="emotion",
            s=170,
        )
        for row in plot_df.itertuples(index=False):
            ax.text(row.probability + 0.01, row.topic, row.emotion, va="center", fontsize=9)

        plt.title("Most likely emotion within each topic")
        plt.xlabel("Conditional probability")
        plt.ylabel("Topic")
        plt.xlim(0, min(1.05, max(0.35, plot_df["probability"].max() + 0.12)))
        plt.legend(title="Top emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = OUTPUT_DIR / "fig11_topic_to_emotion_probability.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
