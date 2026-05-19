import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from common import OUTPUT_DIR, display_name, load_dataset

def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()

    matrix = pd.DataFrame(
        index=[display_name(t) for t in topic_cols],
        columns=[display_name(e) for e in emotion_cols],
        dtype=float,
    )

    for t in topic_cols:
        topic_mask = df[t].astype(bool)
        for e in emotion_cols:
            value = df.loc[topic_mask, e].mean() if topic_mask.any() else np.nan
            matrix.loc[display_name(t), display_name(e)] = value

    plt.figure(figsize=(11, 5.8))
    sns.heatmap(matrix.astype(float), cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Emotion prevalence conditional on topic")
    plt.xlabel("Emotion")
    plt.ylabel("Topic")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_topic_emotion.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
