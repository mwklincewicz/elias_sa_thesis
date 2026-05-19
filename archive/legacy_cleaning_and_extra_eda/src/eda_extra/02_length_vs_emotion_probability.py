from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLEANED_CSV_PATH, CSV_PATH, OUTPUT_DIR
from data_loader import load_lemotif_data, display_name

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def load_preferred_data():
    path = CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() else CSV_PATH
    return load_lemotif_data(path)

def main():
    df, text_col, emotion_cols, topic_cols = load_preferred_data()

    df["word_count"] = df[text_col].astype(str).str.split().str.len()
    top_emotions = df[emotion_cols].astype(int).mean().sort_values(ascending=False).head(4).index.tolist()

    bins = [0, 10, 20, 40, 60, 100, 150, 300]
    labels = ["1-10", "11-20", "21-40", "41-60", "61-100", "101-150", "151-299"]
    df["length_bin"] = pd.cut(df["word_count"], bins=bins, labels=labels, include_lowest=True, right=True)

    plot_df = (
        df.groupby("length_bin", observed=False)[top_emotions]
        .mean()
        .reset_index()
        .melt(id_vars="length_bin", var_name="emotion", value_name="probability")
    )
    plot_df["emotion"] = plot_df["emotion"].apply(display_name)

    plt.figure(figsize=(10, 5.8))
    sns.lineplot(data=plot_df, x="length_bin", y="probability", hue="emotion", marker="o")
    plt.title("Reflection length vs emotion probability")
    plt.xlabel("Reflection length bin (words)")
    plt.ylabel("Probability of emotion label")
    plt.xticks(rotation=20)
    out = OUTPUT_DIR / "extra_fig2_length_vs_emotion_probability.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
