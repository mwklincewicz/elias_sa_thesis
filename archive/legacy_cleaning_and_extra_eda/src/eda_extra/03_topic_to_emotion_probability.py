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

    matrix = pd.DataFrame(index=topic_cols, columns=emotion_cols, dtype=float)

    for t in topic_cols:
        mask = df[t].astype(int) == 1
        for e in emotion_cols:
            matrix.loc[t, e] = df.loc[mask, e].astype(int).mean() if mask.any() else float("nan")

    matrix.index = [display_name(x) for x in matrix.index]
    matrix.columns = [display_name(x) for x in matrix.columns]

    plt.figure(figsize=(11, 6.2))
    sns.heatmap(matrix.astype(float), annot=False, cmap="magma")
    plt.title("Topic to emotion conditional probability")
    plt.xlabel("Emotion")
    plt.ylabel("Topic")
    out = OUTPUT_DIR / "extra_fig3_topic_to_emotion_probability.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
