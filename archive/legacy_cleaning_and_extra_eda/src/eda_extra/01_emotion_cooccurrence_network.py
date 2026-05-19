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

import networkx as nx

def main():
    df, text_col, emotion_cols, topic_cols = load_preferred_data()

    binary = df[emotion_cols].astype(int)
    cooc = binary.T.dot(binary)

    G = nx.Graph()
    for col in emotion_cols:
        G.add_node(display_name(col))

    threshold = max(2, int(len(df) * 0.02))

    for i, col_i in enumerate(emotion_cols):
        for j, col_j in enumerate(emotion_cols):
            if j <= i:
                continue
            weight = int(cooc.loc[col_i, col_j])
            if weight >= threshold:
                G.add_edge(display_name(col_i), display_name(col_j), weight=weight)

    plt.figure(figsize=(10, 8))
    if G.number_of_edges() == 0:
        plt.text(0.5, 0.5, "No edges passed the threshold.", ha="center", va="center")
        plt.axis("off")
    else:
        pos = nx.spring_layout(G, seed=42, k=1.0)
        node_order = list(G.nodes())
        node_sizes = []
        for node in node_order:
            original_col = next(c for c in emotion_cols if display_name(c) == node)
            node_sizes.append(600 + 8 * int(binary[original_col].sum()))
        edge_widths = [0.4 + G[u][v]["weight"] / 10 for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, nodelist=node_order, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=9)
        plt.title("Emotion co-occurrence network")
        plt.axis("off")

    out = OUTPUT_DIR / "extra_fig1_emotion_cooccurrence_network.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
