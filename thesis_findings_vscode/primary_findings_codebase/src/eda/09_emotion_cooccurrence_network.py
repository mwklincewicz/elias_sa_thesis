import matplotlib.pyplot as plt
import networkx as nx

from common import OUTPUT_DIR, display_name, load_dataset


def main():
    df, text_col, emotion_cols, topic_cols = load_dataset()
    binary = df[emotion_cols]
    cooccurrence = binary.T.dot(binary)
    label_map = {col: display_name(col) for col in emotion_cols}

    graph = nx.Graph()
    for label in label_map.values():
        graph.add_node(label)

    threshold = max(2, int(len(df) * 0.02))
    for i, col_i in enumerate(emotion_cols):
        for j, col_j in enumerate(emotion_cols):
            if j <= i:
                continue
            weight = int(cooccurrence.loc[col_i, col_j])
            if weight >= threshold:
                graph.add_edge(label_map[col_i], label_map[col_j], weight=weight)

    plt.figure(figsize=(10, 8))
    if graph.number_of_edges() == 0:
        plt.text(0.5, 0.5, "No edges passed the threshold.", ha="center", va="center")
        plt.axis("off")
    else:
        positions = nx.spring_layout(graph, seed=42, k=1.0)
        node_order = list(graph.nodes())
        label_to_col = {label: col for col, label in label_map.items()}
        node_sizes = [600 + 8 * int(binary[label_to_col[node]].sum()) for node in node_order]
        edge_widths = [0.4 + graph[u][v]["weight"] / 10 for u, v in graph.edges()]

        nx.draw_networkx_nodes(graph, positions, nodelist=node_order, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(graph, positions, width=edge_widths, alpha=0.5)
        nx.draw_networkx_labels(graph, positions, font_size=9)
        plt.title("Emotion co-occurrence network")
        plt.axis("off")

    out = OUTPUT_DIR / "fig9_emotion_cooccurrence_network.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
