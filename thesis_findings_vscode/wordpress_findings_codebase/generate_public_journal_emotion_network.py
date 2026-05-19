from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "output" / "selected_site_analysis" / "public_reflection_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "output" / "selected_site_analysis" / "report"
FIGURE_PATH = REPORT_DIR / "fig6_public_journal_emotion_cooccurrence_network.png"
EDGE_LIST_PATH = REPORT_DIR / "public_journal_emotion_cooccurrence_edges.csv"


def friendly_label(column: str) -> str:
    return column.split("__", 1)[1].replace("_", " ").title()


def load_binary_matrix() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["error"].fillna("") == ""].copy()
    emotion_cols = [c for c in df.columns if c.startswith("emotion__") and not c.startswith("emotion_prob__")]
    binary = df[emotion_cols].fillna(0).astype(int).copy()
    binary.columns = [friendly_label(column) for column in emotion_cols]
    return binary


def build_graph(binary: pd.DataFrame) -> tuple[nx.Graph, pd.DataFrame, int]:
    cooccurrence = binary.T.dot(binary)
    threshold = max(2, int(len(binary) * 0.02))

    graph = nx.Graph()
    for label in binary.columns:
        graph.add_node(label)

    edge_rows: list[dict[str, int | str]] = []
    for i, label_i in enumerate(binary.columns):
        for j, label_j in enumerate(binary.columns):
            if j <= i:
                continue
            weight = int(cooccurrence.loc[label_i, label_j])
            if weight >= threshold:
                graph.add_edge(label_i, label_j, weight=weight)
                edge_rows.append(
                    {
                        "source": label_i,
                        "target": label_j,
                        "cooccurrence_count": weight,
                    }
                )

    edge_frame = pd.DataFrame(edge_rows).sort_values("cooccurrence_count", ascending=False)
    return graph, edge_frame, threshold


def plot_graph(binary: pd.DataFrame, graph: nx.Graph, threshold: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 8.4))

    if graph.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No emotion pairs passed the threshold.", ha="center", va="center", fontsize=13)
        ax.axis("off")
    else:
        positions = nx.spring_layout(graph, seed=42, k=1.15, weight="weight")
        node_order = list(graph.nodes())
        node_sizes = [650 + 0.55 * int(binary[node].sum()) for node in node_order]
        edge_widths = [0.5 + graph[u][v]["weight"] / 180 for u, v in graph.edges()]

        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            width=edge_widths,
            alpha=0.28,
            edge_color="#7A8AA0",
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            nodelist=node_order,
            node_size=node_sizes,
            node_color="#3E86C3",
            alpha=0.9,
            linewidths=0.8,
            edgecolors="#2E5B84",
        )
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=8.5, font_color="#17324D")

        ax.set_title("Public journals: Emotion co-occurrence network", fontsize=15, pad=12)
        ax.axis("off")

    note = (
        f"Nodes scale with emotion prevalence; edges show predicted co-occurrence across reflections. "
        f"Only pairs with at least {threshold} joint occurrences are shown."
    )
    fig.text(0.02, 0.01, note, ha="left", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    binary = load_binary_matrix()
    graph, edge_frame, threshold = build_graph(binary)
    edge_frame.to_csv(EDGE_LIST_PATH, index=False)
    plot_graph(binary, graph, threshold)
    print(f"Saved figure to: {FIGURE_PATH}")
    print(f"Saved edge list to: {EDGE_LIST_PATH}")


if __name__ == "__main__":
    main()
