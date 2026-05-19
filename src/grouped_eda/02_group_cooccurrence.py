from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (
    GROUP_DISPLAY_ORDER,
    GROUP_ORDER,
    OUTPUT_DIR,
    build_group_frame,
    group_display_name,
    load_dataset,
)


def compute_pair_metrics(binary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phi = pd.DataFrame(index=GROUP_DISPLAY_ORDER, columns=GROUP_DISPLAY_ORDER, dtype=float)
    lift = pd.DataFrame(index=GROUP_DISPLAY_ORDER, columns=GROUP_DISPLAY_ORDER, dtype=float)

    for row_key in GROUP_ORDER:
        row_label = group_display_name(row_key)
        pa = float(binary[row_key].mean())
        for col_key in GROUP_ORDER:
            col_label = group_display_name(col_key)
            pb = float(binary[col_key].mean())

            if row_key == col_key:
                phi.loc[row_label, col_label] = 1.0
                lift.loc[row_label, col_label] = 1.0
                continue

            pab = float((binary[row_key] & binary[col_key]).mean())
            denom = math.sqrt(pa * (1 - pa) * pb * (1 - pb))
            phi.loc[row_label, col_label] = 0.0 if denom == 0 else (pab - (pa * pb)) / denom
            lift.loc[row_label, col_label] = 0.0 if pa == 0 or pb == 0 else pab / (pa * pb)

    return phi, lift


def main() -> None:
    df, text_col, emotion_cols, topic_cols = load_dataset()
    group_frame, group_columns = build_group_frame(df, emotion_cols)
    binary = group_frame[GROUP_ORDER]

    phi, lift = compute_pair_metrics(binary)
    phi_path = OUTPUT_DIR / "group_phi_matrix.csv"
    lift_path = OUTPUT_DIR / "group_lift_matrix.csv"
    phi.to_csv(phi_path)
    lift.to_csv(lift_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    sns.heatmap(phi.astype(float), annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[0])
    axes[0].set_title("Grouped emotion association (phi)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(lift.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, ax=axes[1])
    axes[1].set_title("Grouped emotion co-occurrence lift")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.tight_layout()
    figure_path = OUTPUT_DIR / "fig2_group_cooccurrence_heatmaps.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {phi_path}")
    print(f"Saved: {lift_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
