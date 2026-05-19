from __future__ import annotations

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


def main() -> None:
    df, text_col, emotion_cols, topic_cols = load_dataset()
    group_frame, group_columns = build_group_frame(df, emotion_cols)

    prevalence_rows = []
    for key in GROUP_ORDER:
        prevalence_rows.append(
            {
                "group": group_display_name(key),
                "prevalence": float(group_frame[key].mean()),
                "avg_member_labels_per_entry": float(group_frame[f"{key}_count"].mean()),
            }
        )

    prevalence_rows.extend(
        [
            {
                "group": "Any Positive Group",
                "prevalence": float(group_frame["any_positive_group"].mean()),
                "avg_member_labels_per_entry": float(group_frame["positive_group_count"].mean()),
            },
            {
                "group": "Any Negative Group",
                "prevalence": float(group_frame["any_negative_group"].mean()),
                "avg_member_labels_per_entry": float(group_frame["negative_group_count"].mean()),
            },
            {
                "group": "Mixed Affect",
                "prevalence": float(group_frame["mixed_affect"].mean()),
                "avg_member_labels_per_entry": float(group_frame["group_count"].mean()),
            },
        ]
    )

    prevalence = pd.DataFrame(prevalence_rows)
    prevalence_path = OUTPUT_DIR / "group_prevalence.csv"
    prevalence.to_csv(prevalence_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ordered_prevalence = prevalence.iloc[: len(GROUP_ORDER)].copy()
    ordered_prevalence["group"] = pd.Categorical(
        ordered_prevalence["group"],
        categories=GROUP_DISPLAY_ORDER,
        ordered=True,
    )
    ordered_prevalence = ordered_prevalence.sort_values("group")

    sns.barplot(
        data=ordered_prevalence,
        x="prevalence",
        y="group",
        orient="h",
        color="#4C72B0",
        ax=axes[0],
    )
    axes[0].set_title("Grouped emotion prevalence")
    axes[0].set_xlabel("Share of reflections")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, min(1.0, ordered_prevalence["prevalence"].max() + 0.10))

    for row in ordered_prevalence.itertuples(index=False):
        axes[0].text(row.prevalence + 0.01, row.group, f"{row.prevalence:.2f}", va="center", fontsize=9)

    group_count = group_frame["group_count"].value_counts().sort_index().rename_axis("active_groups").reset_index(
        name="count"
    )
    sns.barplot(data=group_count, x="active_groups", y="count", color="#4C72B0", ax=axes[1])
    axes[1].set_title("Grouped emotions per reflection")
    axes[1].set_xlabel("Number of active grouped emotions")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    figure_path = OUTPUT_DIR / "fig1_group_prevalence_and_density.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {prevalence_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
