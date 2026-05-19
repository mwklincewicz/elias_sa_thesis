from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    GROUP_DISPLAY_ORDER,
    GROUP_ORDER,
    OUTPUT_DIR,
    build_group_frame,
    build_other_positive_label_count,
    group_display_name,
    load_dataset,
    topic_display_name,
)


def fit_topic_adjusted_coefficients(
    df: pd.DataFrame,
    group_frame: pd.DataFrame,
    topic_cols: list[str],
    group_columns: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []

    for group_key in GROUP_ORDER:
        y = group_frame[group_key]
        if int(y.sum()) < 10:
            continue

        features = pd.DataFrame(index=df.index)
        features["word_count"] = df["word_count"].astype(float)
        features["other_positive_label_count"] = build_other_positive_label_count(df, group_key, group_columns).astype(
            float
        )
        for topic in topic_cols:
            features[topic] = df[topic].astype(int)

        numeric_features = ["word_count", "other_positive_label_count"]
        preprocessor = ColumnTransformer(
            [
                ("scale", StandardScaler(), numeric_features),
                ("pass", "passthrough", topic_cols),
            ]
        )
        model = Pipeline(
            [
                ("pre", preprocessor),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
            ]
        )
        model.fit(features, y)

        coefficients = model.named_steps["clf"].coef_[0]
        topic_coefficients = coefficients[len(numeric_features) :]
        for topic, coefficient in zip(topic_cols, topic_coefficients):
            rows.append(
                {
                    "group_key": group_key,
                    "group": group_display_name(group_key),
                    "topic_key": topic.replace("Answer.t1.", "").replace(".raw", ""),
                    "topic": topic_display_name(topic),
                    "adjusted_log_odds": float(coefficient),
                }
            )

    return pd.DataFrame(rows)


def make_raw_prevalence_matrix(
    df: pd.DataFrame,
    group_frame: pd.DataFrame,
    topic_cols: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=[topic_display_name(topic) for topic in topic_cols], columns=GROUP_DISPLAY_ORDER)

    for topic in topic_cols:
        topic_mask = df[topic].astype(bool)
        for group_key in GROUP_ORDER:
            value = group_frame.loc[topic_mask, group_key].mean() if topic_mask.any() else float("nan")
            matrix.loc[topic_display_name(topic), group_display_name(group_key)] = value

    return matrix.astype(float)


def main() -> None:
    df, text_col, emotion_cols, topic_cols = load_dataset()
    group_frame, group_columns = build_group_frame(df, emotion_cols)

    raw_matrix = make_raw_prevalence_matrix(df, group_frame, topic_cols)
    raw_path = OUTPUT_DIR / "topic_group_raw_prevalence.csv"
    raw_matrix.to_csv(raw_path)

    adjusted = fit_topic_adjusted_coefficients(df, group_frame, topic_cols, group_columns)
    adjusted_path = OUTPUT_DIR / "topic_group_adjusted_coefficients.csv"
    adjusted.to_csv(adjusted_path, index=False)

    adjusted_matrix = adjusted.pivot(index="topic", columns="group", values="adjusted_log_odds")
    adjusted_matrix = adjusted_matrix.reindex(index=[topic_display_name(topic) for topic in topic_cols], columns=GROUP_DISPLAY_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    sns.heatmap(raw_matrix, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Raw grouped emotion prevalence by topic")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(adjusted_matrix, cmap="vlag", center=0, ax=axes[1])
    axes[1].set_title("Adjusted topic effects after positive control")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.tight_layout()
    figure_path = OUTPUT_DIR / "fig3_topic_group_raw_vs_adjusted.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {raw_path}")
    print(f"Saved: {adjusted_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
