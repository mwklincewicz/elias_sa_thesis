from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    GROUP_ORDER,
    OUTPUT_DIR,
    build_group_frame,
    build_other_positive_label_count,
    group_display_name,
    load_dataset,
    topic_display_name,
)


def main() -> None:
    df, text_col, emotion_cols, topic_cols = load_dataset()
    group_frame, group_columns = build_group_frame(df, emotion_cols)

    importance_rows = []
    score_rows = []

    for group_key in GROUP_ORDER:
        y = group_frame[group_key]
        class_counts = y.value_counts().sort_index()
        if len(class_counts) < 2:
            continue

        n_splits = min(5, int(class_counts.min()))
        if n_splits < 2:
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

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = cross_val_score(model, features, y, cv=cv, scoring="roc_auc")
        ap_scores = cross_val_score(model, features, y, cv=cv, scoring="average_precision")

        score_rows.append(
            {
                "group_key": group_key,
                "group": group_display_name(group_key),
                "positive_rows": int(y.sum()),
                "roc_auc_mean": float(auc_scores.mean()),
                "roc_auc_std": float(auc_scores.std()),
                "average_precision_mean": float(ap_scores.mean()),
                "average_precision_std": float(ap_scores.std()),
                "n_splits": int(n_splits),
            }
        )

        model.fit(features, y)
        permutation = permutation_importance(
            model,
            features,
            y,
            scoring="roc_auc",
            n_repeats=25,
            random_state=42,
        )

        feature_names = ["word_count", "other_positive_label_count", *topic_cols]
        for feature_name, mean_importance, std_importance in zip(
            feature_names,
            permutation.importances_mean,
            permutation.importances_std,
        ):
            if feature_name in topic_cols:
                feature_label = topic_display_name(feature_name)
                feature_type = "topic"
            elif feature_name == "word_count":
                feature_label = "Word Count"
                feature_type = "control"
            else:
                feature_label = "Other Positive Label Count"
                feature_type = "control"

            importance_rows.append(
                {
                    "group_key": group_key,
                    "group": group_display_name(group_key),
                    "feature_key": feature_name,
                    "feature": feature_label,
                    "feature_type": feature_type,
                    "importance_mean": float(mean_importance),
                    "importance_std": float(std_importance),
                }
            )

    importance = pd.DataFrame(importance_rows).sort_values(
        by=["group", "importance_mean"],
        ascending=[True, False],
    )
    scores = pd.DataFrame(score_rows).sort_values("roc_auc_mean", ascending=False)

    importance_path = OUTPUT_DIR / "group_feature_importance.csv"
    scores_path = OUTPUT_DIR / "group_model_scores.csv"
    importance.to_csv(importance_path, index=False)
    scores.to_csv(scores_path, index=False)

    plot_rows = (
        importance.sort_values(["group", "importance_mean"], ascending=[True, False])
        .groupby("group", as_index=False, group_keys=False)
        .head(5)
        .copy()
    )
    plot_rows["feature_with_group"] = plot_rows["feature"] + " | " + plot_rows["group"]

    fig_height = max(6.5, 0.42 * max(1, len(plot_rows)))
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    sns.barplot(data=plot_rows, x="importance_mean", y="feature_with_group", hue="group", dodge=False, palette="tab10", ax=ax)
    ax.set_title("Top grouped-emotion feature importances")
    ax.set_xlabel("Permutation importance on ROC-AUC")
    ax.set_ylabel("")
    ax.legend(title="Grouped emotion", bbox_to_anchor=(1.02, 1), loc="upper left")

    figure_path = OUTPUT_DIR / "fig4_group_feature_importance.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {importance_path}")
    print(f"Saved: {scores_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
