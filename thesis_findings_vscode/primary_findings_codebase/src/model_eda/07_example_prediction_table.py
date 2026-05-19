from __future__ import annotations

import matplotlib.pyplot as plt

from common import load_test_predictions, parse_label_set, save_figure, truncate_text, wrap_text


def _row_f1(tp: int, fp: int, fn: int) -> float:
    denominator = (2 * tp) + fp + fn
    if denominator == 0:
        return 1.0
    return (2 * tp) / denominator


def _choose_rows(candidates, selected_ids: set[int], amount: int):
    chosen = []
    for row in candidates.itertuples(index=False):
        if int(row.row_id) in selected_ids:
            continue
        chosen.append(row)
        selected_ids.add(int(row.row_id))
        if len(chosen) == amount:
            break
    return chosen


def main() -> None:
    predictions = load_test_predictions().copy()

    predictions["true_set"] = predictions["true_emotions"].apply(parse_label_set)
    predictions["predicted_set"] = predictions["predicted_emotions"].apply(parse_label_set)
    predictions["true_count"] = predictions["true_set"].apply(len)
    predictions["predicted_count"] = predictions["predicted_set"].apply(len)
    predictions["tp"] = predictions.apply(
        lambda row: len(row["true_set"] & row["predicted_set"]),
        axis=1,
    )
    predictions["fp"] = predictions.apply(
        lambda row: len(row["predicted_set"] - row["true_set"]),
        axis=1,
    )
    predictions["fn"] = predictions.apply(
        lambda row: len(row["true_set"] - row["predicted_set"]),
        axis=1,
    )
    predictions["label_f1"] = predictions.apply(
        lambda row: _row_f1(int(row["tp"]), int(row["fp"]), int(row["fn"])),
        axis=1,
    )
    predictions["exact_match"] = predictions["true_set"] == predictions["predicted_set"]
    predictions["sentiment_match"] = (
        predictions["true_sentiment"] == predictions["predicted_sentiment"]
    )

    selected_ids: set[int] = set()

    strong_candidates = predictions.sort_values(
        ["exact_match", "label_f1", "true_count", "row_id"],
        ascending=[False, False, False, True],
    )
    strong_rows = _choose_rows(strong_candidates, selected_ids, 2)

    overprediction_candidates = predictions[
        (predictions["predicted_count"] > predictions["true_count"]) & (predictions["fp"] > 0)
    ].sort_values(
        ["fp", "predicted_count", "label_f1", "row_id"],
        ascending=[False, False, False, True],
    )
    overprediction_rows = _choose_rows(overprediction_candidates, selected_ids, 2)

    miss_candidates = predictions[
        (predictions["fn"] > 0) | (~predictions["sentiment_match"])
    ].sort_values(
        ["label_f1", "fn", "fp", "row_id"],
        ascending=[True, False, False, True],
    )
    miss_rows = _choose_rows(miss_candidates, selected_ids, 2)

    chosen_rows = strong_rows + overprediction_rows + miss_rows
    if len(chosen_rows) < 6:
        fallback_candidates = predictions.sort_values(
            ["label_f1", "row_id"],
            ascending=[True, True],
        )
        chosen_rows.extend(_choose_rows(fallback_candidates, selected_ids, 6 - len(chosen_rows)))

    row_labels = [
        "Strong match 1",
        "Strong match 2",
        "Overprediction 1",
        "Overprediction 2",
        "Harder miss 1",
        "Harder miss 2",
    ]

    rows = []
    for row in chosen_rows:
        rows.append(
            [
                wrap_text(truncate_text(row.text, 118), 38),
                wrap_text(row.true_emotions, 28),
                wrap_text(row.predicted_emotions, 28),
                str(row.true_sentiment),
                str(row.predicted_sentiment),
            ]
        )

    fig, ax = plt.subplots(figsize=(16.5, 8.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Reflection excerpt",
            "True emotions",
            "Predicted emotions",
            "True sentiment",
            "Predicted sentiment",
        ],
        rowLabels=row_labels[: len(rows)],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.35)

    row_colors = {
        1: "#eaf4ea",
        2: "#eaf4ea",
        3: "#fff3df",
        4: "#fff3df",
        5: "#fdecec",
        6: "#fdecec",
    }
    for row_index, color in row_colors.items():
        for column_index in range(-1, 5):
            if (row_index, column_index) in table.get_celld():
                table[(row_index, column_index)].set_facecolor(color)

    ax.set_title("Representative model prediction patterns", pad=18)
    fig.tight_layout()

    out = save_figure(fig, "fig7_example_prediction_table.png", dpi=240)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
