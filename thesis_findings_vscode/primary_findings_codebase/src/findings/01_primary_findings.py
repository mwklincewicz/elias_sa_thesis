from __future__ import annotations

import json
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
TEST_OUTPUT_DIR = ROOT / "test data" / "output"
FINDINGS_DIR = OUTPUT_DIR / "findings"
FIXED_BERT_DIR = OUTPUT_DIR / "BERT-Hyperparamter-Fixed"

PRIMARY_MARKDOWN_PATH = FINDINGS_DIR / "primary_findings.md"
PRIMARY_METRICS_PATH = FINDINGS_DIR / "primary_metrics.csv"
PRIMARY_ARTIFACT_INDEX_PATH = FINDINGS_DIR / "primary_artifact_index.csv"
ADJUSTED_BERT_SUMMARY_PATH = FINDINGS_DIR / "adjusted_bert_inference_summary.md"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def nested(payload: dict | None, *keys: str) -> object | None:
    current: object = payload or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def metric_record(
    *,
    finding: str,
    dataset: str,
    model: str,
    payload: dict,
    split_label: str,
) -> dict[str, object]:
    split_sizes = payload.get("split_sizes", {})
    return {
        "finding": finding,
        "dataset": dataset,
        "model": model,
        "split": split_label,
        "rows": split_sizes.get(split_label, split_sizes.get("test")),
        "threshold": payload.get("best_threshold", payload.get("threshold", payload.get("tested_threshold"))),
        "minimum_predicted_labels": payload.get("minimum_predicted_labels"),
        "model_weights": payload.get("model_weights"),
        "emotion_micro_f1": nested(payload, "test_metrics", "micro", "f1"),
        "emotion_macro_f1": nested(payload, "test_metrics", "macro", "f1"),
        "emotion_subset_accuracy": nested(payload, "test_metrics", "subset_accuracy"),
        "emotion_hamming_loss": nested(payload, "test_metrics", "hamming_loss"),
        "avg_true_labels": nested(payload, "test_metrics", "avg_true_labels_per_entry"),
        "avg_predicted_labels": nested(payload, "test_metrics", "avg_predicted_labels_per_entry"),
        "zero_predicted_rows": nested(payload, "test_metrics", "zero_predicted_rows"),
        "overpredicted_by_at_least_2_rows": nested(
            payload, "test_metrics", "overpredicted_by_at_least_2_rows"
        ),
        "sentiment_accuracy": nested(payload, "test_sentiment_metrics", "accuracy"),
        "sentiment_macro_f1": nested(payload, "test_sentiment_metrics", "macro", "f1"),
    }


def per_label_brief(payload: dict | None, split_key: str = "test_metrics") -> tuple[str, str]:
    label_metrics = nested(payload, split_key, "label_metrics")
    if not label_metrics:
        return "not available", "not available"

    rows = [row for row in label_metrics if int(row.get("support", 0)) > 0]
    if not rows:
        return "not available", "not available"

    strongest = sorted(
        rows,
        key=lambda row: (float(row.get("f1", 0.0)), int(row.get("support", 0))),
        reverse=True,
    )[:3]
    weakest = sorted(
        rows,
        key=lambda row: (float(row.get("f1", 0.0)), -int(row.get("support", 0))),
    )[:3]

    def render(items: list[dict[str, object]]) -> str:
        parts = []
        for row in items:
            parts.append(
                f"{row.get('label')} F1={fmt(row.get('f1'))} (n={int(row.get('support', 0))})"
            )
        return "; ".join(parts)

    return render(strongest), render(weakest)


def label_count(value: str) -> int:
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "none":
        return 0
    return len([part.strip() for part in cleaned.split(",") if part.strip()])


def summarize_threshold_cardinality() -> list[str]:
    baseline = load_json(OUTPUT_DIR / "bert_emotion_model" / "metrics.json")
    adjusted = load_json(OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json")
    if baseline is None or adjusted is None:
        return ["- Threshold/cardinality adjusted run is not available yet."]

    baseline_predictions = load_csv(OUTPUT_DIR / "bert_emotion_model" / "test_predictions.csv")
    baseline_zero_rows: int | str = "not available"
    baseline_overpredicted_by_at_least_2: int | str = "not available"
    if baseline_predictions:
        surplus_values = []
        zero_count = 0
        for row in baseline_predictions:
            true_count = label_count(row.get("true_emotions", ""))
            predicted_count = label_count(row.get("predicted_emotions", ""))
            if predicted_count == 0:
                zero_count += 1
            surplus_values.append(predicted_count - true_count)
        baseline_zero_rows = zero_count
        baseline_overpredicted_by_at_least_2 = sum(value >= 2 for value in surplus_values)

    baseline_emotion = baseline["test_metrics"]
    adjusted_emotion = adjusted["test_metrics"]
    adjusted_sentiment = adjusted["test_sentiment_metrics"]
    return [
        f"- Evidence: `{rel(OUTPUT_DIR / 'bert_emotion_model' / 'threshold_cardinality_test' / 'metrics.json')}`",
        f"- Adjusted predictions: `{rel(OUTPUT_DIR / 'bert_emotion_model' / 'threshold_cardinality_test' / 'test_predictions_threshold_0_59_min1.csv')}`",
        f"- Model weights: {adjusted.get('model_weights', 'unchanged fine-tuned BERT')}",
        f"- Tested threshold: {float(adjusted['tested_threshold']):.2f}",
        f"- Minimum predicted emotions per reflection: {int(adjusted['minimum_predicted_labels'])}",
        (
            f"- Average predicted labels: baseline "
            f"{float(baseline_emotion['avg_predicted_labels_per_entry']):.3f} "
            f"vs adjusted {float(adjusted_emotion['avg_predicted_labels_per_entry']):.3f}"
        ),
        (
            f"- Rows with zero predicted emotions: baseline {baseline_zero_rows} "
            f"vs adjusted {int(adjusted_emotion['zero_predicted_rows'])}"
        ),
        (
            f"- Rows with at least two extra predicted emotions: baseline "
            f"{baseline_overpredicted_by_at_least_2} "
            f"vs adjusted {int(adjusted_emotion['overpredicted_by_at_least_2_rows'])}"
        ),
        f"- Adjusted emotion micro-F1: {fmt(adjusted_emotion['micro']['f1'])}",
        f"- Adjusted derived sentiment accuracy: {fmt(adjusted_sentiment['accuracy'])}",
        "",
        "Thesis use: include this as the earlier adjusted inference run that reduces emotion overestimation and prevents non-labeled/zero-emotion predictions. It is not a new trained model.",
    ]


def summarize_fixed_bert_segment() -> list[str]:
    if not FIXED_BERT_DIR.exists():
        return ["- BERT-Hyperparamter-Fixed segment is not available yet."]

    figures = sorted((FIXED_BERT_DIR / "figures").glob("fig*.png"))
    tables = sorted((FIXED_BERT_DIR / "comparisons").glob("*.csv"))
    graphs = sorted((FIXED_BERT_DIR / "comparisons").glob("fig*.png"))
    return [
        f"- Segment folder: `{rel(FIXED_BERT_DIR)}`",
        f"- Standard BERT figures generated: {len(figures)}",
        f"- Comparative tables generated: {len(tables)}",
        f"- Comparative graphs generated: {len(graphs)}",
        f"- Metrics: `{rel(FIXED_BERT_DIR / 'metrics.json')}`",
        f"- Predictions: `{rel(FIXED_BERT_DIR / 'test_predictions.csv')}`",
        f"- Run summary: `{rel(FIXED_BERT_DIR / 'run_summary.txt')}`",
    ]


def artifact_row(
    section: str,
    priority: str,
    path: Path,
    suggested_use: str,
) -> dict[str, object]:
    return {
        "section": section,
        "priority": priority,
        "path": rel(path),
        "exists": path.exists(),
        "suggested_use": suggested_use,
    }


def test_output_suggested_use(path: Path) -> str:
    try:
        category = path.relative_to(TEST_OUTPUT_DIR).parts[0]
    except (IndexError, ValueError):
        category = "test data"
    uses = {
        "eda": "Stories EDA output generated from the formatted test data.",
        "external_evaluation": "Stories external-test output for the trained Lemotif BERT model.",
        "untrained_bert": "Stories control output for the untrained BERT baseline.",
        "gemma_sentiment": "Stories sentiment output for the optional Gemma comparison.",
    }
    return uses.get(category, "Generated test-data output artifact.")


def test_output_inventory_rows(existing_paths: set[str]) -> list[dict[str, object]]:
    if not TEST_OUTPUT_DIR.exists():
        return []

    rows: list[dict[str, object]] = []
    for path in sorted(TEST_OUTPUT_DIR.rglob("*")):
        if not path.is_file():
            continue
        relative_path = rel(path)
        if relative_path in existing_paths:
            continue
        rows.append(
            artifact_row(
                "Test data output inventory",
                "Reference",
                path,
                test_output_suggested_use(path),
            )
        )
    return rows


def artifact_rows() -> list[dict[str, object]]:
    artifacts = [
        (
            "Dataset comparability",
            "Required",
            OUTPUT_DIR / "dataset_comparison" / "fig_emotion_distribution_comparison.png",
            "Use before model results to show Lemotif and Stories label-distribution shift.",
        ),
        (
            "Dataset comparability",
            "Required",
            OUTPUT_DIR / "dataset_comparison" / "fig_sentiment_distribution_comparison.png",
            "Use to justify derived sentiment as a summary of emotion labels.",
        ),
        (
            "In-domain BERT",
            "Required",
            OUTPUT_DIR / "bert_emotion_model" / "figures" / "fig2_validation_vs_test_metrics.png",
            "Use for validation/test agreement and headline metric reporting.",
        ),
        (
            "In-domain BERT",
            "Required",
            OUTPUT_DIR / "bert_emotion_model" / "figures" / "fig3_per_emotion_f1.png",
            "Use for RQ1 emotion-level performance.",
        ),
        (
            "In-domain BERT",
            "Recommended",
            OUTPUT_DIR / "bert_emotion_model" / "figures" / "fig6_sentiment_confusion_matrix.png",
            "Use to support the claim that sentiment is easier than exact emotion prediction.",
        ),
        (
            "Stories external test",
            "Required",
            OUTPUT_DIR / "presentation_assets" / "training_run_comparison.png",
            "Use for the domain-shift summary from Lemotif held-out test to Stories external test.",
        ),
        (
            "Stories external test",
            "Required",
            OUTPUT_DIR / "presentation_assets" / "stories_external_per_emotion_f1.png",
            "Use to show which Stories emotions generalize best and worst.",
        ),
        (
            "Training-control baseline",
            "Recommended",
            TEST_OUTPUT_DIR
            / "untrained_bert"
            / "comparisons"
            / "fig_compare_overall_emotion_metrics.png",
            "Use to show that Lemotif supervision improves over random BERT weights.",
        ),
        (
            "Adjusted BERT inference",
            "Required",
            OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "run_summary.txt",
            "Earlier adjusted run: threshold/cardinality inference change, no retrained weights.",
        ),
        (
            "Adjusted BERT inference",
            "Required",
            OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "test_predictions_threshold_0_59_min1.csv",
            "Adjusted predictions with reduced emotion overestimation and no zero-emotion predictions.",
        ),
        (
            "Adjusted BERT inference",
            "Recommended",
            OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "threshold_sweep.csv",
            "Threshold sweep used to justify the selected cardinality setting.",
        ),
        (
            "Adjusted BERT inference",
            "Required",
            ADJUSTED_BERT_SUMMARY_PATH,
            "Standalone thesis-facing summary of the adjusted inference run.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Required",
            FIXED_BERT_DIR / "run_summary.txt",
            "Standalone model-output segment for the adjusted inference run.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Required",
            FIXED_BERT_DIR / "metrics.json",
            "Metrics packaged for the BERT-Hyperparamter-Fixed segment.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Required",
            FIXED_BERT_DIR / "test_predictions.csv",
            "Adjusted test predictions used by the fixed segment figures.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Required",
            FIXED_BERT_DIR / "comparisons" / "overall_metrics_comparison.csv",
            "Overall baseline-vs-fixed comparison table.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Required",
            FIXED_BERT_DIR / "comparisons" / "per_emotion_metrics_comparison.csv",
            "Per-emotion baseline-vs-fixed comparison table.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Required",
            FIXED_BERT_DIR / "comparisons" / "prediction_cardinality_comparison.csv",
            "Prediction-count comparison table showing reduced overestimation.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Recommended",
            FIXED_BERT_DIR / "comparisons" / "fig_compare_overall_metrics.png",
            "Readable visual summary of the score, density, and overestimation tradeoffs.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Recommended",
            FIXED_BERT_DIR / "comparisons" / "fig_compare_per_emotion_f1.png",
            "Dumbbell plot showing per-emotion F1 changes from baseline to fixed inference.",
        ),
        (
            "BERT-Hyperparamter-Fixed",
            "Recommended",
            FIXED_BERT_DIR / "comparisons" / "fig_compare_prediction_cardinality.png",
            "Readable distribution plot for true vs baseline vs fixed predicted label counts.",
        ),
        *[
            (
                "BERT-Hyperparamter-Fixed",
                "Required",
                FIXED_BERT_DIR / "figures" / f"fig{index}_{name}.png",
                "Standard BERT figure regenerated for the fixed segment.",
            )
            for index, name in [
                (1, "training_progress"),
                (2, "validation_vs_test_metrics"),
                (3, "per_emotion_f1"),
                (4, "precision_recall_scatter"),
                (5, "true_vs_predicted_label_counts"),
                (6, "sentiment_confusion_matrix"),
                (7, "example_prediction_table"),
            ]
        ],
    ]
    rows = [artifact_row(section, priority, path, suggested_use) for section, priority, path, suggested_use in artifacts]
    rows.extend(test_output_inventory_rows({str(row["path"]) for row in rows}))
    return rows


def build_markdown(metrics: list[dict[str, object]], artifact_index: list[dict[str, object]]) -> str:
    baseline_row = next(
        (row for row in metrics if row.get("finding") == "RQ1 in-domain baseline"),
        {},
    )
    external_row = next(
        (row for row in metrics if row.get("finding") == "RQ1 Stories external test"),
        {},
    )

    bert_metrics = load_json(OUTPUT_DIR / "bert_emotion_model" / "metrics.json")
    external_metrics = load_json(TEST_OUTPUT_DIR / "external_evaluation" / "metrics.json")
    untrained_summary = load_csv(TEST_OUTPUT_DIR / "untrained_bert" / "comparison_summary.csv")

    baseline_best, baseline_weak = per_label_brief(bert_metrics)
    external_best, external_weak = per_label_brief(external_metrics)

    lines: list[str] = [
        "# Primary Findings Evidence Summary",
        "",
        "This file is generated from existing artifacts. It does not retrain models.",
        "",
        "## Finding 1: Lemotif-trained BERT works as the main in-domain baseline",
        "",
        f"- Evidence: `{rel(OUTPUT_DIR / 'bert_emotion_model' / 'metrics.json')}`",
        f"- Held-out test rows: {baseline_row.get('rows', 'not available')}",
        f"- Emotion micro-F1: {fmt(baseline_row.get('emotion_micro_f1'))}",
        f"- Emotion macro-F1: {fmt(baseline_row.get('emotion_macro_f1'))}",
        f"- Exact subset accuracy: {fmt(baseline_row.get('emotion_subset_accuracy'))}",
        f"- Hamming loss: {fmt(baseline_row.get('emotion_hamming_loss'))}",
        f"- Derived sentiment accuracy: {fmt(baseline_row.get('sentiment_accuracy'))}",
        f"- Best-supported emotion results: {baseline_best}",
        f"- Weakest nonzero-support emotion results: {baseline_weak}",
        "",
        "Thesis use: report this as the clean single-entry baseline for RQ1 before any model variants.",
        "",
        "## Finding 1b: Earlier adjusted BERT run reduces overestimated emotions",
        "",
        *summarize_threshold_cardinality(),
        "",
        "## Finding 1c: BERT-Hyperparamter-Fixed is packaged as its own output segment",
        "",
        *summarize_fixed_bert_segment(),
        "",
        "## Finding 2: Generalization to Stories drops substantially",
        "",
        f"- Evidence: `{rel(TEST_OUTPUT_DIR / 'external_evaluation' / 'metrics.json')}`",
        f"- Stories test rows in current artifact: {external_row.get('rows', 'not available')}",
        f"- Emotion micro-F1: {fmt(external_row.get('emotion_micro_f1'))}",
        f"- Emotion macro-F1: {fmt(external_row.get('emotion_macro_f1'))}",
        f"- Exact subset accuracy: {fmt(external_row.get('emotion_subset_accuracy'))}",
        f"- Hamming loss: {fmt(external_row.get('emotion_hamming_loss'))}",
        f"- Derived sentiment accuracy: {fmt(external_row.get('sentiment_accuracy'))}",
        f"- Best-supported emotion results: {external_best}",
        f"- Weakest nonzero-support emotion results: {external_weak}",
        "",
        "Thesis use: frame Stories as the external validity check. The current results support a domain-shift finding, not a claim of deployment-ready accuracy.",
        "",
    ]

    if untrained_summary:
        micro_row = next((row for row in untrained_summary if row.get("metric") == "Micro F1"), None)
        sentiment_row = next((row for row in untrained_summary if row.get("metric") == "Accuracy"), None)
        if micro_row:
            row = micro_row
            lines.extend(
                [
                    "## Finding 3: Lemotif supervision matters",
                    "",
                    f"- Trained Stories emotion micro-F1: {fmt(row.get('trained'))}",
                    f"- Untrained Stories emotion micro-F1: {fmt(row.get('untrained'))}",
                    f"- Difference: {fmt(row.get('delta_trained_minus_untrained'))}",
                ]
            )
        if sentiment_row:
            row = sentiment_row
            lines.extend(
                [
                    f"- Trained Stories sentiment accuracy: {fmt(row.get('trained'))}",
                    f"- Untrained Stories sentiment accuracy: {fmt(row.get('untrained'))}",
                    f"- Difference: {fmt(row.get('delta_trained_minus_untrained'))}",
                    "",
                    "Thesis use: this is a useful control showing the external-test result is not just random label behavior.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Visuals To Use First",
            "",
        ]
    )
    for row in artifact_index:
        if row["section"] == "Test data output inventory":
            continue
        status = "found" if row["exists"] else "missing"
        lines.append(f"- {row['priority']}: `{row['path']}` ({status}) - {row['suggested_use']}")

    test_output_rows = [row for row in artifact_index if row["section"] == "Test data output inventory"]
    if test_output_rows:
        lines.extend(["", "## Complete Test Data Output Inventory", ""])
        for row in test_output_rows:
            status = "found" if row["exists"] else "missing"
            lines.append(f"- `{row['path']}` ({status}) - {row['suggested_use']}")

    lines.extend(
        [
            "",
            "## Do Not Overclaim Yet",
            "",
            "- WordPress/public-blog validation is not present in the tracked source files.",
            "- SHAP or feature-attribution outputs are not present yet.",
            "- Krippendorff's alpha or inter-annotator agreement scripts are not present yet.",
            "- Translation of Stories entries is described in the thesis draft, but the current CSV still appears to contain Dutch text.",
            "",
            f"Machine-readable metrics: `{rel(PRIMARY_METRICS_PATH)}`",
            f"Artifact index: `{rel(PRIMARY_ARTIFACT_INDEX_PATH)}`",
            f"Adjusted BERT inference summary: `{rel(ADJUSTED_BERT_SUMMARY_PATH)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_adjusted_bert_summary_markdown() -> str:
    lines = [
        "# Adjusted BERT Inference Summary",
        "",
        "This is the earlier fine-tuned BERT run with adjusted inference settings. The model weights are unchanged; only the prediction threshold and minimum-label fallback are changed.",
        "",
        *summarize_threshold_cardinality(),
        "",
        "Primary placement: use this immediately after the main in-domain BERT result when discussing emotion overestimation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    bert_metrics = load_json(OUTPUT_DIR / "bert_emotion_model" / "metrics.json")
    if bert_metrics is not None:
        metric_rows.append(
            metric_record(
                finding="RQ1 in-domain baseline",
                dataset="Lemotif held-out split",
                model="bert-base-uncased",
                payload=bert_metrics,
                split_label="test",
            )
        )

    threshold_metrics = load_json(OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json")
    if threshold_metrics is not None:
        metric_rows.append(
            metric_record(
                finding="RQ1 adjusted threshold/cardinality run",
                dataset="Lemotif held-out split",
                model="bert-base-uncased (adjusted inference)",
                payload=threshold_metrics,
                split_label="test",
            )
        )

    external_metrics = load_json(TEST_OUTPUT_DIR / "external_evaluation" / "metrics.json")
    if external_metrics is not None:
        metric_rows.append(
            metric_record(
                finding="RQ1 Stories external test",
                dataset="Stories gold/external test",
                model="bert-base-uncased",
                payload=external_metrics,
                split_label="test",
            )
        )

    metric_fields = [
        "finding",
        "dataset",
        "model",
        "split",
        "rows",
        "threshold",
        "minimum_predicted_labels",
        "model_weights",
        "emotion_micro_f1",
        "emotion_macro_f1",
        "emotion_subset_accuracy",
        "emotion_hamming_loss",
        "avg_true_labels",
        "avg_predicted_labels",
        "zero_predicted_rows",
        "overpredicted_by_at_least_2_rows",
        "sentiment_accuracy",
        "sentiment_macro_f1",
    ]
    write_csv(PRIMARY_METRICS_PATH, metric_rows, fieldnames=metric_fields)

    ADJUSTED_BERT_SUMMARY_PATH.write_text(
        build_adjusted_bert_summary_markdown(),
        encoding="utf-8",
    )

    artifact_index = artifact_rows()
    write_csv(
        PRIMARY_ARTIFACT_INDEX_PATH,
        artifact_index,
        fieldnames=["section", "priority", "path", "exists", "suggested_use"],
    )

    PRIMARY_MARKDOWN_PATH.write_text(
        build_markdown(metric_rows, artifact_index),
        encoding="utf-8",
    )

    print(f"Saved primary findings markdown to: {PRIMARY_MARKDOWN_PATH}")
    print(f"Saved primary metrics table to: {PRIMARY_METRICS_PATH}")
    print(f"Saved primary artifact index to: {PRIMARY_ARTIFACT_INDEX_PATH}")
    print(f"Saved adjusted BERT summary to: {ADJUSTED_BERT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
