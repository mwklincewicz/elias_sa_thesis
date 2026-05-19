from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
TEST_OUTPUT_DIR = ROOT / "test data" / "output"
FINDINGS_DIR = OUTPUT_DIR / "findings"

SECONDARY_MARKDOWN_PATH = FINDINGS_DIR / "secondary_findings.md"
SECONDARY_TABLE_PATH = FINDINGS_DIR / "secondary_evidence_tables.csv"
SECONDARY_ARTIFACT_INDEX_PATH = FINDINGS_DIR / "secondary_artifact_index.csv"
ADJUSTED_BERT_SUMMARY_PATH = FINDINGS_DIR / "adjusted_bert_inference_summary.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def load_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def summarize_sentiment_shift() -> list[str]:
    summary = load_csv(OUTPUT_DIR / "dataset_comparison" / "sentiment_distribution_summary.csv")
    if not summary:
        return ["- Sentiment distribution comparison is not available yet."]

    lines = []
    for dataset in ["Lemotif", "Stories"]:
        subset = [row for row in summary if row.get("dataset") == dataset]
        pieces = []
        for row in subset:
            pieces.append(
                f"{row.get('sentiment')} {fmt(row.get('percentage'))}% (n={int(float(row.get('count', 0)))})"
            )
        lines.append(f"- {dataset}: " + ", ".join(pieces))
    return lines


def summarize_grouped_eda() -> list[str]:
    summary = load_csv(OUTPUT_DIR / "grouped_eda" / "group_summary.csv")
    if not summary:
        return ["- Grouped emotion EDA is not available yet."]

    summary = sorted(summary, key=lambda row: float(row.get("prevalence", 0.0)), reverse=True)
    lines = []
    for row in summary:
        lines.append(
            f"- {row.get('group_label')}: {fmt(float(row.get('prevalence', 0.0)) * 100)}% "
            f"prevalence; members: {row.get('member_emotions')}"
        )
    return lines


def summarize_optuna() -> list[str]:
    text = read_text(OUTPUT_DIR / "bert_emotion_model_optuna" / "comparisons" / "comparison_summary.txt")
    if text is None:
        return ["- Optuna comparison is optional and not available yet."]
    return [f"- {line}" for line in text.splitlines() if line and not set(line) <= {"="}]


def summarize_threshold_cardinality() -> list[str]:
    baseline = load_json(OUTPUT_DIR / "bert_emotion_model" / "metrics.json")
    adjusted = load_json(OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json")
    if baseline is None or adjusted is None:
        return ["- Threshold/cardinality test is optional and not available yet."]

    def label_count(value: str) -> int:
        cleaned = str(value).strip()
        if not cleaned or cleaned.lower() == "none":
            return 0
        return len([part.strip() for part in cleaned.split(",") if part.strip()])

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
    baseline_sentiment = baseline["test_sentiment_metrics"]
    adjusted_emotion = adjusted["test_metrics"]
    adjusted_sentiment = adjusted["test_sentiment_metrics"]
    return [
        "- Baseline vs threshold/cardinality-tested BERT",
        (
            f"- Test micro F1: baseline {float(baseline_emotion['micro']['f1']):.4f} "
            f"vs threshold/cardinality {float(adjusted_emotion['micro']['f1']):.4f}"
        ),
        (
            f"- Test macro F1: baseline {float(baseline_emotion['macro']['f1']):.4f} "
            f"vs threshold/cardinality {float(adjusted_emotion['macro']['f1']):.4f}"
        ),
        (
            f"- Test sentiment accuracy: baseline {float(baseline_sentiment['accuracy']):.4f} "
            f"vs threshold/cardinality {float(adjusted_sentiment['accuracy']):.4f}"
        ),
        (
            f"- Average predicted labels: baseline "
            f"{float(baseline_emotion['avg_predicted_labels_per_entry']):.4f} "
            f"vs threshold/cardinality "
            f"{float(adjusted_emotion['avg_predicted_labels_per_entry']):.4f}"
        ),
        (
            f"- Rows with zero predicted emotions: baseline {baseline_zero_rows} "
            f"vs threshold/cardinality {int(adjusted_emotion['zero_predicted_rows'])}"
        ),
        (
            f"- Rows with at least two extra predicted emotions: baseline "
            f"{baseline_overpredicted_by_at_least_2} "
            f"vs threshold/cardinality "
            f"{int(adjusted_emotion['overpredicted_by_at_least_2_rows'])}"
        ),
        (
            f"- Inference hyperparameters: threshold "
            f"{float(adjusted['tested_threshold']):.2f}; "
            f"minimum predicted labels {int(adjusted['minimum_predicted_labels'])}"
        ),
    ]


def summarize_gemma() -> list[str]:
    def parse_summary(text: str) -> str:
        section = ""
        values: dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line == "Emotion metrics":
                section = "emotion"
                continue
            if line == "Sentiment metrics" or line == "Stories sentiment metrics":
                section = "sentiment"
                continue
            if line.startswith("Average latency"):
                values["latency"] = line
            elif section == "emotion" and line.startswith("Micro F1:"):
                values["emotion_micro_f1"] = line.replace("Micro F1:", "emotion micro-F1:")
            elif section == "emotion" and line.startswith("Macro F1:"):
                values["emotion_macro_f1"] = line.replace("Macro F1:", "emotion macro-F1:")
            elif section == "sentiment" and line.startswith("Accuracy:"):
                values["sentiment_accuracy"] = line.replace("Accuracy:", "sentiment accuracy:")
            elif section == "sentiment" and line.startswith("Macro F1:"):
                values["sentiment_macro_f1"] = line.replace("Macro F1:", "sentiment macro-F1:")

        ordered_keys = [
            "latency",
            "emotion_micro_f1",
            "emotion_macro_f1",
            "sentiment_accuracy",
            "sentiment_macro_f1",
        ]
        return "; ".join(values[key] for key in ordered_keys if key in values)

    lines: list[str] = []
    emotion_zero = read_text(OUTPUT_DIR / "gemma_2b_emotion_model" / "zero_shot" / "run_summary.txt")
    emotion_finetuned = read_text(OUTPUT_DIR / "gemma_2b_emotion_model" / "finetuned" / "run_summary.txt")
    stories_sentiment = read_text(
        TEST_OUTPUT_DIR / "gemma_sentiment" / "google_gemma-3n-E2B-it" / "run_summary.txt"
    )

    for label, text in [
        ("Gemma zero-shot emotion on Lemotif", emotion_zero),
        ("Gemma fine-tuned emotion on Lemotif", emotion_finetuned),
        ("Gemma zero-shot sentiment on Stories", stories_sentiment),
    ]:
        if text is None:
            lines.append(f"- {label}: not available.")
            continue
        lines.append(f"- {label}: {parse_summary(text)}")
    return lines


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


def evidence_table_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    sentiment = load_csv(OUTPUT_DIR / "dataset_comparison" / "sentiment_distribution_summary.csv")
    if sentiment is not None:
        for row in sentiment:
            rows.append(
                {
                    "section": "dataset_sentiment_shift",
                    "name": f"{row.get('dataset')}_{row.get('sentiment')}",
                    "value": row.get("percentage"),
                    "count": row.get("count"),
                    "source": rel(OUTPUT_DIR / "dataset_comparison" / "sentiment_distribution_summary.csv"),
                }
            )

    grouped = load_csv(OUTPUT_DIR / "grouped_eda" / "group_summary.csv")
    if grouped is not None:
        for row in grouped:
            rows.append(
                {
                    "section": "grouped_emotion_prevalence",
                    "name": row.get("group_label"),
                    "value": row.get("prevalence"),
                    "count": row.get("positive_rows"),
                    "source": rel(OUTPUT_DIR / "grouped_eda" / "group_summary.csv"),
                }
            )

    threshold_metrics = load_json(
        OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json"
    )
    if threshold_metrics is not None:
        emotion = threshold_metrics["test_metrics"]
        sentiment = threshold_metrics["test_sentiment_metrics"]
        rows.extend(
            [
                {
                    "section": "threshold_cardinality_test",
                    "name": "emotion_micro_f1",
                    "value": emotion["micro"]["f1"],
                    "count": "",
                    "source": rel(
                        OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json"
                    ),
                },
                {
                    "section": "threshold_cardinality_test",
                    "name": "emotion_macro_f1",
                    "value": emotion["macro"]["f1"],
                    "count": "",
                    "source": rel(
                        OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json"
                    ),
                },
                {
                    "section": "threshold_cardinality_test",
                    "name": "sentiment_accuracy",
                    "value": sentiment["accuracy"],
                    "count": "",
                    "source": rel(
                        OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json"
                    ),
                },
                {
                    "section": "threshold_cardinality_test",
                    "name": "overpredicted_by_at_least_2_rows",
                    "value": emotion["overpredicted_by_at_least_2_rows"],
                    "count": emotion["overpredicted_by_at_least_2_rows"],
                    "source": rel(
                        OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json"
                    ),
                },
            ]
        )

    return rows


def artifact_rows() -> list[dict[str, object]]:
    artifacts = [
        (
            "Dataset description",
            "Required",
            OUTPUT_DIR / "fig1_word_count_distribution.png",
            "Lemotif reflection length distribution for Methods/Data.",
        ),
        (
            "Dataset description",
            "Required",
            OUTPUT_DIR / "fig2_emotion_prevalence.png",
            "Main Lemotif label imbalance figure.",
        ),
        (
            "Dataset description",
            "Recommended",
            OUTPUT_DIR / "fig5_topic_emotion.png",
            "Topic-emotion association heatmap for exploratory context.",
        ),
        (
            "Dataset structure",
            "Optional",
            OUTPUT_DIR / "fig9_emotion_cooccurrence_network.png",
            "Optional network view if the results chapter has room.",
        ),
        (
            "Grouped emotion analysis",
            "Recommended",
            OUTPUT_DIR / "grouped_eda" / "fig1_group_prevalence_and_density.png",
            "Useful simplification of the 18-label taxonomy.",
        ),
        (
            "Grouped emotion analysis",
            "Optional",
            OUTPUT_DIR / "grouped_eda" / "fig4_group_feature_importance.png",
            "Permutation-style grouped feature importance; do not label this as SHAP.",
        ),
        (
            "Model variant",
            "Optional",
            OUTPUT_DIR
            / "bert_emotion_model_optuna"
            / "comparisons"
            / "fig_compare_overall_metrics.png",
            "Optional hyperparameter tuning comparison.",
        ),
        (
            "Model variant",
            "Optional",
            OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "metrics.json",
            "Optional threshold/cardinality hyperparameter test for reducing overprediction.",
        ),
        (
            "Model variant",
            "Optional",
            OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "run_summary.txt",
            "Earlier adjusted BERT inference run summary; no retrained weights.",
        ),
        (
            "Model variant",
            "Optional",
            OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "test_predictions_threshold_0_59_min1.csv",
            "Adjusted predictions with reduced emotion overestimation and no zero-emotion predictions.",
        ),
        (
            "Model variant",
            "Optional",
            OUTPUT_DIR / "bert_emotion_model" / "threshold_cardinality_test" / "threshold_sweep.csv",
            "Threshold sweep for the adjusted cardinality setting.",
        ),
        (
            "Model variant",
            "Recommended",
            ADJUSTED_BERT_SUMMARY_PATH,
            "Standalone thesis-facing summary of the adjusted BERT inference run.",
        ),
        (
            "Model variant",
            "Optional",
            OUTPUT_DIR / "gemma_2b_emotion_model" / "comparison" / "fig_compare_runs.png",
            "Optional BERT vs Gemma emotion-model comparison on Lemotif.",
        ),
        (
            "Model variant",
            "Optional",
            TEST_OUTPUT_DIR
            / "gemma_sentiment"
            / "comparisons"
            / "fig_compare_overall_sentiment_metrics.png",
            "Optional Stories sentiment comparison between Gemma and BERT.",
        ),
        (
            "WordPress validation",
            "Missing",
            OUTPUT_DIR / "wordpress" / "fig_wordpress_model_sanity_check.png",
            "Placeholder: public-blog/WordPress validation is not currently implemented.",
        ),
    ]
    rows = [artifact_row(section, priority, path, suggested_use) for section, priority, path, suggested_use in artifacts]
    rows.extend(test_output_inventory_rows({str(row["path"]) for row in rows}))
    return rows


def build_markdown(artifact_index: list[dict[str, object]]) -> str:
    cleaning_summary = read_text(OUTPUT_DIR / "cleaning_summary.txt")
    grouping_summary = read_text(OUTPUT_DIR / "grouped_eda" / "grouping_summary.txt")

    lines: list[str] = [
        "# Secondary Findings Evidence Summary",
        "",
        "This file gathers supporting evidence for the thesis background, Methods, and optional Results subsections.",
        "",
        "## Dataset And Label Structure",
        "",
    ]

    if cleaning_summary:
        lines.extend(["```text", cleaning_summary, "```", ""])
    else:
        lines.append("- Cleaning summary is not available yet.")

    lines.extend(
        [
            "## Dataset Sentiment Shift",
            "",
            *summarize_sentiment_shift(),
            "",
            "Use this to explain why Stories is a harder external test: the current Stories artifact is less positive and more neutral/negative than Lemotif.",
            "",
            "## Grouped Emotion Findings",
            "",
        ]
    )

    if grouping_summary:
        lines.extend(["```text", grouping_summary, "```", ""])
    lines.extend(summarize_grouped_eda())

    lines.extend(
        [
            "",
            "Use grouped emotions as an interpretability bridge, not as a replacement for the 18-label headline results.",
            "",
            "## Optional Model-Variant Findings",
            "",
            "### Optuna BERT",
            "",
            *summarize_optuna(),
            "",
            "### Threshold / Cardinality BERT",
            "",
            *summarize_threshold_cardinality(),
            "",
            "### Gemma",
            "",
            *summarize_gemma(),
            "",
            "Use model variants as secondary findings unless your thesis committee expects a direct BERT-vs-Gemma comparison.",
            "",
            "## Missing Secondary Claims",
            "",
            "- WordPress/public-blog generator and analysis are not present in tracked source files.",
            "- The available feature-importance output is grouped permutation-style analysis, not SHAP.",
            "- Longitudinal/user-history modeling is not implemented in the visible codebase.",
            "- Weak-label generation for Stories is not implemented as a reproducible script here.",
            "",
            "## Visuals To Consider",
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
            f"Machine-readable evidence table: `{rel(SECONDARY_TABLE_PATH)}`",
            f"Artifact index: `{rel(SECONDARY_ARTIFACT_INDEX_PATH)}`",
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
        "Secondary placement: use this as a model-variant finding about reducing emotion overestimation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

    evidence = evidence_table_rows()
    write_csv(
        SECONDARY_TABLE_PATH,
        evidence,
        fieldnames=["section", "name", "value", "count", "source"],
    )

    ADJUSTED_BERT_SUMMARY_PATH.write_text(
        build_adjusted_bert_summary_markdown(),
        encoding="utf-8",
    )

    artifact_index = artifact_rows()
    write_csv(
        SECONDARY_ARTIFACT_INDEX_PATH,
        artifact_index,
        fieldnames=["section", "priority", "path", "exists", "suggested_use"],
    )

    SECONDARY_MARKDOWN_PATH.write_text(
        build_markdown(artifact_index),
        encoding="utf-8",
    )

    print(f"Saved secondary findings markdown to: {SECONDARY_MARKDOWN_PATH}")
    print(f"Saved secondary evidence table to: {SECONDARY_TABLE_PATH}")
    print(f"Saved secondary artifact index to: {SECONDARY_ARTIFACT_INDEX_PATH}")
    print(f"Saved adjusted BERT summary to: {ADJUSTED_BERT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
