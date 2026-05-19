from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SITE_LIST_PATH = PROJECT_ROOT / "data" / "selected_sites.txt"
RESULTS_PATH = PROJECT_ROOT / "output" / "selected_site_analysis" / "public_reflection_predictions.csv"
RUN_SUMMARY_PATH = PROJECT_ROOT / "output" / "selected_site_analysis" / "run_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "selected_site_analysis" / "thesis_input_tables"

INPUT_DATASET_PATH = OUTPUT_DIR / "public_reflection_input_table.csv"
SITE_OVERVIEW_PATH = OUTPUT_DIR / "public_journal_site_overview.csv"
EXPERIMENT_METADATA_PATH = OUTPUT_DIR / "public_journal_experiment_metadata.csv"


def load_site_order() -> dict[str, int]:
    sites = [line.strip() for line in SITE_LIST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {site: index for index, site in enumerate(sites, start=1)}


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_PATH)
    frame = frame[frame["error"].fillna("") == ""].copy()
    frame["published_datetime_utc"] = pd.to_datetime(frame["published"], utc=True, errors="coerce")
    frame["publication_date"] = frame["published_datetime_utc"].dt.date.astype("string")
    frame["published_datetime_utc"] = frame["published_datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame


def build_input_dataset(frame: pd.DataFrame, site_order: dict[str, int]) -> pd.DataFrame:
    dataset = frame[
        [
            "site",
            "site_domain",
            "post_url",
            "title",
            "published",
            "published_datetime_utc",
            "publication_date",
            "terms",
            "text_length",
            "days_back_limit",
            "text",
        ]
    ].copy()
    dataset.insert(0, "reflection_id", range(1, len(dataset) + 1))
    dataset.insert(1, "site_selection_order", dataset["site"].map(site_order))
    dataset = dataset.rename(
        columns={
            "site": "site_url",
            "title": "post_title",
            "published": "published_original",
            "terms": "wordpress_terms",
            "text_length": "text_length_chars",
            "days_back_limit": "collection_window_days",
            "text": "reflection_text",
        }
    )
    dataset = dataset.sort_values(["site_selection_order", "published_datetime_utc", "reflection_id"], ascending=[True, False, True])
    return dataset


def build_site_overview(frame: pd.DataFrame, site_order: dict[str, int]) -> pd.DataFrame:
    overview = (
        frame.groupby(["site", "site_domain"], as_index=False)
        .agg(
            reflections_used=("post_url", "count"),
            earliest_publication_utc=("published_datetime_utc", "min"),
            latest_publication_utc=("published_datetime_utc", "max"),
            avg_text_length_chars=("text_length", "mean"),
            median_text_length_chars=("text_length", "median"),
        )
        .rename(columns={"site": "site_url"})
    )
    overview.insert(0, "site_selection_order", overview["site_url"].map(site_order))
    overview["avg_text_length_chars"] = overview["avg_text_length_chars"].round(1)
    overview["median_text_length_chars"] = overview["median_text_length_chars"].round(1)
    overview = overview.sort_values("site_selection_order")
    return overview


def build_experiment_metadata(site_order: dict[str, int]) -> pd.DataFrame:
    summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    metadata = pd.DataFrame(
        [
            {
                "dataset_name": "Public journals selected-site experiment",
                "site_list_file": str(SITE_LIST_PATH),
                "site_count_requested": summary["site_count_requested"],
                "site_count_successful": summary["site_count_successful"],
                "reflection_count_successful": summary["reflection_count_successful"],
                "collection_window_days": summary["days_back_limit"],
                "posts_per_site_limit": summary["posts_per_site_limit"],
                "emotion_model_path": summary["emotion_model_path"],
                "theme_model_path": summary["theme_model_path"],
                "site_selection_count": len(site_order),
            }
        ]
    )
    return metadata


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    site_order = load_site_order()
    frame = load_results()

    input_dataset = build_input_dataset(frame, site_order)
    site_overview = build_site_overview(frame, site_order)
    experiment_metadata = build_experiment_metadata(site_order)

    input_dataset.to_csv(INPUT_DATASET_PATH, index=False, encoding="utf-8-sig")
    site_overview.to_csv(SITE_OVERVIEW_PATH, index=False, encoding="utf-8-sig")
    experiment_metadata.to_csv(EXPERIMENT_METADATA_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved input dataset to: {INPUT_DATASET_PATH}")
    print(f"Saved site overview to: {SITE_OVERVIEW_PATH}")
    print(f"Saved experiment metadata to: {EXPERIMENT_METADATA_PATH}")


if __name__ == "__main__":
    main()
