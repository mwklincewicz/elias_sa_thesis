from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from journal_sentiment.multilabel import BertMultiLabelAnalyzer
from journal_sentiment.scraper import fetch_site_posts
from journal_sentiment.sentiment import derive_sentiment


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
SITE_LIST_PATH = PROJECT_ROOT / "data" / "selected_sites.txt"
OUTPUT_DIR = PROJECT_ROOT / "output" / "selected_site_analysis"
HF_CACHE_DIR = PROJECT_ROOT / ".hf-cache"
DEFAULT_TAGS = ("journal", "personal", "daily", "life", "mental-health", "diary", "travel", "thoughts")
MAX_TEXT_CHARS = 4000
REQUEST_TIMEOUT = 30
POSTS_PER_SITE_LIMIT = 365
DAYS_BACK = 365
INFERENCE_BATCH_SIZE = 8


def first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def resolve_emotion_model_path() -> Path:
    env_value = os.getenv("WORDPRESS_EMOTION_MODEL_PATH") or os.getenv("MODEL_PATH")
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            WORKSPACE_ROOT / "primary_findings_codebase" / "output" / "bert_emotion_model" / "model",
            WORKSPACE_ROOT / "model_rebuild_codebase" / "output" / "bert_emotion_model" / "model",
            WORKSPACE_ROOT.parent / "output" / "bert_emotion_model" / "model",
        ]
    )
    return first_existing_path(candidates, candidates[0])


def resolve_theme_model_path() -> Path:
    env_value = os.getenv("WORDPRESS_THEME_MODEL_PATH")
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(PROJECT_ROOT / "output" / "bert_theme_model" / "model")
    return first_existing_path(candidates, candidates[0])


def load_sites(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def persist_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    (OUTPUT_DIR / "public_reflection_predictions.csv").write_text(frame.to_csv(index=False), encoding="utf-8")
    (OUTPUT_DIR / "public_reflection_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = load_sites(SITE_LIST_PATH)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    emotion_model_path = resolve_emotion_model_path()
    theme_model_path = resolve_theme_model_path()

    emotion_analyzer = BertMultiLabelAnalyzer(
        model_path=emotion_model_path,
        cache_dir=HF_CACHE_DIR,
    )
    theme_analyzer = BertMultiLabelAnalyzer(
        model_path=theme_model_path,
        cache_dir=HF_CACHE_DIR,
    )

    emotion_label_names = emotion_analyzer.label_names
    theme_label_names = theme_analyzer.label_names
    rows: list[dict[str, object]] = []
    successful_sites = 0
    successful_posts = 0

    for index, site in enumerate(sites, start=1):
        try:
            posts = fetch_site_posts(
                site_url=site,
                timeout=REQUEST_TIMEOUT,
                max_chars=MAX_TEXT_CHARS,
                limit=POSTS_PER_SITE_LIMIT,
                tags=DEFAULT_TAGS,
                cutoff_date=cutoff_date,
            )
        except Exception as exc:
            rows.append(
                {
                    "site": site,
                    "site_domain": urlparse(site).netloc,
                    "post_url": "",
                    "title": "",
                    "published": "",
                    "text": "",
                    "sentiment_label": "FETCH_ERROR",
                    "sentiment_score": 0.0,
                    "predicted_emotions": "",
                    "predicted_themes": "",
                    "predicted_emotion_count": 0,
                    "predicted_theme_count": 0,
                    "emotion_threshold": emotion_analyzer.threshold,
                    "theme_threshold": theme_analyzer.threshold,
                    "text_length": 0,
                    "days_back_limit": DAYS_BACK,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(sites)}] {site} -> fetch error")
            persist_rows(rows)
            continue

        valid_posts = [post for post in posts if post.text.strip()]
        if not valid_posts:
            print(f"[{index}/{len(sites)}] {site} -> no valid posts")
            continue

        site_rows_before = len(rows)
        for batch_start in range(0, len(valid_posts), INFERENCE_BATCH_SIZE):
            batch_posts = valid_posts[batch_start : batch_start + INFERENCE_BATCH_SIZE]
            emotion_results = emotion_analyzer.predict_many([post.text for post in batch_posts])
            theme_results = theme_analyzer.predict_many([post.text for post in batch_posts])

            for post, emotion_result, theme_result in zip(batch_posts, emotion_results, theme_results):
                emotion_binary = [emotion_result.binary_predictions[label] for label in emotion_label_names]
                sentiment_label, sentiment_score = derive_sentiment(emotion_binary, emotion_label_names)

                row = {
                    "site": post.site,
                    "site_domain": urlparse(post.site).netloc,
                    "post_url": post.post_url,
                    "title": post.title,
                    "published": post.published,
                    "text": post.text,
                    "terms": "|".join(post.terms),
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "predicted_emotions": "|".join(emotion_result.active_labels),
                    "predicted_themes": "|".join(theme_result.active_labels),
                    "predicted_emotion_count": len(emotion_result.active_labels),
                    "predicted_theme_count": len(theme_result.active_labels),
                    "emotion_threshold": emotion_result.threshold,
                    "theme_threshold": theme_result.threshold,
                    "text_length": len(post.text),
                    "days_back_limit": DAYS_BACK,
                    "error": "",
                }
                for label_name in emotion_label_names:
                    safe_label = label_name.lower().replace(" ", "_")
                    row[f"emotion__{safe_label}"] = emotion_result.binary_predictions[label_name]
                    row[f"emotion_prob__{safe_label}"] = emotion_result.probabilities[label_name]
                for label_name in theme_label_names:
                    safe_label = label_name.lower().replace(" ", "_")
                    row[f"theme__{safe_label}"] = theme_result.binary_predictions[label_name]
                    row[f"theme_prob__{safe_label}"] = theme_result.probabilities[label_name]
                rows.append(row)
                successful_posts += 1

        site_post_count = len(rows) - site_rows_before
        if site_post_count > 0:
            successful_sites += 1
            print(
                f"[{index}/{len(sites)}] {site} -> analyzed {site_post_count} posts "
                f"(successful sites: {successful_sites}, successful posts: {successful_posts})"
            )
        persist_rows(rows)

    summary = {
        "site_count_requested": len(sites),
        "site_count_successful": successful_sites,
        "reflection_count_successful": successful_posts,
        "days_back_limit": DAYS_BACK,
        "posts_per_site_limit": POSTS_PER_SITE_LIMIT,
        "emotion_model_path": str(emotion_model_path),
        "theme_model_path": str(theme_model_path),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
