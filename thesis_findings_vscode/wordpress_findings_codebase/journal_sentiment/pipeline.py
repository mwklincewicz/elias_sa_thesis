from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from journal_sentiment.discovery import DEFAULT_TAGS, discover_wordpress_sites
from journal_sentiment.scraper import JournalPost, fetch_site_posts
from journal_sentiment.sentiment import BertSentimentAnalyzer


def _persist_rows(output_dir: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "journal_sentiment_results.csv"
    json_path = output_dir / "journal_sentiment_results.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return frame


def run_pipeline(
    analyzer: BertSentimentAnalyzer,
    output_dir: Path,
    timeout: int,
    max_text_chars: int,
    limit_sites: int,
    posts_per_site: int,
    days_back: int = 365,
    target_posts: int | None = None,
    target_sites: int | None = None,
    inference_batch_size: int = 8,
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    sites = discover_wordpress_sites(tags=tags, timeout=timeout)[:limit_sites]
    rows: list[dict[str, object]] = []
    successful_posts = 0
    successful_sites = 0
    for index, site in enumerate(sites, start=1):
        if target_posts is not None and successful_posts >= target_posts:
            break
        if target_sites is not None and successful_sites >= target_sites:
            break

        remaining_target = target_posts - successful_posts if target_posts is not None else posts_per_site
        site_limit = min(posts_per_site, remaining_target)
        if site_limit <= 0:
            break

        try:
            posts: list[JournalPost] = fetch_site_posts(
                site_url=site,
                timeout=timeout,
                max_chars=max_text_chars,
                limit=site_limit,
                tags=tags,
                cutoff_date=cutoff_date,
            )
        except Exception as exc:
            rows.append(
                {
                    "site": site,
                    "post_url": "",
                    "title": "",
                    "published": "",
                    "text": "",
                    "sentiment_label": "FETCH_ERROR",
                    "sentiment_score": 0.0,
                    "predicted_emotions": "",
                    "predicted_emotion_count": 0,
                    "site_domain": urlparse(site).netloc,
                    "text_length": 0,
                    "days_back_limit": days_back,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(sites)}] {site} -> fetch error")
            _persist_rows(output_dir, rows)
            continue

        site_had_success = False
        valid_posts = [post for post in posts if post.text.strip()]
        for batch_start in range(0, len(valid_posts), inference_batch_size):
            batch_posts = valid_posts[batch_start : batch_start + inference_batch_size]
            if hasattr(analyzer, "predict_many"):
                batch_results = analyzer.predict_many([post.text for post in batch_posts])
            else:
                batch_results = [analyzer.predict(post.text) for post in batch_posts]
            for post, result in zip(batch_posts, batch_results):
                rows.append(
                    {
                        "site": post.site,
                        "post_url": post.post_url,
                        "title": post.title,
                        "published": post.published,
                        "text": post.text,
                        "sentiment_label": result.label,
                        "sentiment_score": result.score,
                        "predicted_emotions": "|".join(result.predicted_emotions or []),
                        "predicted_emotion_count": len(result.predicted_emotions or []),
                        "emotion_threshold": result.threshold,
                        "site_domain": urlparse(post.site).netloc,
                        "text_length": len(post.text),
                        "days_back_limit": days_back,
                        "error": "",
                    }
                )
                successful_posts += 1
                site_had_success = True
                if target_posts is not None and successful_posts >= target_posts:
                    break
            if target_posts is not None and successful_posts >= target_posts:
                break
        if site_had_success:
            successful_sites += 1
            print(
                f"[{index}/{len(sites)}] {site} -> analyzed {len(valid_posts)} posts "
                f"(successful sites: {successful_sites}, successful posts: {successful_posts})"
            )
        else:
            print(f"[{index}/{len(sites)}] {site} -> no valid posts in date window")
        _persist_rows(output_dir, rows)

    return _persist_rows(output_dir, rows)
