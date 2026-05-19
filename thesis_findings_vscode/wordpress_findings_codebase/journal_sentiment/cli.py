from __future__ import annotations

import argparse

from journal_sentiment.config import load_settings
from journal_sentiment.pipeline import run_pipeline
from journal_sentiment.sentiment import BertSentimentAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover public journal blogs and run BERT sentiment analysis.")
    parser.add_argument("--limit-sites", type=int, default=5)
    parser.add_argument("--posts-per-site", type=int, default=3)
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--target-posts", type=int, default=None)
    parser.add_argument("--target-sites", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    analyzer = BertSentimentAnalyzer(
        model_path=settings.model_path,
        fallback_labels=settings.labels,
        cache_dir=settings.huggingface_cache_dir,
    )
    frame = run_pipeline(
        analyzer=analyzer,
        output_dir=settings.output_dir,
        timeout=settings.request_timeout,
        max_text_chars=settings.max_text_chars,
        limit_sites=args.limit_sites,
        posts_per_site=args.posts_per_site,
        days_back=args.days_back,
        target_posts=args.target_posts,
        target_sites=args.target_sites,
    )
    print(f"Saved {len(frame)} results to {settings.output_dir}")


if __name__ == "__main__":
    main()
