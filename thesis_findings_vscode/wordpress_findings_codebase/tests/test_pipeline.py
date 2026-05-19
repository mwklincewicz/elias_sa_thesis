from __future__ import annotations

import json
from pathlib import Path

from journal_sentiment.pipeline import run_pipeline
from journal_sentiment.sentiment import SentimentResult


class StubAnalyzer:
    def predict(self, text: str) -> SentimentResult:
        label = "positive" if "good" in text.lower() else "negative"
        return SentimentResult(label=label, score=0.99)


def test_pipeline_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "journal_sentiment.pipeline.discover_wordpress_sites",
        lambda tags, timeout: ["https://example.com"],
    )
    monkeypatch.setattr(
        "journal_sentiment.pipeline.fetch_site_posts",
        lambda site_url, timeout, max_chars, limit, tags, cutoff_date: [
            type(
                "Post",
                (),
                {
                    "site": site_url,
                    "post_url": "https://example.com/post",
                    "title": "A day",
                    "published": "2026-04-30",
                    "text": "Today felt good.",
                },
            )()
        ],
    )

    frame = run_pipeline(
        analyzer=StubAnalyzer(),
        output_dir=tmp_path,
        timeout=10,
        max_text_chars=500,
        limit_sites=1,
        posts_per_site=1,
        days_back=365,
        target_posts=1,
    )

    assert len(frame) == 1
    assert frame.iloc[0]["sentiment_label"] == "positive"
    assert frame.iloc[0]["predicted_emotion_count"] == 0
    assert (tmp_path / "journal_sentiment_results.csv").exists()
    payload = json.loads((tmp_path / "journal_sentiment_results.json").read_text(encoding="utf-8"))
    assert payload[0]["site"] == "https://example.com"
