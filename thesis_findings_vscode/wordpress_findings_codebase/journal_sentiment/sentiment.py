from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from journal_sentiment.multilabel import BertMultiLabelAnalyzer, MultiLabelResult


@dataclass
class SentimentResult:
    label: str
    score: float
    predicted_emotions: list[str] | None = None
    threshold: float | None = None


SENTIMENT_VALENCE = {
    "afraid": -1.0,
    "angry": -1.0,
    "anxious": -1.0,
    "ashamed": -1.0,
    "awkward": -1.0,
    "bored": -1.0,
    "calm": 1.0,
    "confused": 0.0,
    "disgusted": -1.0,
    "excited": 1.0,
    "frustrated": -1.0,
    "happy": 1.0,
    "jealous": -1.0,
    "nostalgic": 0.0,
    "proud": 1.0,
    "sad": -1.0,
    "satisfied": 1.0,
    "surprised": 0.0,
}


def derive_sentiment(binary_labels: list[int], label_names: list[str]) -> tuple[str, float]:
    active_scores = [
        SENTIMENT_VALENCE[label_name.lower()]
        for label_name, is_active in zip(label_names, binary_labels)
        if int(is_active) == 1
    ]
    if not active_scores:
        return "neutral", 0.0

    score = float(sum(active_scores) / len(active_scores))
    if score > 0.25:
        return "positive", score
    if score < -0.25:
        return "negative", score
    return "neutral", score


class BertSentimentAnalyzer:
    def __init__(self, model_path: Path, fallback_labels: tuple[str, ...], cache_dir: Path | None = None):
        self.label_analyzer = BertMultiLabelAnalyzer(
            model_path=model_path,
            fallback_labels=fallback_labels,
            cache_dir=cache_dir,
        )
        self.label_names = self.label_analyzer.label_names
        self.threshold = self.label_analyzer.threshold

    def _result_from_multi_label_probs(self, probs: list[float]) -> SentimentResult:
        binary_predictions = [1 if prob >= self.threshold else 0 for prob in probs]
        predicted_emotions = [
            label_name
            for label_name, is_active in zip(self.label_names, binary_predictions)
            if is_active == 1
        ]
        sentiment_label, sentiment_score = derive_sentiment(binary_predictions, self.label_names)
        return SentimentResult(
            label=sentiment_label,
            score=sentiment_score,
            predicted_emotions=predicted_emotions,
            threshold=self.threshold,
        )

    def _result_from_multilabel_result(self, result: MultiLabelResult) -> SentimentResult:
        binary_predictions = [result.binary_predictions[label_name] for label_name in self.label_names]
        sentiment_label, sentiment_score = derive_sentiment(binary_predictions, self.label_names)
        return SentimentResult(
            label=sentiment_label,
            score=sentiment_score,
            predicted_emotions=result.active_labels,
            threshold=result.threshold,
        )

    def predict_many(self, texts: list[str]) -> list[SentimentResult]:
        return [self._result_from_multilabel_result(result) for result in self.label_analyzer.predict_many(texts)]

    def predict(self, text: str) -> SentimentResult:
        return self.predict_many([text])[0]
