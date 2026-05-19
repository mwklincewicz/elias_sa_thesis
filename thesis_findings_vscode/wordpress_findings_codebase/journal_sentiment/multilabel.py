from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))


@dataclass
class MultiLabelResult:
    active_labels: list[str]
    threshold: float
    probabilities: dict[str, float]
    binary_predictions: dict[str, int]


class BertMultiLabelAnalyzer:
    def __init__(self, model_path: Path, fallback_labels: tuple[str, ...] = (), cache_dir: Path | None = None):
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_dir))

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        raw_id2label = self.model.config.id2label or {idx: label for idx, label in enumerate(fallback_labels)}
        self.id2label = {int(idx): label for idx, label in raw_id2label.items()}
        self.label_names = [self.id2label[idx] for idx in sorted(self.id2label)]
        self.threshold = self._load_threshold(model_path)

    def _load_threshold(self, model_path: Path) -> float:
        metrics_path = model_path.parent / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            threshold = metrics.get("best_threshold")
            if threshold is not None:
                return float(threshold)
        return 0.5

    def predict_many(self, texts: list[str]) -> list[MultiLabelResult]:
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits

        if self.model.config.problem_type == "multi_label_classification":
            results = []
            for probs in torch.sigmoid(logits).tolist():
                probability_map = {
                    label_name: float(prob)
                    for label_name, prob in zip(self.label_names, probs)
                }
                binary_predictions = {
                    label_name: int(prob >= self.threshold)
                    for label_name, prob in probability_map.items()
                }
                active_labels = [label for label, is_active in binary_predictions.items() if is_active == 1]
                results.append(
                    MultiLabelResult(
                        active_labels=active_labels,
                        threshold=self.threshold,
                        probabilities=probability_map,
                        binary_predictions=binary_predictions,
                    )
                )
            return results

        results = []
        for probs in torch.softmax(logits, dim=-1).tolist():
            best_index = max(range(len(probs)), key=lambda idx: probs[idx])
            probability_map = {
                label_name: float(prob)
                for label_name, prob in zip(self.label_names, probs)
            }
            binary_predictions = {
                label_name: int(idx == best_index)
                for idx, label_name in enumerate(self.label_names)
            }
            results.append(
                MultiLabelResult(
                    active_labels=[self.label_names[best_index]],
                    threshold=self.threshold,
                    probabilities=probability_map,
                    binary_predictions=binary_predictions,
                )
            )
        return results

    def predict(self, text: str) -> MultiLabelResult:
        return self.predict_many([text])[0]
