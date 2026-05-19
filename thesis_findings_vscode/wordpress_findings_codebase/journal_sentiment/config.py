from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    model_path: Path
    output_dir: Path
    huggingface_cache_dir: Path
    request_timeout: int = 20
    max_text_chars: int = 4000
    labels: tuple[str, ...] = ("negative", "neutral", "positive")


def load_settings() -> Settings:
    load_dotenv()
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    default_model_candidates = [
        workspace_root / "primary_findings_codebase" / "output" / "bert_emotion_model" / "model",
        workspace_root / "model_rebuild_codebase" / "output" / "bert_emotion_model" / "model",
        workspace_root.parent / "output" / "bert_emotion_model" / "model",
    ]
    configured_model_path = os.getenv("MODEL_PATH", "")
    if configured_model_path:
        model_path = Path(configured_model_path).expanduser()
    else:
        model_path = next(
            (candidate for candidate in default_model_candidates if candidate.exists()),
            default_model_candidates[0],
        )
    output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
    huggingface_cache_dir = Path(os.getenv("HF_CACHE_DIR", ".hf-cache"))
    request_timeout = int(os.getenv("REQUEST_TIMEOUT", "20"))
    max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "4000"))
    labels = tuple(
        label.strip() for label in os.getenv("LABELS", "negative,neutral,positive").split(",") if label.strip()
    )
    return Settings(
        model_path=model_path,
        output_dir=output_dir,
        huggingface_cache_dir=huggingface_cache_dir,
        request_timeout=request_timeout,
        max_text_chars=max_text_chars,
        labels=labels,
    )
