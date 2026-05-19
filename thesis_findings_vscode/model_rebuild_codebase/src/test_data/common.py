from __future__ import annotations

import re
import os
import sys
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import detect_columns, display_name, load_lemotif_data

DEFAULT_SOURCE_XLSX_PATH = PROJECT_ROOT / "data" / "stories_source.xlsx"
SOURCE_XLSX_PATH = Path(
    os.getenv("STORIES_SOURCE_XLSX_PATH", str(DEFAULT_SOURCE_XLSX_PATH))
)
TEST_DATA_DIR = PROJECT_ROOT / "test data"
STORIES_CSV_PATH = TEST_DATA_DIR / "stories_data.csv"
FORMATTING_SUMMARY_PATH = TEST_DATA_DIR / "stories_formatting_summary.txt"
TEST_DATA_OUTPUT_DIR = TEST_DATA_DIR / "output"
STORIES_EDA_OUTPUT_DIR = TEST_DATA_OUTPUT_DIR / "eda"
EXTERNAL_EVAL_DIR = TEST_DATA_OUTPUT_DIR / "external_evaluation"
GEMMA_SENTIMENT_DIR = TEST_DATA_OUTPUT_DIR / "gemma_sentiment"

TEMPLATE_CSV_PATH = PROJECT_ROOT / "data" / "Lemotif thesis data.csv"
TAIL_PATTERN = re.compile(r"([A-Z,; ]+)$")
EXCEL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

LABEL_NORMALIZATION = {
    "RECREATIONAL": "RECREATION",
    "FRUSTATED": "FRUSTRATED",
    "FRUSTARTED": "FRUSTRATED",
    "FRUSTARTION": "FRUSTRATED",
    "HAPPPY": "HAPPY",
    "SATISAFIED": "SATISFIED",
    "SATISIFED": "SATISFIED",
    "SATSIFIED": "SATISFIED",
    "SUPRISED": "SURPRISED",
    "SCHOO": "SCHOOL",
    "SCHOOLL": "SCHOOL",
    "HEA;TH": "HEALTH",
}

TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORIES_EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
GEMMA_SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)


def load_shared_strings_from_xlsx(xlsx_path: str | Path) -> list[str]:
    with ZipFile(xlsx_path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in string_item.iter(f"{EXCEL_NS}t"))
            for string_item in shared_root.findall(f"{EXCEL_NS}si")
        ]

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    values = []
    for row in sheet_root.findall(f"{EXCEL_NS}sheetData/{EXCEL_NS}row"):
        cell = row.find(f"{EXCEL_NS}c")
        if cell is None:
            continue

        cell_type = cell.attrib.get("t")
        if cell_type == "s":
            value_node = cell.find(f"{EXCEL_NS}v")
            if value_node is None or value_node.text is None:
                values.append("")
            else:
                values.append(shared_strings[int(value_node.text)])
        elif cell_type == "inlineStr":
            values.append("".join(node.text or "" for node in cell.iter(f"{EXCEL_NS}t")))
        else:
            value_node = cell.find(f"{EXCEL_NS}v")
            values.append("" if value_node is None or value_node.text is None else value_node.text)

    return values


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _template_column_maps() -> tuple[list[str], dict[str, str], dict[str, str]]:
    template_header = pd.read_csv(TEMPLATE_CSV_PATH, nrows=0)
    _, emotion_cols, topic_cols = detect_columns(template_header)

    emotion_map = {display_name(col).upper(): col for col in emotion_cols}
    topic_map = {display_name(col).upper(): col for col in topic_cols}
    return template_header.columns.tolist(), emotion_map, topic_map


def normalize_label_tail(tail: str) -> str:
    normalized = tail.strip()
    for wrong, right in LABEL_NORMALIZATION.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def parse_story_entry(
    raw_entry: str,
    valid_topics: set[str],
    valid_emotions: set[str],
) -> tuple[str, list[str], list[str], str]:
    normalized_entry = " ".join(str(raw_entry).split())
    tail_match = TAIL_PATTERN.search(normalized_entry)
    if not tail_match:
        raise ValueError(f"Could not locate label suffix in row: {normalized_entry[:140]}")

    label_tail = normalize_label_tail(tail_match.group(1))
    text = normalized_entry[: tail_match.start()].strip(" \n\t,.;:-")
    tokens = [token for token in re.split(r"[ ,;]+", label_tail) if token]

    topic_tokens = []
    emotion_tokens = []
    unknown_tokens = []

    for token in tokens:
        if token in valid_topics and token not in topic_tokens:
            topic_tokens.append(token)
        elif token in valid_emotions and token not in emotion_tokens:
            emotion_tokens.append(token)
        else:
            unknown_tokens.append(token)

    if unknown_tokens:
        raise ValueError(
            f"Unknown label token(s) {unknown_tokens} in tail '{label_tail}' for text '{text[:120]}'"
        )
    if not emotion_tokens:
        raise ValueError(f"No emotion labels detected for row: {text[:140]}")

    return text, topic_tokens, emotion_tokens, label_tail


def build_stories_dataframe() -> tuple[pd.DataFrame, dict[str, object]]:
    template_columns, emotion_map, topic_map = _template_column_maps()
    valid_topics = set(topic_map)
    valid_emotions = set(emotion_map)
    raw_rows = load_shared_strings_from_xlsx(SOURCE_XLSX_PATH)

    records = []
    normalized_counter: Counter[str] = Counter()
    rows_without_topic = 0

    for row_index, raw_entry in enumerate(raw_rows, start=1):
        normalized_entry = " ".join(str(raw_entry).split())
        tail_match = TAIL_PATTERN.search(normalized_entry)
        if tail_match is None:
            raise ValueError(f"Could not locate label suffix in row {row_index}: {normalized_entry[:140]}")
        tail_before = tail_match.group(1).strip()
        tail_after = normalize_label_tail(tail_before)
        if tail_before != tail_after:
            for wrong, right in LABEL_NORMALIZATION.items():
                if wrong in tail_before:
                    normalized_counter[f"{wrong}->{right}"] += 1

        text, topic_tokens, emotion_tokens, _ = parse_story_entry(
            raw_entry=raw_entry,
            valid_topics=valid_topics,
            valid_emotions=valid_emotions,
        )

        if not topic_tokens:
            rows_without_topic += 1

        record = {column: 0 for column in template_columns}
        record["Answer"] = text
        for topic in _dedupe_preserving_order(topic_tokens):
            record[topic_map[topic]] = 1
        for emotion in _dedupe_preserving_order(emotion_tokens):
            record[emotion_map[emotion]] = 1
        records.append(record)

    stories_df = pd.DataFrame(records, columns=template_columns)
    summary = {
        "source_xlsx": str(SOURCE_XLSX_PATH),
        "stories_csv": str(STORIES_CSV_PATH),
        "row_count": len(stories_df),
        "rows_without_topic": rows_without_topic,
        "normalizations": dict(normalized_counter),
    }
    return stories_df, summary


def build_formatting_summary_text(summary: dict[str, object]) -> str:
    lines = [
        "Stories evaluation dataset formatting summary",
        "===============================",
        f"Source workbook: {summary['source_xlsx']}",
        f"Output CSV: {summary['stories_csv']}",
        "",
        f"Rows formatted: {summary['row_count']}",
        f"Rows without topic labels: {summary['rows_without_topic']}",
        "",
        "Applied label normalizations:",
    ]

    normalizations: dict[str, int] = summary["normalizations"]  # type: ignore[assignment]
    if normalizations:
        for key, count in sorted(normalizations.items()):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Output CSV matches the Lemotif column structure: Answer + 18 emotion columns + 11 topic columns.")
    return "\n".join(lines) + "\n"


def load_stories_data() -> tuple[pd.DataFrame, str, list[str], list[str]]:
    return load_lemotif_data(STORIES_CSV_PATH)


def shorten(text: str, width: int = 80) -> str:
    return textwrap.fill(textwrap.shorten(" ".join(str(text).split()), width=width, placeholder="..."), width=width)
