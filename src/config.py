from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
GROUPED_EDA_OUTPUT_DIR = OUTPUT_DIR / "grouped_eda"

RAW_CSV_PATH = DATA_DIR / "Lemotif thesis data.csv"
CSV_PATH = RAW_CSV_PATH
CLEANED_CSV_PATH = OUTPUT_DIR / "lemotif_cleaned.csv"
CLEANING_SUMMARY_PATH = OUTPUT_DIR / "cleaning_summary.txt"
BERT_OUTPUT_DIR = OUTPUT_DIR / "bert_emotion_model"
MODEL_SCRIPT = "src/train_bert.py"

EDA_SCRIPTS = [
    "src/eda/01_text_length_distribution.py",
    "src/eda/02_emotion_prevalence.py",
    "src/eda/03_topic_prevalence.py",
    "src/eda/04_emotion_correlation_matrix.py",
    "src/eda/05_topic_emotion_heatmap.py",
    "src/eda/06_emotion_labels_per_entry.py",
    "src/eda/07_topic_labels_per_entry.py",
    "src/eda/08_representative_entries.py",
    "src/eda/09_emotion_cooccurrence_network.py",
    "src/eda/10_length_vs_emotion_probability.py",
    "src/eda/11_topic_to_emotion_probability.py",
]

GROUPED_EDA_SCRIPTS = [
    "src/grouped_eda/00_group_definition_summary.py",
    "src/grouped_eda/01_group_prevalence.py",
    "src/grouped_eda/02_group_cooccurrence.py",
    "src/grouped_eda/03_topic_group_associations.py",
    "src/grouped_eda/04_group_feature_importance.py",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUPED_EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
