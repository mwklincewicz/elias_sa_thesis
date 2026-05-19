import os
from pathlib import Path
import sys

import seaborn as sns

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from data_loader import display_name, load_analysis_data, load_lemotif_data

sns.set_theme(style="whitegrid")

OUTPUT_DIR = Path(os.getenv("LEMOTIF_ANALYSIS_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    custom_dataset = os.getenv("LEMOTIF_ANALYSIS_DATASET_PATH")
    if custom_dataset:
        return load_lemotif_data(Path(custom_dataset))
    return load_analysis_data(prefer_cleaned=True)
