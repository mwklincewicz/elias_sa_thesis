# Lemotif Thesis Code Repository Deliverable

This repository is the public Code Repository Deliverable for Elias Eichler's thesis project on multi-label emotion classification and derived sentiment analysis for reflective prediagnostic mental-health text.

Start here for review:

- Finalized Methodology Overview: [`docs/METHODOLOGY_OVERVIEW.md`](docs/METHODOLOGY_OVERVIEW.md)
- Data access and restricted-data notes: [`data/README.md`](data/README.md)
- Clean no-data repository checks: `make check`
- Full Lemotif cleaning and EDA rebuild after placing the approved dataset: `python src/run_pipeline.py`

The public repository intentionally excludes restricted Stories data, raw local workbooks, generated model checkpoints, and full scraped public-reflection text tables. The MO explains which parts are publicly reproducible from code and which require approved data/model access.

Submission note: after final review, mark the submission commit with a `final` tag and push it with `git push origin final`.

## Lemotif Cleaning + EDA Pipeline

This project is now organized around one clear thesis workflow:

1. Load the raw Lemotif CSV.
2. Clean and standardize the dataset.
3. Save a cleaned analysis file and cleaning summary.
4. Run the full EDA suite on the cleaned data.

The duplicate `src` setup has been consolidated into the root `src/` directory so the cleaning and EDA steps now live in one place.

## Project Structure

```text
lemotif_eda_vscode/
|-- data/
|   `-- Lemotif thesis data.csv
|-- output/
|   |-- lemotif_cleaned.csv
|   |-- cleaning_summary.txt
|   `-- fig*.png
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- clean_data.py
|   |-- train_bert.py
|   |-- grouped_eda/
|   |   |-- common.py
|   |   |-- run_grouped_eda.py
|   |   `-- 0*_*.py
|   |-- model_eda/
|   |   |-- common.py
|   |   |-- run_model_eda.py
|   |   `-- 0*_*.py
|   |-- test_data/
|   |   |-- common.py
|   |   |-- prepare_stories_data.py
|   |   |-- run_stories_eda.py
|   |   |-- train_full_lemotif_external.py
|   |   `-- run_all.py
|   |-- run_pipeline.py
|   |-- run_all.py
|   `-- eda/
|       |-- common.py
|       |-- 01_text_length_distribution.py
|       |-- 02_emotion_prevalence.py
|       |-- 03_topic_prevalence.py
|       |-- 04_emotion_correlation_matrix.py
|       |-- 05_topic_emotion_heatmap.py
|       |-- 06_emotion_labels_per_entry.py
|       |-- 07_topic_labels_per_entry.py
|       |-- 08_representative_entries.py
|       |-- 09_emotion_cooccurrence_network.py
|       |-- 10_length_vs_emotion_probability.py
|       `-- 11_topic_to_emotion_probability.py
`-- archive/
    `-- legacy_cleaning_and_extra_eda/
```

## Installation

```bash
python -m pip install -r requirements.txt
```

Run the lightweight Code Repository Deliverable checks from a clean checkout:

```bash
make check
```

These checks compile the public code, run standard-library repository contract tests, and verify that the CRD documentation links the methodology, data access notes, and reproducibility commands. They do not require restricted datasets, model downloads, or internet access.

If `make` is not available, run the same checks directly:

```bash
python -m compileall -q src thesis_findings_vscode/wordpress_findings_codebase
python -m unittest discover -s tests -p "test_*.py"
python scripts/check_crd_docs.py
```

## Public Code-Only Repository

This GitHub repository is intended to be shared as a public, reproducible code package. It does not publish restricted Stories evaluation data, raw local workbooks, generated model checkpoints, or full scraped WordPress post text.

For full local reproduction, place approved datasets in the documented `data/` folders:

- `Lemotif thesis data.csv`
- `stories_source.xlsx` for authorized reviewers only

All generated outputs under `output/` and `test data/output/` can be recreated from the scripts and runners described below. The supervisor-facing workspace is:

```text
thesis_findings_vscode/thesis_findings.code-workspace
```

## Full Pipeline

Run the complete workflow from raw data to final figures:

```bash
python src/run_pipeline.py
```

This will:

- clean `data/Lemotif thesis data.csv`
- write `output/lemotif_cleaned.csv`
- write `output/cleaning_summary.txt`
- generate all EDA figures in `output/`
- generate grouped-emotion EDA outputs in `output/grouped_eda/`

`python src/run_all.py` still works as a compatibility alias for the same full pipeline.

If you also want to run the local BERT baseline after the EDA steps:

```bash
$env:LEMOTIF_RUN_MODEL="1"
python src/run_pipeline.py
```

This appends the single-entry BERT training stage to the existing workflow.

## Run Individual Steps

Run only the cleaning step:

```bash
python src/clean_data.py
```

Run one EDA script:

```bash
python src/eda/04_emotion_correlation_matrix.py
```

Run only the grouped-emotion analysis workflow:

```bash
python src/grouped_eda/run_grouped_eda.py
```

Run the local BERT baseline directly:

```bash
python src/train_bert.py
```

This script fine-tunes a local `bert-base-uncased` emotion classifier on the Lemotif labels and derives three-way sentiment (`negative`, `neutral`, `positive`) from the predicted emotion set.

Optional environment variables:

- `LEMOTIF_BERT_MODEL`: Hugging Face model id or a local model directory
- `LEMOTIF_LOCAL_FILES_ONLY=1`: force offline loading from a local cache/directory
- `LEMOTIF_EPOCHS`, `LEMOTIF_BATCH_SIZE`, `LEMOTIF_MAX_LENGTH`, `LEMOTIF_LEARNING_RATE`: override the default training settings

Training outputs are written to `output/bert_emotion_model/`, including the saved checkpoint, metrics, threshold scan, and test predictions.

Generate the standalone model-evaluation figures after training:

```bash
python src/model_eda/run_model_eda.py
```

This writes the BERT evaluation visuals to `output/bert_emotion_model/figures/`.

Run the threshold/cardinality hyperparameter test:

```bash
python src/model_eda/run_threshold_cardinality_test.py
```

This keeps the fine-tuned BERT weights unchanged, but tests a stricter inference rule:

- emotion threshold: `0.59`
- minimum predicted emotions per reflection: `1`
- target: reduce rows with at least two extra predicted emotions to roughly 20-30

Outputs are written to `output/bert_emotion_model/threshold_cardinality_test/` and are annotated as a hyperparameter-testing change.

Run the Optuna-tuned BERT workflow:

```bash
python src/train_bert_optuna.py
```

This keeps the original baseline untouched and writes the Optuna-selected model artifacts to `output/bert_emotion_model_optuna/`, including the best checkpoint, metrics, trial history, threshold scan, and test predictions.

Optuna-specific environment variables:

- `LEMOTIF_OPTUNA_TRIALS`: number of Optuna trials to run, default `10`
- `LEMOTIF_OPTUNA_OUTPUT_DIR`: output directory for the tuned run, default `output/bert_emotion_model_optuna/`
- `LEMOTIF_OPTUNA_STUDY_NAME`: optional study label stored in the outputs
- `LEMOTIF_OPTUNA_SEED`: optional seed override for the tuning workflow

Generate the Optuna comparison visuals after the run:

```bash
python src/model_eda/run_optuna_comparison.py
```

This writes the baseline-vs-Optuna comparison figures to `output/bert_emotion_model_optuna/comparisons/`, including:

- manual default hyperparameters vs Optuna-selected hyperparameters
- validation/test metric comparison for the two BERT runs
- per-emotion and per-sentiment F1 comparisons
- Optuna optimization history
- parameter-importance chart when available

Run the separate Stories external-test workflow:

```bash
python src/test_data/run_all.py
```

This workflow:

- converts `stories_source.xlsx` into `test data/stories_data.csv` with the Lemotif column structure
- runs the full EDA suite on the Stories CSV and writes the outputs to `test data/output/eda/`
- retrains BERT on the full Lemotif dataset and evaluates it on the Stories CSV as an external test set, saving outputs to `test data/output/external_evaluation/`

Each EDA script automatically prefers `output/lemotif_cleaned.csv` when it exists, and falls back to the raw CSV otherwise.

## What Each Script Produces

- `01_text_length_distribution.py`: distribution of reflection word counts
- `02_emotion_prevalence.py`: prevalence of emotion labels
- `03_topic_prevalence.py`: prevalence of topic labels
- `04_emotion_correlation_matrix.py`: emotion correlation heatmap
- `05_topic_emotion_heatmap.py`: topic-conditioned emotion prevalence heatmap
- `06_emotion_labels_per_entry.py`: number of emotion labels per entry
- `07_topic_labels_per_entry.py`: number of topic labels per entry
- `08_representative_entries.py`: table of highly labeled example reflections
- `09_emotion_cooccurrence_network.py`: emotion co-occurrence network graph
- `10_length_vs_emotion_probability.py`: relationship between text length and top emotion probabilities
- `11_topic_to_emotion_probability.py`: strongest emotion associated with each topic
- `grouped_eda/00_group_definition_summary.py`: grouped-emotion mapping table and summary
- `grouped_eda/01_group_prevalence.py`: grouped-emotion prevalence and grouped labels per entry
- `grouped_eda/02_group_cooccurrence.py`: normalized grouped co-occurrence heatmaps using phi and lift
- `grouped_eda/03_topic_group_associations.py`: raw topic prevalence vs adjusted topic effects after controlling for positive co-occurrence
- `grouped_eda/04_group_feature_importance.py`: permutation-based feature importance for grouped emotions
- `train_bert.py`: local single-entry BERT baseline for multi-label emotion classification with derived sentiment reporting

## Cleaning Rules

The cleaning step currently:

- removes URLs
- normalizes whitespace
- removes very short responses
- removes entries with no emotion labels
- removes duplicate reflections
- removes long-text outliers with 300 or more words
- recomputes word, character, emotion, and topic counts

## Notes

- The default dataset path is `data/Lemotif thesis data.csv`.
- Shared paths and script order are defined in `src/config.py`.
- The BERT baseline is intentionally a single-entry model so it can serve as a clean thesis baseline before adding any longitudinal user-history variant.
- The previous split implementation has been kept under `archive/legacy_cleaning_and_extra_eda/` for reference.
