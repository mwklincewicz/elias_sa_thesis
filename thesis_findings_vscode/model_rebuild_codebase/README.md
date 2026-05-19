# Model Rebuild Codebase

This mini-project lets a thesis supervisor rebuild every model artifact used in the thesis one step at a time.

It is separate from the primary and secondary findings projects:

- findings projects summarize evidence and regenerate figures;
- this project retrains/evaluates model artifacts.

## Install

```powershell
python -m pip install -r requirements.txt
```

Inside the current thesis repository, you can reuse the parent virtual environment:

```powershell
..\..\.venv\Scripts\python.exe run_models.py list
```

## List Available Model Steps

```powershell
python run_models.py list
```

Recommended order:

```powershell
python run_models.py recommended-order
```

Run one step:

```powershell
python run_models.py bert-baseline
```

Dry run without executing:

```powershell
python run_models.py bert-baseline --dry-run
```

Force cached/local Hugging Face models where supported:

```powershell
python run_models.py bert-baseline --local-files-only
```

Override runtime hyperparameters:

```powershell
python run_models.py bert-baseline --env LEMOTIF_EPOCHS=1
python run_models.py bert-optuna --env LEMOTIF_OPTUNA_TRIALS=3
```

## Available Tasks

- `prepare-data`: clean Lemotif and format Stories.
- `bert-baseline`: train the main fine-tuned BERT multi-label emotion classifier.
- `bert-baseline-figures`: generate BERT result figures.
- `bert-threshold-cardinality`: run the threshold/cardinality hyperparameter test.
- `bert-hyperparameter-fixed`: generate the BERT-Hyperparamter-Fixed segment with 7 figures, 3 comparison tables, and 3 comparison graphs.
- `bert-optuna`: run Optuna tuning for BERT.
- `bert-optuna-comparison`: compare baseline BERT and Optuna BERT.
- `stories-external-bert`: train on full Lemotif and evaluate on Stories.
- `untrained-bert`: evaluate random-weight BERT on Stories.
- `untrained-bert-comparison`: compare trained and untrained BERT on Stories.
- `dataset-comparisons`: recreate Lemotif-vs-Stories distribution comparisons.
- `gemma-lemotif`: run the Gemma Lemotif emotion pipeline.
- `gemma-stories-sentiment`: evaluate Gemma on Stories sentiment.
- `gemma-stories-figures`: create Gemma Stories figures and BERT-vs-Gemma comparisons.

## Reproducibility Notes

- See `REPRODUCIBILITY.md` for the supervisor-facing run order and artifact manifest.
- Raw input files are stored under `data/`.
- All generated artifacts are written under `output/` or `test data/output/`.
- BERT tasks may download `bert-base-uncased` unless it is cached.
- Gemma tasks require access to the gated Gemma model repository and may require GPU memory. Use `GEMMA_LOCAL_FILES_ONLY=1` after caching the model locally.
- The threshold/cardinality step is an inference hyperparameter test, not a newly trained model.
- The `BERT-Hyperparamter-Fixed` segment packages that inference setting as a reproducible model-output folder; it still uses unchanged fine-tuned BERT weights.
