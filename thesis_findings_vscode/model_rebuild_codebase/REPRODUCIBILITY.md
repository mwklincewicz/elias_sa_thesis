# Model Rebuild Reproducibility Manifest

This file documents the model rebuild steps for thesis review.

## Scope

This codebase rebuilds model artifacts only. It does not write the thesis narrative and does not replace the primary/secondary findings codebases.

## Raw Inputs

- `data/Lemotif thesis data.csv`
- `data/stories_source.xlsx`

## Recommended Run Order

Run each step separately:

```powershell
python run_models.py prepare-data
python run_models.py bert-baseline
python run_models.py bert-baseline-figures
python run_models.py bert-threshold-cardinality
python run_models.py bert-hyperparameter-fixed
python run_models.py bert-optuna
python run_models.py bert-optuna-comparison
python run_models.py stories-external-bert
python run_models.py untrained-bert
python run_models.py untrained-bert-comparison
python run_models.py dataset-comparisons
python run_models.py gemma-lemotif
python run_models.py gemma-stories-sentiment
python run_models.py gemma-stories-figures
```

## BERT Rebuilds

### `bert-baseline`

Trains the main `bert-base-uncased` multi-label emotion classifier on Lemotif.

Primary outputs:

- `output/bert_emotion_model/model/`
- `output/bert_emotion_model/metrics.json`
- `output/bert_emotion_model/test_predictions.csv`
- `output/bert_emotion_model/run_summary.txt`

### `bert-threshold-cardinality`

Runs an inference hyperparameter test on the fine-tuned BERT predictions.

This does not retrain model weights.

Default inference hyperparameters:

- threshold: `0.59`
- minimum predicted labels per row: `1`

Primary outputs:

- `output/bert_emotion_model/threshold_cardinality_test/metrics.json`
- `output/bert_emotion_model/threshold_cardinality_test/test_predictions_threshold_0_59_min1.csv`
- `output/bert_emotion_model/threshold_cardinality_test/threshold_sweep.csv`

### `bert-hyperparameter-fixed`

Packages the adjusted inference run as `BERT-Hyperparamter-Fixed`.

This does not retrain model weights. It creates a model-output segment with the same seven figure names as `output/bert_emotion_model/figures/` plus three comparison tables and three readable comparison graphs against the baseline BERT run.

Primary outputs:

- `output/BERT-Hyperparamter-Fixed/metrics.json`
- `output/BERT-Hyperparamter-Fixed/test_predictions.csv`
- `output/BERT-Hyperparamter-Fixed/figures/fig1_training_progress.png` through `fig7_example_prediction_table.png`
- `output/BERT-Hyperparamter-Fixed/comparisons/overall_metrics_comparison.csv`
- `output/BERT-Hyperparamter-Fixed/comparisons/per_emotion_metrics_comparison.csv`
- `output/BERT-Hyperparamter-Fixed/comparisons/prediction_cardinality_comparison.csv`
- `output/BERT-Hyperparamter-Fixed/comparisons/fig_compare_overall_metrics.png`
- `output/BERT-Hyperparamter-Fixed/comparisons/fig_compare_per_emotion_f1.png`
- `output/BERT-Hyperparamter-Fixed/comparisons/fig_compare_prediction_cardinality.png`

### `bert-optuna`

Runs Optuna tuning for BERT.

Useful runtime override:

```powershell
python run_models.py bert-optuna --env LEMOTIF_OPTUNA_TRIALS=3
```

Primary outputs:

- `output/bert_emotion_model_optuna/model/`
- `output/bert_emotion_model_optuna/metrics.json`
- `output/bert_emotion_model_optuna/study_trials.csv`

## Stories External Test

### `stories-external-bert`

Trains on full Lemotif and evaluates on Stories.

Primary outputs:

- `test data/output/external_evaluation/model/`
- `test data/output/external_evaluation/metrics.json`
- `test data/output/external_evaluation/stories_test_predictions.csv`

### `untrained-bert`

Evaluates random BERT weights as a control.

Primary outputs:

- `test data/output/untrained_bert/metrics.json`
- `test data/output/untrained_bert/stories_test_predictions.csv`

## Gemma Rebuilds

Gemma tasks require accepted Hugging Face model access and may require substantial GPU memory.

Useful local-cache command:

```powershell
python run_models.py gemma-lemotif --local-files-only
```

Primary outputs:

- `output/gemma_2b_emotion_model/`
- `test data/output/gemma_sentiment/`

## Environment And Randomness

The BERT scripts use fixed seeds from the source configuration where available. Exact results may still vary slightly across:

- CPU vs GPU execution
- installed PyTorch/Transformers versions
- Hugging Face model cache revision
- CUDA/cuDNN behavior
- Optuna trial count and sampler behavior

For strict replication, record:

- Python version
- package versions from `pip freeze`
- device used for training
- whether models were loaded from local cache or downloaded
- all environment overrides passed with `--env`
