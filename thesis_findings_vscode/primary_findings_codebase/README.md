# Primary Findings Codebase

This mini-project produces the thesis primary findings:

- Lemotif in-domain BERT multi-label emotion baseline
- derived sentiment metrics from emotion predictions
- Stories external-test generalization result
- trained-vs-untrained BERT control comparison
- primary thesis-facing markdown and artifact index

## Install

```powershell
python -m pip install -r requirements.txt
```

Inside the current thesis repository you can also reuse the parent virtual environment:

```powershell
..\..\.venv\Scripts\python.exe run_primary_findings.py
```

## Fast Run

This uses bundled metrics and selected figures, then regenerates primary summary markdown:

```powershell
python run_primary_findings.py
```

Outputs:

- `output/findings/primary_findings.md`
- `output/findings/primary_metrics.csv`
- `output/findings/primary_artifact_index.csv`
- `output/bert_emotion_model/threshold_cardinality_test/*`
- `output/BERT-Hyperparamter-Fixed/figures/fig1_training_progress.png` through `fig7_example_prediction_table.png`
- `output/BERT-Hyperparamter-Fixed/comparisons/overall_metrics_comparison.csv`
- `output/BERT-Hyperparamter-Fixed/comparisons/per_emotion_metrics_comparison.csv`
- `output/BERT-Hyperparamter-Fixed/comparisons/prediction_cardinality_comparison.csv`
- `output/BERT-Hyperparamter-Fixed/comparisons/fig_compare_overall_metrics.png`
- `output/BERT-Hyperparamter-Fixed/comparisons/fig_compare_per_emotion_f1.png`
- `output/BERT-Hyperparamter-Fixed/comparisons/fig_compare_prediction_cardinality.png`
- `output/presentation_assets/training_run_comparison.png`
- `output/presentation_assets/stories_external_per_emotion_f1.png`
- `test data/output/eda/*`
- complete discovered `test data/output/*` entries in the primary artifact index

The threshold/cardinality output is annotated as a hyperparameter-testing change:

- emotion threshold: `0.59`
- minimum predicted emotions per reflection: `1`
- target: reduce rows with at least two extra predicted emotions to roughly 20-30

## Full Rebuild

This retrains BERT and reruns the Stories external evaluation. It can take a long time and may download Hugging Face model files.

```powershell
python run_primary_findings.py --full
```

## Notes

- The default Stories source workbook is `data/stories_source.xlsx`.
- If you need another workbook, set `STORIES_SOURCE_XLSX_PATH`.
- The fast run intentionally avoids expensive model training.
