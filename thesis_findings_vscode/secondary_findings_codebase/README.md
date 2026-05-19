# Secondary Findings Codebase

This mini-project produces the thesis secondary findings:

- Lemotif cleaning summary
- Lemotif descriptive EDA figures
- Stories formatting into the Lemotif label structure
- Lemotif-vs-Stories emotion and sentiment distribution comparisons
- grouped emotion prevalence, co-occurrence, topic associations, and grouped feature-importance outputs
- threshold/cardinality hyperparameter-test summary for the fine-tuned BERT model
- secondary thesis-facing markdown and artifact index

## Install

```powershell
python -m pip install -r requirements.txt
```

Inside the current thesis repository you can also reuse the parent virtual environment:

```powershell
..\..\.venv\Scripts\python.exe run_secondary_findings.py
```

## Run

```powershell
python run_secondary_findings.py
```

Outputs:

- `output/findings/secondary_findings.md`
- `output/findings/secondary_evidence_tables.csv`
- `output/findings/secondary_artifact_index.csv`
- `output/fig*.png`
- `output/grouped_eda/*`
- `output/dataset_comparison/*`
- `test data/output/eda/*`
- complete discovered `test data/output/*` entries in the secondary artifact index

## Notes

- This project is lightweight compared with the primary model project.
- The available grouped feature-importance output is permutation-style analysis, not SHAP.
- The threshold/cardinality section is an inference hyperparameter test, not a newly trained model.
- WordPress/public-blog validation is intentionally not included because that code is not present in the current source repository.
