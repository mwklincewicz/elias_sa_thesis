# WordPress Reproducibility Manifest

## Scope

This codebase reproduces the public WordPress reflection sanity-check section. It is separate from the model-performance findings because the WordPress data has no human emotion labels.

## Inputs

- `data/selected_sites.txt`: selected public WordPress sites.
- Lemotif emotion BERT model from the primary/model rebuild codebase.
- Lemotif cleaned dataset from the primary/secondary/model rebuild codebase for training the local topic/theme model.
- Live public WordPress posts at run time.

## Outputs

- `output/selected_site_analysis/public_reflection_predictions.csv`: full scrape and predictions, regenerated locally.
- `output/selected_site_analysis/run_summary.json`: collection summary.
- `output/selected_site_analysis/report/`: aggregate figures and descriptive tables.
- `output/selected_site_analysis/thesis_input_tables/`: thesis input tables and metadata.

## Recommended Run Order

```powershell
python run_wordpress_findings.py smoke
python run_wordpress_findings.py train-theme-model
python run_wordpress_findings.py selected-sites
python run_wordpress_findings.py report
python run_wordpress_findings.py thesis-tables
```

## Reproducibility Caveats

- The WordPress results are time-sensitive because public posts can be edited, deleted, or made private.
- Exact counts can change if sites change after the original collection.
- The corpus is unannotated, so only descriptive statistics are reproducible; formal predictive performance is not measurable on this dataset.
- Full scraped text files are not bundled by default because they contain large public personal reflections. The selected-site list and code are included so an authorized reviewer can regenerate the data.
- Network access and respectful use of public websites are required for a full rebuild.

## Thesis Claim Boundary

Use WordPress to support claims about large-scale descriptive behavior of the trained model on public reflective writing. Do not use it to claim accuracy, F1, clinical validity, or formal bias measurement.
