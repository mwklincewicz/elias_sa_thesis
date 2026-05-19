# WordPress Public Reflection Findings Codebase

This mini-project contains the WordPress public-reflection code used for the thesis sanity-check section. It discovers or loads public WordPress sites, scrapes recent journal-style posts, runs the Lemotif emotion model, optionally runs a local Lemotif topic/theme model, derives sentiment, and builds aggregate report figures.

The WordPress data is unannotated. Treat this section as descriptive external sanity-check evidence only: it cannot produce formal accuracy, F1, or bias metrics without gold labels.

## Quick Checks

Run the deterministic smoke test first. It does not use the internet or model files:

```powershell
python run_wordpress_findings.py smoke
```

List all reproducibility steps:

```powershell
python run_wordpress_findings.py recommended-order
```

Dry-run a full selected-site step:

```powershell
python run_wordpress_findings.py selected-sites --dry-run
```

## Full Rebuild Order

```powershell
python run_wordpress_findings.py train-theme-model
python run_wordpress_findings.py selected-sites
python run_wordpress_findings.py report
python run_wordpress_findings.py thesis-tables
```

The selected-sites step requires internet access and local model artifacts.

## Model Inputs

By default, the code looks for the emotion BERT model in sibling thesis codebases:

- `../primary_findings_codebase/output/bert_emotion_model/model`
- `../model_rebuild_codebase/output/bert_emotion_model/model`
- `../../output/bert_emotion_model/model`

Override if needed:

```powershell
python run_wordpress_findings.py selected-sites --env WORDPRESS_EMOTION_MODEL_PATH=C:\path\to\bert_emotion_model\model
```

The theme model is trained locally by `train-theme-model` and saved to:

```text
output/bert_theme_model/model
```

## Data And Ethics Note

The reproducibility workspace includes:

- the selected site list: `data/selected_sites.txt`
- aggregate report figures and tables under `output/selected_site_analysis/report`
- collection metadata under `output/selected_site_analysis/run_summary.json`

The full scraped text tables are intentionally not bundled by default because they contain large public personal reflections. They can be regenerated with `python run_wordpress_findings.py selected-sites` when a supervisor has internet access and local model files.

## Thesis Wording

Recommended wording:

> The WordPress corpus was generated from public WordPress posts using a reproducible scraper and selected-site list. Because these posts were not human annotated, the WordPress section is used only as descriptive external sanity-check evidence. It evaluates whether model outputs appear plausible at scale and whether obvious distributional skews appear, but it does not estimate formal accuracy, F1-score, or true bias.
