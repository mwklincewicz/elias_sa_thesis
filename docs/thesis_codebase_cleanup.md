# Thesis Codebase Cleanup And Findings Map

This document maps the current repository to the thesis draft and separates thesis-critical code from optional experiments and archive candidates.

## Short Answer

The current codebase already supports the main thesis spine:

- Lemotif cleaning and descriptive EDA
- in-domain BERT multi-label emotion classification
- derived sentiment from predicted emotion labels
- Stories formatting into the Lemotif label structure
- Stories external evaluation
- untrained BERT control baseline
- optional Optuna and Gemma model comparisons

The codebase does not currently contain these thesis claims as reproducible tracked code:

- WordPress/public-blog dataset search generator and analysis
- SHAP feature attribution
- Krippendorff's alpha or inter-annotator agreement
- Stories weak-label generation by LLM
- translation of Stories entries into English
- longitudinal/user-history modeling

So the safest thesis cleanup is: keep the Lemotif/Stories BERT pipeline as the primary result, keep dataset comparison and grouped EDA as secondary findings, and mark WordPress/SHAP/longitudinal work as future or not-yet-implemented unless those scripts are added.

## Primary Findings

### Primary Finding 1: Lemotif BERT Baseline

Thesis claim supported:

> A single-entry BERT classifier trained on Lemotif can learn useful multi-label emotion signals, with sentiment derived afterward from the predicted emotion set.

Keep these files:

- `src/config.py`
- `src/data_loader.py`
- `src/clean_data.py`
- `src/run_pipeline.py`
- `src/train_bert.py`
- `src/model_eda/common.py`
- `src/model_eda/run_model_eda.py`
- `src/model_eda/01_training_progress.py`
- `src/model_eda/02_validation_vs_test_metrics.py`
- `src/model_eda/03_per_emotion_f1.py`
- `src/model_eda/04_precision_recall_scatter.py`
- `src/model_eda/05_true_vs_predicted_label_counts.py`
- `src/model_eda/06_sentiment_confusion_matrix.py`
- `src/model_eda/07_example_prediction_table.py`

Most important current evidence:

- `output/bert_emotion_model/run_summary.txt`
- `output/bert_emotion_model/metrics.json`
- `output/bert_emotion_model/figures/fig2_validation_vs_test_metrics.png`
- `output/bert_emotion_model/figures/fig3_per_emotion_f1.png`
- `output/bert_emotion_model/figures/fig6_sentiment_confusion_matrix.png`

Runnable thesis-facing summary:

```powershell
python src/findings/01_primary_findings.py
```

Output:

- `output/findings/primary_findings.md`
- `output/findings/primary_metrics.csv`
- `output/findings/primary_artifact_index.csv`

### Primary Finding 2: Stories External Test / Domain Shift

Thesis claim supported:

> Models trained on Lemotif generalize only partially to Stories reflections, showing a clear external-test drop.

Keep these files:

- `src/test_data/common.py`
- `src/test_data/prepare_stories_data.py`
- `src/test_data/run_stories_eda.py`
- `src/test_data/train_full_lemotif_external.py`
- `src/test_data/run_all.py`
- `src/test_data/evaluate_untrained_bert.py`
- `src/test_data/run_untrained_bert_comparisons.py`
- `src/presentation/plot_emotion_dataset_comparison.py`
- `src/presentation/plot_sentiment_dataset_comparison.py`

Most important current evidence:

- `test data/stories_data.csv`
- `test data/stories_formatting_summary.txt`
- `test data/output/external_evaluation/run_summary.txt`
- `test data/output/external_evaluation/metrics.json`
- `test data/output/untrained_bert/comparison_summary.txt`
- `output/dataset_comparison/fig_emotion_distribution_comparison.png`
- `output/dataset_comparison/fig_sentiment_distribution_comparison.png`

Important note for thesis consistency:

The draft says the Stories evaluation dataset has 198 reflections, while the current formatted artifact has 114 rows. Either update the thesis text to match the specific annotated/evaluated subset, or add the missing rows to the reproducible data path.

## Secondary Findings

### Secondary Finding 1: Dataset Imbalance And Label Structure

Thesis claim supported:

> Lemotif is a highly imbalanced multi-label dataset, so micro/macro F1 and per-label F1 are more informative than accuracy alone.

Keep these files:

- `src/eda/common.py`
- `src/eda/01_text_length_distribution.py`
- `src/eda/02_emotion_prevalence.py`
- `src/eda/03_topic_prevalence.py`
- `src/eda/04_emotion_correlation_matrix.py`
- `src/eda/05_topic_emotion_heatmap.py`
- `src/eda/06_emotion_labels_per_entry.py`
- `src/eda/07_topic_labels_per_entry.py`
- `src/eda/08_representative_entries.py`

Optional EDA files:

- `src/eda/09_emotion_cooccurrence_network.py`
- `src/eda/10_length_vs_emotion_probability.py`
- `src/eda/11_topic_to_emotion_probability.py`

### Secondary Finding 2: Grouped Emotion Interpretation

Thesis claim supported:

> Grouping the 18 Lemotif emotions into broader affective families helps interpret imbalance and co-occurrence patterns.

Keep if you discuss grouped emotion structure:

- `src/grouped_eda/common.py`
- `src/grouped_eda/run_grouped_eda.py`
- `src/grouped_eda/00_group_definition_summary.py`
- `src/grouped_eda/01_group_prevalence.py`
- `src/grouped_eda/02_group_cooccurrence.py`
- `src/grouped_eda/03_topic_group_associations.py`
- `src/grouped_eda/04_group_feature_importance.py`

Use carefully:

- `src/grouped_eda/04_group_feature_importance.py` is useful, but it is not SHAP. In the thesis, call it permutation-style grouped feature importance unless SHAP is added.

Runnable secondary summary:

```powershell
python src/findings/02_secondary_findings.py
```

Output:

- `output/findings/secondary_findings.md`
- `output/findings/secondary_evidence_tables.csv`
- `output/findings/secondary_artifact_index.csv`

Run both findings summaries:

```powershell
python src/findings/run_all_findings.py
```

## Optional Experiments

Keep these only if they appear in the Results, Discussion, or appendix:

- `src/train_bert_optuna.py`
- `src/model_eda/run_optuna_comparison.py`
- `src/gemma_2b_emotion_pipeline.py`
- `src/test_data/evaluate_gemma_sentiment.py`
- `src/test_data/run_gemma_visuals.py`
- `src/presentation/build_thesis_presentation.py`

Recommended thesis framing:

- Optuna: secondary hyperparameter tuning result, not the headline model unless you rerun and document the search cleanly.
- Gemma emotion pipeline: optional comparison against a stronger generative model on Lemotif.
- Gemma Stories sentiment: optional separate sentiment-only comparison, not the same task as multi-label emotion classification.
- Presentation builder: useful for defense slides, not necessary for the thesis code repository.

## Cutdown And Archive Suggestions

### Keep In Main Thesis Repo

- `README.md`
- `requirements.txt`
- `data/Lemotif thesis data.csv`
- `data/stories_source.xlsx`, if allowed to be stored in the repository
- `src/config.py`
- `src/data_loader.py`
- `src/clean_data.py`
- `src/run_pipeline.py`
- `src/eda/`
- `src/model_eda/`
- `src/train_bert.py`
- `src/test_data/common.py`
- `src/test_data/prepare_stories_data.py`
- `src/test_data/run_stories_eda.py`
- `src/test_data/train_full_lemotif_external.py`
- `src/test_data/evaluate_untrained_bert.py`
- `src/test_data/run_untrained_bert_comparisons.py`
- `src/findings/`
- selected final figures under `output/`

### Move To Archive Or Appendix Folder

- `archive/legacy_cleaning_and_extra_eda/`
- `src/run_all.py`, because it is only a compatibility alias for `src/run_pipeline.py`
- `src/presentation/build_thesis_presentation.py`, unless defense-slide generation is part of the repo
- `src/train_bert_optuna.py`, if Optuna is not reported
- `src/gemma_2b_emotion_pipeline.py`, if Gemma emotion comparison is not reported
- `src/test_data/evaluate_gemma_sentiment.py`, if Gemma Stories sentiment is not reported
- `src/test_data/run_gemma_visuals.py`, if Gemma Stories sentiment is not reported

### Do Not Commit Heavy Generated Artifacts Unless Required

These are useful locally but usually too heavy or too generated for a clean thesis repo:

- `output/**/model/`
- `output/**/adapter/`
- `test data/output/**/model/`
- repeated generated PNGs that are not used in the thesis
- raw prediction CSVs if they contain sensitive Stories text

For thesis submission, prefer committing:

- `metrics.json`
- `run_summary.txt`
- selected final figures
- small aggregate CSVs
- scripts needed to reproduce them

## WordPress/Public-Blog Work

The thesis draft mentions a WordPress/publicly listed reflection dataset, but the repository currently has no crawler, generator, filter, or WordPress analysis script.

If this section remains in the thesis, add a small reproducible module:

- `src/public_reflections/collect_wordpress_posts.py`
- `src/public_reflections/filter_reflections.py`
- `src/public_reflections/evaluate_public_reflections.py`
- `src/public_reflections/run_all.py`

Recommended outputs:

- `data/public_reflections.csv`
- `output/public_reflections/collection_summary.txt`
- `output/public_reflections/predicted_emotion_distribution.csv`
- `output/public_reflections/fig_public_vs_lemotif_emotion_distribution.png`
- `output/public_reflections/fig_public_prediction_confidence.png`

Minimum ethical/reproducibility fields:

- source URL
- retrieval date
- public accessibility note
- title/date if available
- extracted text
- filtering reason
- deduplication hash
- whether text was excluded from public release

Do not claim WordPress bias validation in the Results section until this exists.

## Most Important Results Visuals

Use these first:

- Dataset flow diagram: Lemotif training, Stories external test, optional public-reflection sanity check
- Emotion prevalence comparison: `output/dataset_comparison/fig_emotion_distribution_comparison.png`
- Sentiment distribution comparison: `output/dataset_comparison/fig_sentiment_distribution_comparison.png`
- In-domain model metrics: `output/bert_emotion_model/figures/fig2_validation_vs_test_metrics.png`
- Per-emotion F1: `output/bert_emotion_model/figures/fig3_per_emotion_f1.png`
- Stories domain-shift summary: `output/presentation_assets/training_run_comparison.png`
- Stories per-emotion F1: `output/presentation_assets/stories_external_per_emotion_f1.png`
- Sentiment confusion matrix: `output/bert_emotion_model/figures/fig6_sentiment_confusion_matrix.png`

Optional visuals:

- Emotion co-occurrence network: `output/fig9_emotion_cooccurrence_network.png`
- Grouped emotion prevalence: `output/grouped_eda/fig1_group_prevalence_and_density.png`
- Grouped co-occurrence heatmaps: `output/grouped_eda/fig2_group_cooccurrence_heatmaps.png`
- Optuna comparison: `output/bert_emotion_model_optuna/comparisons/fig_compare_overall_metrics.png`
- Gemma comparison: `output/gemma_2b_emotion_model/comparison/fig_compare_runs.png`

## Suggested Results Section Shape

1. Dataset preparation and label distributions
2. Lemotif in-domain BERT baseline
3. Stories external test and domain-shift result
4. Derived sentiment results
5. Secondary model checks: untrained BERT, Optuna, Gemma
6. Error analysis and limitations
7. Optional public-reflection/WordPress sanity check, only if implemented

