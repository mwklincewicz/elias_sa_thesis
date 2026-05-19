# Tilburg DSS Reproducibility Audit

Checked on 2026-05-17 against:

- `C:/Users/ee/Downloads/Tilburg_DSS_thesis_overleaf_ready_20260512_221015 (2).zip`
- `C:/Users/ee/Downloads/DSS_Master Thesis guidelines - vS2025.pdf`
- `thesis_findings_vscode/` in this repository

## Overall Verdict

The codebase is close to thesis-supervisor reproducible for the Lemotif, Stories, BERT, Optuna, Gemma, WordPress, primary findings, and secondary findings parts. The runnable structure is strong: findings can be regenerated, and each model family or corpus section can be rebuilt one step at a time.

The main remaining issues are documentation and claim alignment, not major code rewrites:

- Fill the DSECT placeholders in `main.tex`: Lemotif source, permission/publication year, and GitHub link.
- Add exact package versions or an environment snapshot for strict reproducibility.
- WordPress reproducibility is now included as `thesis_findings_vscode/wordpress_findings_codebase`; describe it as a public, unannotated, descriptive sanity-check pipeline.
- Remove or rewrite unsupported method claims in the thesis text unless new artifacts are added: weak-label training, SHAP, longitudinal/user-history modeling, formal inter-annotator reliability, and translation if no reproducible translation pipeline is supplied.
- State clearly that Stories evaluation data is private/anonymized and cannot be redistributed publicly.

## Tilburg DSS Expectations Relevant To This Thesis

The guideline PDF expects the thesis to provide:

- A DSECT statement for Data Source, Ethics, Code, and Technology.
- Enough experimental detail for another researcher to replicate the work from the description alone.
- Raw dataset description: owner, acquisition channel, sample size, collection timing, features, labels, and known biases.
- Cleaning and preprocessing decisions, including discarded data, missing values, transformations, and why these decisions were made.
- Experimental procedure: train/test/validation selection, order of pipeline steps, algorithms used, parameter values, and tuning choices.
- Evaluation criteria: metrics used and why they fit the task.
- Robustness/generalization procedure: validation, held-out testing, and out-of-sample evaluation.
- Implementation details: programming language, package/library versions, proprietary tools, and any technology-assisted writing/coding.
- Repository availability where possible, with GitHub explicitly mentioned as a common choice.
- Error-pattern visualizations for classification, such as confusion matrices and class-level error analysis.

## Reproducibility Components Present

- `thesis_findings_vscode/README.md` gives a supervisor-facing overview of the three runnable mini-codebases.
- `primary_findings_codebase/run_primary_findings.py` completed successfully in the smoke test. It regenerates primary findings without retraining by default and supports `--full` for retraining.
- `secondary_findings_codebase/run_secondary_findings.py` completed successfully in the smoke test. It regenerates EDA, grouped emotion analyses, dataset comparison figures, and secondary summaries.
- `model_rebuild_codebase/run_models.py recommended-order` lists the one-by-one model rebuild order.
- `model_rebuild_codebase/run_models.py <task> --dry-run` prints exact scripts for a model step without executing it.
- `wordpress_findings_codebase/run_wordpress_findings.py` documents and runs the WordPress smoke test, theme-model training, selected-site scrape, aggregate report generation, and thesis-table generation.
- `model_rebuild_codebase/REPRODUCIBILITY.md` documents raw inputs, rebuild order, expected outputs, model-specific caveats, and randomness caveats.
- `wordpress_findings_codebase/WORDPRESS_REPRODUCIBILITY.md` documents the public WordPress input list, outputs, claim boundary, and live-web caveats.
- The BERT scripts store key reproducibility metadata in metrics artifacts: dataset path, model name, seed, maximum sequence length, batch size, epochs, learning rate, weight decay, warmup ratio, best epoch, threshold, label names, split sizes, validation metrics, and test metrics.
- The Optuna script stores the objective metric, requested trials, selected hyperparameters, search space, best trial, and metrics.
- The fixed BERT threshold/cardinality change is annotated as a hyperparameter-testing change, not a retrained model.
- The Gemma rebuild scripts document access and compute caveats and save model/evaluation metadata.
- The Overleaf package includes `figure_manifest.csv`, linking packaged figures to source artifacts.

## Gaps To Fix Or State Explicitly

- Requirements are not pinned to exact versions. Add a `pip freeze` snapshot or state that exact numeric results may vary slightly across package/hardware versions.
- Hardware and package versions are not consistently stored in all model metrics. At minimum, report Python version, PyTorch version, Transformers version, GPU/CPU, CUDA availability, and whether Hugging Face models were cached or downloaded.
- WordPress full scraped-text tables and generated output folders are not bundled by default because they contain large public personal reflections or generated artifacts. The selected-site list, code, documentation, and regeneration commands are included.
- The current DSECT paragraph in `main.tex` still contains placeholders for the Lemotif data source, year of permission/publication, and code repository link.
- The thesis text should not claim SHAP, weak-label training, longitudinal modeling, translation, or formal agreement statistics unless those exact artifacts are added.
- Stories evaluation data should be described as restricted/private. Public reproducibility can include code, schema, aggregate outputs, and synthetic examples, while full replication requires approved access to the anonymized workbook.
- Gemma reproducibility depends on accepted Hugging Face access and available compute. This should be stated in Methods or the reproducibility appendix.

## Tailored DSECT Bullet Points To Write Out

### Data Source

- Lemotif data was used as the primary training dataset for multi-label emotion classification. It contains free-text daily reflections annotated with 18 emotion labels and 11 topic labels. The cleaned analysis file contains 1,473 rows after preprocessing.
- Lemotif ownership and permission should be stated precisely: name the dataset owner/source, acquisition channel, date or year of access, and whether reuse was permitted through public release or request-based access.
- The Stories evaluation dataset was used as the out-of-domain human-annotated evaluation set. It originates from the Stories reflection project in collaboration with the University of Twente and contains 114 human-labeled reflections in the reproducible analysis file.
- Stories evaluation data is anonymized and should not be made public because of its sensitive reflective/mental-health context. Full replication requires authorized access; public reproducibility is provided through code, schema, aggregate metrics, and generated figures.
- WordPress reflections, if retained in the thesis, were used only as an unannotated descriptive sanity check. Because there are no gold labels, the thesis should not report formal accuracy, F1, or bias metrics for WordPress.
- All thesis figures were generated by the author from the analysis code or packaged outputs. Any external images, logos, or dataset screenshots need permission/credit.

### Ethics

- No personally identifying Stories fields should be published. User identifiers, timestamps, image URLs, and other potentially identifying fields should be excluded from public artifacts unless anonymized and necessary.
- Sensitive data and model processing were kept local where possible to reduce exposure of reflective mental-health text.
- The thesis should state known dataset biases: Lemotif is crowdsourced, short-form, and skewed toward common/positive labels; Stories is small and domain-specific; WordPress is public but unannotated and not representative of clinical reflections.
- The WordPress corpus should be framed carefully: public availability does not remove ethical responsibility, especially for reflective or personal writing.

### Code

- The code should be linked through a GitHub repository containing `thesis_findings_vscode/`.
- Primary findings can be reproduced with `python run_primary_findings.py`.
- Secondary findings can be reproduced with `python run_secondary_findings.py`.
- Model rebuilds can be inspected and run one by one with `python run_models.py recommended-order` and `python run_models.py <task-name>`.
- WordPress sanity-check results can be inspected and regenerated from `wordpress_findings_codebase` with `python run_wordpress_findings.py recommended-order`.
- BERT uses `bert-base-uncased`, fixed random seed 42 by default, a 70/15/15 Lemotif split, validation-selected thresholding, and saved JSON/CSV artifacts for metrics and predictions.
- The fixed BERT threshold/cardinality segment uses unchanged BERT weights with threshold 0.59 and a minimum of one predicted emotion per reflection. This should be called an inference hyperparameter test.
- Optuna BERT tuning should be described as validation-based hyperparameter search, including trial count and selected hyperparameters.
- Gemma tasks require gated model access and sufficient compute, so exact reproduction may require local cache, Hugging Face access approval, and a compatible GPU.

### Technology

- Python was used for cleaning, modeling, evaluation, and figure generation.
- LaTeX/Overleaf was used for typesetting.
- ChatGPT/Codex assistance should be acknowledged for language polishing, code organization, debugging, and reproducibility documentation where applicable. The author remained responsible for analysis decisions, interpretation, and final text.
- If no separate reference manager was used, state that references were managed through the LaTeX bibliography files.
- If no generated text was inserted without review, state that AI assistance was used for drafting support and revision, not as an unverified source of scientific claims.

### Reproducibility Paragraph Content

- The repository contains three runnable codebases: primary findings, secondary findings, and model rebuilds.
- Raw restricted data files are not all publicly redistributable; therefore, public reproducibility is split from authorized full replication.
- All generated outputs are written to `output/` or `test data/output/`.
- The thesis reports both in-domain Lemotif evaluation and external Stories evaluation to reduce overreliance on training-domain results.
- Exact deep-learning results may differ slightly across hardware and library versions; this is why seeds, metrics JSON files, prediction CSVs, thresholds, and model rebuild commands are preserved.

## Recommended Small Additions Before Submission

1. Add the final GitHub link to the DSECT statement.
2. Add `pip freeze > environment_snapshot.txt` or an equivalent exported environment file.
3. Add a short model info sheet appendix for BERT, Optuna BERT, fixed-threshold BERT, and Gemma.
4. In the thesis text, state that the WordPress code is reproducible but exact live-web counts may change if public sites edit, delete, or privatize posts.
5. Replace unsupported planned-method text in `main.tex` with the completed analyses: LIME, feature ablation, threshold calibration, rare-label error analysis, and external Stories evaluation.
