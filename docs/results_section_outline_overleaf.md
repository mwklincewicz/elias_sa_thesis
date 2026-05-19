# Results Section Outline and Overleaf Implementation Guide

This guide implements the verified-only Results plan for the thesis. It is designed to be used while drafting the Overleaf Results chapter and while revising the Methods section so both sections describe the same completed analyses.

## 1. Figure Package

Desktop package:

`figures/results`

Upload this folder to Overleaf as:

`figures/results/`

The local folder contains:

- `main_text/`: ten figures intended for the main Results chapter.
- `appendix/`: supporting figures for appendices.
- `figure_manifest.csv`: source-to-Overleaf file map.

### Main-Text Figures

| Figure | Overleaf path | Results placement | Purpose |
|---|---|---|---|
| Figure 1 | `figures/results/main_text/fig01_lemotif_emotion_prevalence.png` | Section 5.1 | Shows Lemotif emotion imbalance and the positive-affect skew. |
| Figure 2 | `figures/results/main_text/fig02_lemotif_stories_distribution_shift.png` | Section 5.1 | Shows Lemotif vs. Stories emotion distribution shift. |
| Figure 3 | `figures/results/main_text/fig03_bert_validation_test_metrics.png` | Section 5.2 | Summarizes validation/test performance for the in-domain BERT model. |
| Figure 4 | `figures/results/main_text/fig04_bert_per_emotion_f1.png` | Section 5.2 | Shows which emotions are detected reliably and which are weak. |
| Figure 5 | `figures/results/main_text/fig05_sentiment_confusion_matrix.png` | Section 5.2 | Shows derived sentiment error patterns. |
| Figure 6 | `figures/results/main_text/fig06_fixed_inference_cardinality.png` | Section 5.3 | Shows the baseline vs. threshold/cardinality inference tradeoff. |
| Figure 7 | `figures/results/main_text/fig07_stories_external_model_drop.png` | Section 5.5 | Shows in-domain to Stories external-test performance drop. |
| Figure 8 | `figures/results/main_text/fig08_stories_per_emotion_f1.png` | Section 5.5 | Shows Stories per-emotion generalization. |
| Figure 9 | `figures/results/main_text/fig09_stories_sentiment_model_comparison.png` | Section 5.5 | Compares trained BERT, untrained BERT, and Gemma on Stories sentiment. |
| Figure 10 | `figures/results/main_text/fig10_wordpress_theme_emotion_heatmap.png` | Section 5.6 | Shows WordPress theme-emotion structure for descriptive RQ2. |

### Appendix Figures

Use these when the main text needs support without crowding the chapter.

| Appendix figure | Overleaf path | Recommended appendix |
|---|---|---|
| A1 | `figures/results/appendix/app_fig01_lemotif_word_count_distribution.png` | Appendix A: Dataset EDA |
| A2 | `figures/results/appendix/app_fig02_lemotif_topic_prevalence.png` | Appendix A: Dataset EDA |
| A3 | `figures/results/appendix/app_fig03_lemotif_topic_emotion_heatmap.png` | Appendix A: Dataset EDA |
| A4 | `figures/results/appendix/app_fig04_lemotif_emotion_cooccurrence_network.png` | Appendix A: Dataset EDA |
| A5 | `figures/results/appendix/app_fig05_grouped_emotion_prevalence_density.png` | Appendix A: Grouped emotions |
| B1 | `figures/results/appendix/app_fig06_fixed_inference_overall_metrics.png` | Appendix B: Model variants |
| B2 | `figures/results/appendix/app_fig07_bootstrap_metric_ci.png` | Appendix B: Bootstrap diagnostics |
| C1 | `figures/results/appendix/app_fig08_rare_label_error_rate.png` | Appendix C: Error analysis |
| C2 | `figures/results/appendix/app_fig09_multilabel_error_flow_heatmap.png` | Appendix C: Error analysis |
| C3 | `figures/results/appendix/app_fig10_threshold_sensitivity_curves.png` | Appendix C: Calibration |
| C4 | `figures/results/appendix/app_fig11_feature_ablation_overall_metrics.png` | Appendix C: Feature diagnostics |
| C5 | `figures/results/appendix/app_fig12_lime_representative_panel.png` | Appendix C: LIME explanations |
| D1 | `figures/results/appendix/app_fig13_wordpress_emotion_prevalence.png` | Appendix D: WordPress descriptive results |
| D2 | `figures/results/appendix/app_fig14_wordpress_emotion_correlation_matrix.png` | Appendix D: WordPress descriptive results |
| D3 | `figures/results/appendix/app_fig15_wordpress_emotion_cooccurrence_network.png` | Appendix D: WordPress descriptive results |

## 2. Results Chapter Outline

Use academic past tense. Avoid saying "will be trained" or "will be tested" in Results. Each section should name the research question it answers, give the relevant statistics, interpret them, and then direct the reader to figures or appendices.

### 5. Results Overview

Purpose:

- Introduce the structure of the Results chapter.
- State that RQ1 is answered through in-domain Lemotif evaluation and external Stories evaluation.
- State that RQ2 is answered through descriptive WordPress inference, not formal validation.
- Warn the reader that WordPress lacks gold labels, so no accuracy, F1, or model-bias score is claimed for that corpus.

Suggested paragraph:

The Results section reports the completed evaluation pipeline in four steps. First, the label distributions of Lemotif and Stories are compared to establish the affective and domain conditions under which the models were evaluated. Second, the in-domain Lemotif BERT results are presented for multi-label emotion classification and derived sentiment classification. Third, model variants and diagnostic analyses are used to clarify the tradeoff between emotion-level recall, prediction density, and sentiment-level stability. Finally, external generalization is assessed on the human-labeled Stories test set, while the WordPress corpus is used only as an ungold-labeled descriptive sanity check for inferred public-journal patterns.

Key writing constraints:

- Do not describe WordPress as "validated" unless you explicitly say "descriptive validation/sanity check without gold labels."
- Do not report WordPress accuracy, F1, or bias metrics.
- Do not claim SHAP, longitudinal modeling, or inter-annotator reliability unless separate artifacts are added.

### 5.1 Dataset and Label Distribution

Purpose:

- Establish class imbalance before model evaluation.
- Explain why the Lemotif baseline data has a strong positive-affect skew.
- Introduce Stories as a harder external test because it is less positive and more neutral/negative.

Main statistics:

- Lemotif rows: `1,473`.
- Lemotif sentiment distribution: positive `81.7%` (`n = 1,204`), negative `14.7%` (`n = 217`), neutral `3.5%` (`n = 52`).
- Stories sentiment distribution: positive `65.8%` (`n = 75`), negative `22.8%` (`n = 26`), neutral `11.4%` (`n = 13`).

Main figures:

- Figure 1: `fig01_lemotif_emotion_prevalence.png`
- Figure 2: `fig02_lemotif_stories_distribution_shift.png`

Bullet outline:

- Open by explaining that multi-label emotion classification must be interpreted in relation to label prevalence because rare labels impose stricter macro-F1 and per-label performance constraints.
- Use Figure 1 to describe the Lemotif distribution. Emphasize that positive labels, especially Happy, Satisfied, Calm, Proud, and Excited, dominate the dataset.
- Interpret this skew through the Pollyanna hypothesis/principle: positive evaluative language tends to be used more frequently and diversely than negative evaluative language in ordinary communication (Boucher & Osgood, 1969; Matlin & Stang, 1978).
- Add a thesis-specific interpretation: Lemotif asked participants to reflect on salient aspects of their previous day, which can plausibly elicit ordinary-life positive reflection rather than clinical distress narratives.
- Add a second interpretation: crowdsourced reflective data may select for short, socially acceptable, easily narratable events, which can reduce the frequency of rare negative emotions such as Jealous, Afraid, Disgusted, Ashamed, and Awkward.
- Use Figure 2 to compare Lemotif and Stories. State that Stories is less positive and contains more neutral/negative reflections, creating a harder external generalization setting.
- End by previewing that high derived-sentiment performance on Lemotif should not be assumed to transfer directly to Stories.

Suggested figure captions:

```latex
\caption{Emotion label prevalence in the Lemotif dataset. The distribution shows strong positive-affect dominance, which is relevant for interpreting macro-F1 and rare-label performance.}
```

```latex
\caption{Emotion label prevalence across Lemotif and Stories. The Stories test set is less positive and more affectively mixed, indicating a distribution shift between training and external evaluation data.}
```

Appendix links:

- Appendix A, Figure A1: word count distribution.
- Appendix A, Figure A2: topic prevalence.
- Appendix A, Figure A3: topic-emotion heatmap.
- Appendix A, Figure A4: emotion co-occurrence network.

### 5.2 In-Domain Emotion Classification on Lemotif

Purpose:

- Report the main BERT baseline.
- Distinguish multi-label emotion performance from derived sentiment performance.
- Identify the strongest and weakest labels.

Main model:

- Fine-tuned `bert-base-uncased` multi-label emotion classifier.
- Held-out Lemotif test rows: `221`.
- Emotion micro-F1: `0.502`.
- Emotion macro-F1: `0.319`.
- Exact subset accuracy: `0.009`.
- Hamming loss: `0.177`.
- Derived sentiment accuracy: `0.914`.
- Derived sentiment macro-F1: `0.652`.
- Average predicted labels per entry: `4.330`.
- Validation-selected threshold: `0.55`.

Best-supported emotion labels:

- Happy: F1 `0.766`, support `n = 120`.
- Frustrated: F1 `0.750`, support `n = 28`.
- Satisfied: F1 `0.607`, support `n = 86`.

Weakest nonzero-support labels:

- Afraid: F1 `0.000`, support `n = 1`.
- Jealous: F1 `0.000`, support `n = 1`.
- Disgusted: F1 `0.091`, support `n = 2`.

Main figures:

- Figure 3: `fig03_bert_validation_test_metrics.png`
- Figure 4: `fig04_bert_per_emotion_f1.png`
- Figure 5: `fig05_sentiment_confusion_matrix.png`

Bullet outline:

- Introduce BERT as the primary in-domain model because BERT is a transformer language model that can be fine-tuned for downstream classification tasks with limited task-specific architecture changes (Devlin et al., 2019; Vaswani et al., 2017).
- Report the headline statistics in text and in Table 1.
- Interpret micro-F1 and macro-F1 separately. Micro-F1 reflects the model's average label-level decisions across all labels, while macro-F1 exposes poor performance on rare labels.
- Explain why exact subset accuracy is low: in multi-label emotion classification, a prediction is counted correct only when the full predicted label set exactly matches the gold set; this is strict when entries can carry multiple emotions.
- Use Figure 4 to identify that common positive and moderately frequent negative labels are more reliable than sparse labels.
- Use Figure 5 to argue that derived sentiment is more stable than fine-grained emotion classification because multiple emotion labels collapse into broader positive, neutral, and negative categories.
- Avoid overclaiming clinical usefulness. The result supports in-domain affect summarization, not deployment-ready diagnostic inference.

Suggested compact table:

```latex
\begin{table}[htbp]
\centering
\caption{In-domain Lemotif test performance for the main BERT emotion classifier.}
\label{tab:lemotif_bert_main}
\begin{tabular}{lrrrrrr}
\toprule
Model & Micro-F1 & Macro-F1 & Subset acc. & Hamming loss & Sent. acc. & Sent. macro-F1 \\
\midrule
BERT baseline & .502 & .319 & .009 & .177 & .914 & .652 \\
\bottomrule
\end{tabular}
\end{table}
```

Suggested captions:

```latex
\caption{Validation and held-out test metrics for the in-domain BERT emotion classifier.}
```

```latex
\caption{Per-emotion F1 scores on the Lemotif held-out test set. Frequent labels such as Happy and Satisfied are detected more reliably than sparse labels such as Afraid and Jealous.}
```

```latex
\caption{Confusion matrix for derived sentiment on the Lemotif held-out test set. Derived sentiment is substantially more reliable than exact multi-label emotion prediction.}
```

### 5.3 Model Variants and Inference Tradeoffs

Purpose:

- Report Optuna and fixed-inference variants without overstating them as the main model.
- Show the central tradeoff: maximizing emotion F1 is not identical to producing sparse, plausible emotion sets.

Main statistics:

| Model | Emotion micro-F1 | Emotion macro-F1 | Subset acc. | Hamming loss | Sentiment acc. | Avg. predicted labels |
|---|---:|---:|---:|---:|---:|---:|
| BERT baseline | .502 | .319 | .009 | .177 | .914 | 4.330 |
| BERT Optuna | .566 | .299 | .072 | .127 | .923 | 3.172 |
| BERT fixed inference | .445 | .258 | .109 | .130 | .941 | 2.127 |

Main figure:

- Figure 6: `fig06_fixed_inference_cardinality.png`

Bullet outline:

- Present Optuna as a hyperparameter-tuned model variant, not the main baseline. It increases micro-F1 to `.566` and hamming loss improves to `.127`, but macro-F1 remains low at `.299`, showing that tuning does not solve rare-label weakness.
- Present fixed inference as a post hoc threshold/cardinality adjustment. It uses unchanged fine-tuned BERT weights, threshold `.59`, and a minimum one-label fallback.
- State explicitly that the fixed-inference segment is not a newly trained model.
- Explain the tradeoff: fixed inference reduces overprediction and improves sentiment accuracy from `.914` to `.941`, but emotion micro-F1 decreases from `.502` to `.445`.
- Report the most thesis-relevant practical result: average predicted labels decrease from `4.330` to `2.127`; rows with zero predicted emotions decrease from `9` to `0`; rows with at least two extra predicted emotions decrease from `153` to `27`.
- Interpret this as a calibration tradeoff useful for readable summarization, not as the best fine-grained emotion classifier.
- Put threshold sweeps, bootstrap CIs, and calibration plots in Appendix B/C.

Suggested table:

```latex
\begin{table}[htbp]
\centering
\caption{Comparison of BERT model variants on the Lemotif held-out test set.}
\label{tab:bert_variants}
\begin{tabular}{lrrrrrr}
\toprule
Model & Micro-F1 & Macro-F1 & Subset acc. & Hamming loss & Sent. acc. & Avg. labels \\
\midrule
BERT baseline & .502 & .319 & .009 & .177 & .914 & 4.330 \\
BERT Optuna & .566 & .299 & .072 & .127 & .923 & 3.172 \\
BERT fixed inference & .445 & .258 & .109 & .130 & .941 & 2.127 \\
\bottomrule
\end{tabular}
\end{table}
```

Suggested caption:

```latex
\caption{Prediction-cardinality comparison for true labels, baseline BERT predictions, and fixed-inference BERT predictions. The adjusted inference setting reduces label overestimation but lowers fine-grained emotion F1.}
```

### 5.4 Error Analysis and Interpretability

Purpose:

- Fulfill the rubric requirement for error-pattern analysis.
- Replace unsupported SHAP wording with supported LIME and feature-ablation diagnostics.
- Explain the failure modes behind the aggregate metrics.

Use:

- Figure 4: per-emotion F1.
- Figure 5: sentiment confusion matrix.
- Appendix C1/C2: rare-label and multilabel error diagnostics.
- Appendix C4: feature-ablation diagnostics.
- Appendix C5: LIME representative panel.

Bullet outline:

- Start with the core error pattern: the model performs better on frequent and affectively clear labels than rare or ambiguous labels.
- Discuss rare labels as a data limitation rather than purely a model limitation. Afraid and Jealous have only one held-out example each, making stable estimation impossible.
- Discuss multi-label overlap. Reflections can contain blended emotions, so prediction mistakes often involve missing one label while predicting a semantically adjacent label.
- Discuss derived sentiment errors. The model handles broad positive/negative valence better than exact emotion identity, but neutral remains weaker because neutral reflections can contain low-intensity affect or mixed signals.
- Use LIME carefully: LIME explains local predictions by fitting interpretable surrogate models around individual predictions (Ribeiro et al., 2016). Do not treat LIME as a global causal explanation.
- Use feature-ablation diagnostics carefully: describe them as lightweight diagnostic checks, not as definitive feature importance.

Suggested main-text sentence:

The interpretability analyses therefore support a conservative reading of the classifier: BERT learned broad valence and common emotion categories, but rare-label performance and blended-affect errors limit its suitability for exact emotion-set prediction.

Appendix placement:

- Put the LIME representative panel in Appendix C unless Results has enough space.
- Put all large error tables in Appendix C.

### 5.5 External Validation on Stories

Purpose:

- Answer the generalization part of RQ1.
- Show that the model transfers imperfectly from Lemotif to Stories.
- Compare trained BERT against untrained BERT and Gemma sentiment.

Main statistics:

Stories BERT external emotion evaluation:

- Test rows: `114`.
- Emotion micro-F1: `0.347`.
- Emotion macro-F1: `0.246`.
- Subset accuracy: `0.018`.
- Hamming loss: `0.167`.
- Derived sentiment accuracy: `0.544`.
- Derived sentiment macro-F1: `0.490`.

Trained vs. untrained BERT on Stories:

- Emotion micro-F1: trained `0.347` vs. untrained `0.254`.
- Emotion macro-F1: trained `0.246` vs. untrained `0.038`.
- Sentiment accuracy: trained `0.544` vs. untrained `0.140`.
- Sentiment macro-F1: trained `0.490` vs. untrained `0.097`.

Gemma Stories sentiment:

- Sentiment accuracy: `0.798`.
- Sentiment macro-F1: `0.674`.
- Average latency: `20.50` seconds per example.

Main figures:

- Figure 7: `fig07_stories_external_model_drop.png`
- Figure 8: `fig08_stories_per_emotion_f1.png`
- Figure 9: `fig09_stories_sentiment_model_comparison.png`

Bullet outline:

- Open by saying Stories is the key out-of-domain, human-labeled test of the Lemotif-trained emotion model.
- Report the performance drop from Lemotif to Stories and explain it as domain shift: Stories entries differ in length, context, tone, source population, and distribution of sentiment labels.
- Use Figure 7 to show the global drop.
- Use Figure 8 to show which emotions generalize better: Happy, Anxious, and Satisfied are comparatively stronger; Ashamed, Awkward, and Angry are weak in the current artifact.
- Compare trained vs. untrained BERT to show that Lemotif supervision matters even though external performance is limited.
- Introduce Gemma sentiment as a separate sentiment-only comparison. It is not an 18-label emotion comparison on Stories; it answers whether an instruction-tuned model can better recover coarse sentiment from Stories.
- Explain that Gemma's higher sentiment accuracy suggests that broad sentiment may be easier to transfer than the complete Lemotif emotion taxonomy.

Suggested table:

```latex
\begin{table}[htbp]
\centering
\caption{External Stories test performance.}
\label{tab:stories_external}
\begin{tabular}{lrrrr}
\toprule
Model/task & Emotion micro-F1 & Emotion macro-F1 & Sent. acc. & Sent. macro-F1 \\
\midrule
Trained BERT emotion model & .347 & .246 & .544 & .490 \\
Untrained BERT control & .254 & .038 & .140 & .097 \\
Gemma sentiment only & -- & -- & .798 & .674 \\
\bottomrule
\end{tabular}
\end{table}
```

Suggested captions:

```latex
\caption{Performance comparison between the in-domain Lemotif held-out test and the external Stories test. The drop in performance indicates substantial domain shift.}
```

```latex
\caption{Per-emotion F1 scores for the external Stories evaluation. The model generalizes better for frequent or affectively clear categories than for sparse or ambiguous labels.}
```

```latex
\caption{Sentiment comparison on Stories for trained BERT, untrained BERT, and Gemma. Gemma is evaluated as a sentiment-only model and should not be interpreted as an 18-label emotion classifier in this comparison.}
```

### 5.6 WordPress Public-Journal Descriptive Analysis

Purpose:

- Answer RQ2 cautiously.
- Use WordPress only as ungold-labeled descriptive inference.
- Show whether model outputs on public journals collapse into Lemotif's positive skew.

Main statistics:

- Requested sites: `72`.
- Successful sites: `72`.
- Reflections analyzed: `10,671`.
- Window: last `365` days.
- Per-site cap: `365` reflections.
- Top inferred theme: God (`11.0%`).
- Top inferred emotion: Anxious (`48.9%`).

Main figure:

- Figure 10: `fig10_wordpress_theme_emotion_heatmap.png`

Bullet outline:

- Start with the strongest caveat: WordPress posts do not have human Lemotif labels, so this section cannot evaluate accuracy, F1, or true bias.
- State that the section analyzes model outputs descriptively to check whether inferred labels are plausible and whether the model simply reproduces the positive Lemotif training distribution.
- Report that public-journal predictions are not overwhelmingly positive; the strongest inferred emotions include Anxious, Frustrated, Surprised, Sad, Angry, Confused, and Happy.
- Use Figure 10 to explain theme-emotion structure. Health, Sleep, and Work are especially associated with anxious/frustrated outputs, while God and Food show relatively more positive activation.
- Discuss possible skews cautiously:
  - Crawler/tag selection may over-sample active public bloggers and religious/personal-reflection sites.
  - Public self-presentation may differ from private journaling or clinical reflections.
  - Domain shift may cause BERT to over-activate high-arousal negative labels in long public narratives.
  - Model transfer bias may reflect Lemotif supervision rather than true WordPress emotional prevalence.
  - Negative entries may contain richer affective density, increasing the number of predicted labels.
- End with the key RQ2 answer: WordPress inference does not formally validate the model, but it descriptively suggests the model did not merely copy Lemotif's positive-class dominance.

Suggested caption:

```latex
\caption{Inferred theme-emotion conditional prevalence in the public WordPress journal corpus. Because the corpus is not gold-labeled, this figure is interpreted descriptively rather than as model evaluation.}
```

Appendix links:

- Appendix D1: WordPress emotion prevalence.
- Appendix D2: WordPress emotion correlation matrix.
- Appendix D3: WordPress emotion co-occurrence network.

### 5.7 Results Summary by Research Question

Purpose:

- Close the Results chapter by answering RQ1 and RQ2 without drifting into full Discussion.

Bullet outline:

- RQ1, in-domain: BERT learned the Lemotif label space moderately well at the multi-label level and strongly at the derived sentiment level.
- RQ1, external Stories: performance dropped substantially, indicating that Lemotif-trained emotion classification does not yet generalize robustly to Stories reflections.
- RQ1, model comparison: Optuna improved micro-F1, fixed inference improved label-density readability and sentiment accuracy, and Gemma improved Stories sentiment but was not a complete replacement for emotion taxonomy classification.
- RQ2, WordPress: the descriptive WordPress analysis suggests non-collapse into positive predictions, but no formal validation or bias metric is possible without gold labels.
- Bridge to Discussion: the findings support sentiment-level summarization as the stronger near-term use case, while exact multi-label emotion prediction requires more Stories-specific annotations and stronger rare-label handling.

Suggested closing paragraph:

Overall, the results support a cautious interpretation of automatic affect analysis for reflective writing. The Lemotif-trained BERT model captured broad affective patterns and performed especially well when fine-grained emotions were collapsed into derived sentiment. However, the external Stories evaluation showed that exact Lemotif-style emotion classification remained sensitive to domain shift, label sparsity, and multi-label ambiguity. The WordPress analysis extended the pipeline to a larger public-journal setting, but only as descriptive evidence because the corpus was not human-labeled.

## 3. Methods Alignment Checklist

Revise the Methods section before writing final Results prose. The current compiled thesis draft contains future-tense and unsupported planned analyses that should be corrected.

### Replace Future-Tense Planning With Completed Pipeline

Change:

- "The models will be trained..."
- "The Stories evaluation dataset currently has no emotion labels..."
- "Two researchers will manually annotate..."
- "The study conducts a feature and context ablation analysis using SHAP..."
- "The second model will be a longitudinal variant..."

To:

- The analysis trained/evaluated a single-entry BERT emotion classifier on Lemotif.
- The analysis evaluated external generalization on a human-labeled Stories test artifact of 114 rows.
- The analysis compared trained BERT against an untrained BERT control and Gemma sentiment output.
- The analysis used LIME and feature-ablation diagnostics, not SHAP.
- Longitudinal/user-history modeling, weak-label generation, and formal inter-annotator reliability are outside the completed result scope unless new artifacts are added.

### Methods Section Rewrite Targets

Suggested revised Methods subsections:

1. **4.1 Data Sources**
   - Lemotif: 1,473 human-annotated reflective entries, 18 emotions, 11 topics.
   - Stories: external human-labeled test set used for out-of-domain evaluation.
   - WordPress: public-journal corpus used for descriptive inference only.

2. **4.2 Preprocessing and Label Mapping**
   - Text cleaning, label column detection, and sentiment derivation from emotion labels.
   - State that sentiment is derived analytically from emotion predictions, not trained as a separate BERT emotion output.

3. **4.3 Models**
   - Fine-tuned BERT emotion classifier.
   - Optuna BERT hyperparameter variant.
   - Fixed-inference BERT threshold/cardinality variant.
   - Untrained BERT control.
   - Gemma emotion and sentiment comparisons.
   - WordPress theme BERT model.

4. **4.4 Evaluation Design**
   - Lemotif held-out test for in-domain performance.
   - Stories external test for generalization.
   - WordPress descriptive inference without gold-label metrics.

5. **4.5 Metrics and Diagnostics**
   - Micro-F1, macro-F1, subset accuracy, hamming loss, sentiment accuracy, sentiment macro-F1.
   - Per-label F1 and confusion matrices for error patterns.
   - Bootstrap confidence intervals, threshold calibration, rare-label errors, feature-ablation diagnostics, and LIME in appendices.

6. **4.6 Reproducibility**
   - Point to the codebase and artifact folders.
   - State that fixed inference changes threshold/minimum-label logic only and does not retrain weights.

### Unsupported Claims to Remove or Move to Future Work

- SHAP analysis.
- Longitudinal or user-history model.
- Weak-label generation as a completed training source.
- Formal WordPress validation metrics.
- WordPress F1, accuracy, or bias score.
- Krippendorff's alpha/Cohen's kappa unless a separate annotator-agreement artifact is supplied.
- Claims that the current model is deployment-ready for clinical prediagnostics.

## 4. Overleaf Snippets

Use this generic figure pattern. Adjust `width` if a figure is too tall.

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/results/main_text/fig01_lemotif_emotion_prevalence.png}
    \caption{Emotion label prevalence in the Lemotif dataset. The distribution shows strong positive-affect dominance, which is relevant for interpreting macro-F1 and rare-label performance.}
    \label{fig:lemotif_emotion_prevalence}
\end{figure}
```

Recommended labels:

```latex
\label{fig:lemotif_emotion_prevalence}
\label{fig:lemotif_stories_distribution_shift}
\label{fig:bert_validation_test_metrics}
\label{fig:bert_per_emotion_f1}
\label{fig:sentiment_confusion_matrix}
\label{fig:fixed_inference_cardinality}
\label{fig:stories_external_model_drop}
\label{fig:stories_per_emotion_f1}
\label{fig:stories_sentiment_model_comparison}
\label{fig:wordpress_theme_emotion_heatmap}
```

In-text citation examples:

- BERT/transformer methods: `\parencite{vaswani2017attention,devlin2019bert}` or `\citep{vaswani2017attention,devlin2019bert}`.
- Pollyanna explanation: `\parencite{boucher1969pollyanna,matlin1978pollyanna}` or `\citep{boucher1969pollyanna,matlin1978pollyanna}`.
- LIME: `\parencite{ribeiro2016why}` or `\citep{ribeiro2016why}`.

Use the citation command already used by your Overleaf template. If the template uses `biblatex-apa`, prefer `\parencite{}`. If it uses `natbib` or `apacite`, prefer `\citep{}`.

## 5. Citation Anchors

Add the companion BibTeX entries from:

`docs/results_references_apa7.bib`

Key citation uses:

- Pollyanna explanation: Boucher and Osgood (1969); Matlin and Stang (1978).
- Transformer/BERT basis: Vaswani et al. (2017); Devlin et al. (2019).
- Emotion benchmark context: Demszky et al. (2020).
- Interpretability and tooling: Ribeiro et al. (2016); Pedregosa et al. (2011); Akiba et al. (2019).
- Gemma comparison: Gemma Team et al. (2024).

## 6. First-Draft Submission Checklist

### Before Writing Results

- Upload `overleaf_results_figures` to Overleaf as `figures/results`.
- Add or merge `results_references_apa7.bib` entries into the thesis `.bib` file.
- Revise Methods so it matches verified-only artifacts.
- Remove SHAP/longitudinal/weak-label/formal WordPress validation claims from completed methodology.
- Decide whether Figure 2 should remain emotion distribution only or whether the sentiment distribution comparison should be added as a second panel/appendix figure.

### While Writing Results

- Cite every figure in the text before or immediately after it appears.
- Interpret each figure; do not simply restate axes.
- Keep WordPress language descriptive.
- Keep model comparison grounded in metrics tables.
- Put large diagnostic figures and long tables in appendices.
- Use decimal points and three decimals for performance metrics.

### Before Sending First Draft to Supervisor

- Compile Overleaf and check that all figures render.
- Check that all references compile in APA7 style.
- Check that table captions and figure captions are self-contained.
- Search the thesis for unsupported terms: `SHAP`, `longitudinal`, `weak labels`, `Krippendorff`, `Cohen`, `WordPress validation`, `accuracy on WordPress`, `bias metric`.
- Confirm the word count remains within the 8,000 words +/- 10% requirement excluding front matter, references, and appendices.
- Add appendix cross-references for all non-main diagnostics.
- Update the DSECT statement to mention WordPress public posts, code repository/artifacts, AI writing assistance, and self-made figures.
- Ensure the Discussion does not introduce new results that were not reported in Results.

## 7. Source Artifact Trace

Primary evidence:

- `thesis_findings_vscode/primary_findings_codebase/output/findings/primary_findings.md`
- `thesis_findings_vscode/secondary_findings_codebase/output/findings/secondary_findings.md`
- `output/bert_emotion_model/metrics.json`
- `output/BERT-Hyperparamter-Fixed/metrics.json`
- `output/bert_emotion_model_optuna/metrics.json`
- `test data/output/external_evaluation/metrics.json`
- `test data/output/untrained_bert/metrics.json`
- `test data/output/gemma_sentiment/google_gemma-3n-E2B-it/metrics.json`
- `thesis_findings_vscode/wordpress_findings_codebase/output/selected_site_analysis/report/selected_site_summary.md`
- `thesis_findings_vscode/wordpress_findings_codebase/output/selected_site_analysis/run_summary.json`

Do not report claims that cannot be traced to these artifacts or a clearly cited literature source.
