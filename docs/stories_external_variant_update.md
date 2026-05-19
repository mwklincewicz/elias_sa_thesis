# Stories External Variant Update

## Model-selection conclusion

For the Results chapter, report the Optuna-tuned BERT checkpoint as the strongest BERT variant for external Stories generalization. On Stories, it outperformed the baseline and fixed-inference variants on emotion micro-F1 and derived sentiment accuracy:

| Model | Emotion micro-F1 | Emotion macro-F1 | Sentiment accuracy | Sentiment macro-F1 | Avg. predicted labels |
|---|---:|---:|---:|---:|---:|
| BERT baseline checkpoint | .351 | .229 | .500 | .452 | 2.377 |
| BERT fixed inference | .319 | .203 | .561 | .440 | 1.623 |
| BERT Optuna | .414 | .192 | .640 | .457 | 2.114 |
| BERT full-Lemotif external run | .347 | .246 | .544 | .490 | 2.482 |
| Untrained BERT | .254 | .038 | .140 | .097 | 1.956 |
| Gemma sentiment-only | n/a | n/a | .798 | .674 | n/a |

The best interpretation is that Optuna gives the strongest BERT-only external transfer result, while fixed inference remains useful as a calibration/cardinality analysis rather than as the best external emotion classifier.

## Figure 7 replacement

Use `fig07_stories_external_optuna_fixed_comparison.png` as the updated Figure 7. It compares Lemotif held-out performance and Stories external performance for the baseline, fixed-inference, and Optuna BERT variants.

Suggested caption:

> Performance comparison between in-domain Lemotif held-out evaluation and Stories external evaluation for three BERT variants. All BERT variants output the same 18 Lemotif emotion labels; sentiment scores are derived from the predicted emotion labels. The Optuna-tuned BERT checkpoint shows the strongest BERT-only external transfer, particularly for sentiment accuracy, although all variants still show a clear domain-shift penalty.

## Figure 8 replacement

Use `fig08_stories_per_emotion_f1_variants.png` as the updated per-emotion Stories figure.

Suggested caption:

> Per-emotion F1 scores on the Stories external dataset for baseline, fixed-inference, and Optuna BERT variants. The figure reports the full 18-label Lemotif emotion taxonomy where labels are present in Stories. Performance varies strongly by emotion, indicating that external transfer is uneven across the taxonomy.

## Figure 9 replacement

Use `fig09_stories_sentiment_variant_comparison.png` as the updated sentiment-comparison figure.

Suggested caption:

> Sentiment comparison on Stories for BERT variants, untrained BERT, and Gemma. BERT sentiment is derived from 18-label emotion predictions, whereas Gemma is included here as a direct sentiment-only comparator and should not be interpreted as an 18-label emotion classifier in this figure.

## Important clarification about BERT vs. Gemma

The BERT models do report on the full 18-label Lemotif emotion taxonomy. Their sentiment metrics are not a separate sentiment classifier; sentiment is derived after prediction by mapping predicted emotions to positive, negative, or neutral valence.

Gemma was evaluated in two different ways across the project:

- On Lemotif, Gemma has 18-label emotion outputs and derived sentiment metrics in the Gemma emotion-model artifacts.
- On Stories, the current Gemma comparison is sentiment-only. It directly predicts coarse sentiment, so it is valid for the Figure 9 sentiment comparison but not for claims about 18-label emotion classification on Stories.

## Cardinality and over/underreporting

Stories true labels average 2.123 emotions per reflection. The Optuna checkpoint is closest in average cardinality, predicting 2.114 labels per reflection. Fixed inference underpredicts more strongly on Stories, predicting 1.623 labels per reflection.

| Model | Overreported rows | Underreported rows | Same-count rows | False-positive labels | False-negative labels |
|---|---:|---:|---:|---:|---:|
| BERT baseline checkpoint | 43/114 | 52/114 | 19/114 | 181 | 152 |
| BERT fixed inference | 14/114 | 66/114 | 34/114 | 117 | 174 |
| BERT Optuna | 46/114 | 46/114 | 22/114 | 141 | 142 |

Suggested interpretation:

> Fixed inference substantially reduces overreporting but does so by shifting toward underreporting on Stories. Optuna produces the most realistic average label count and the strongest BERT-only sentiment accuracy on Stories, making it the better primary tuned BERT result for the external-validation narrative.

## Files

- Raw summary: `test data/output/stories_model_variants/stories_model_variant_summary.csv`
- Run report: `test data/output/stories_model_variants/stories_model_variant_report.md`
- Workspace figures: `test data/output/stories_model_variants/figures/`
- Desktop Overleaf copies: `C:/Users/ee/Desktop/Graphs/overleaf_results_figures/main_text/`
