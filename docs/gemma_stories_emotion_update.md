# Gemma Stories 18-Label Emotion Update

This update reports the full Stories evaluation for two Gemma variants using the 18-label Lemotif emotion taxonomy:

- Gemma zero-shot: the base instruction model prompted to select Lemotif emotions.
- Gemma fine-tuned: the Lemotif LoRA adapter evaluated on the Stories evaluation dataset.

Sentiment metrics in this update are derived from the predicted emotion labels, matching the BERT evaluation procedure. This differs from the earlier Gemma Stories figure, which evaluated Gemma as a direct sentiment-only classifier.

## Main Results

| Model | Emotion micro-F1 | Emotion macro-F1 | Subset accuracy | Hamming loss | Sentiment accuracy | Sentiment macro-F1 | Avg. predicted labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma zero-shot | 0.410 | 0.276 | 0.018 | 0.165 | 0.737 | 0.594 | 2.921 |
| Gemma fine-tuned | 0.463 | 0.267 | 0.035 | 0.146 | 0.807 | 0.593 | 2.763 |

Gold Stories annotations contain an average of 2.123 emotion labels per reflection. Both Gemma variants therefore overpredict label cardinality on average, although the fine-tuned model is closer to the gold distribution than zero-shot Gemma.

## Comparison to BERT Optuna on Stories

The prior BERT Optuna Stories result reported emotion micro-F1 = 0.414, emotion macro-F1 = 0.192, sentiment accuracy = 0.640, sentiment macro-F1 = 0.457, and 2.114 predicted labels per reflection.

Against that benchmark, Gemma fine-tuned on Lemotif improves emotion micro-F1 and derived sentiment accuracy on Stories:

- Emotion micro-F1 improves from 0.414 to 0.463.
- Emotion macro-F1 improves from 0.192 to 0.267.
- Sentiment accuracy improves from 0.640 to 0.807.
- Sentiment macro-F1 improves from 0.457 to 0.593.
- Average predicted labels increase from 2.114 to 2.763, indicating more generous emotion assignment.

## Thesis-Safe Interpretation

The revised Results narrative can now say that Gemma was evaluated in two ways. The earlier direct sentiment prompt tested coarse sentiment only. The new 18-label Stories evaluation tests whether Gemma can assign Lemotif emotion labels and then derive sentiment from those labels, making it directly comparable to BERT on the external Stories evaluation dataset.

The strongest external result is the Lemotif-fine-tuned Gemma model, which outperforms the tuned BERT model on Stories sentiment accuracy and emotion micro-F1. However, this result should be interpreted with two caveats. First, Gemma is substantially slower, averaging about 31 seconds per Stories example in this run. Second, Gemma predicts more emotion labels per reflection than the gold Stories annotations, so part of its recall advantage may come from more liberal label assignment.

## Figure Files

- `test data/output/gemma_emotion_stories/figures/fig_gemma_stories_emotion_sentiment_metrics.png`
- `test data/output/gemma_emotion_stories/figures/fig_gemma_stories_per_emotion_f1.png`
- `test data/output/gemma_emotion_stories/figures/fig_gemma_stories_cardinality.png`

## Suggested Caption

Gemma 18-label emotion evaluation on the Stories external test set. Sentiment metrics are derived from the predicted emotion labels, matching the BERT evaluation procedure. Fine-tuning on Lemotif improves emotion micro-F1, hamming loss, and sentiment accuracy relative to zero-shot Gemma, but both Gemma variants predict more emotion labels per reflection than the gold Stories annotations.
