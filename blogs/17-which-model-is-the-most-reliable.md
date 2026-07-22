# Which Model Is the Most Reliable?

The most accurate model is not automatically the most reliable model.

Accuracy asks one question:

```text
How often is the model correct on this dataset?
```

Reliability asks a broader question:

```text
How well does the model keep working when conditions change?
```

That difference matters. A model can perform well on clean data but become unstable under missing values, noisy measurements, distribution shift, or poor confidence calibration.

## Experiment 17

Experiment 17 ranks every model using the Reliability Index from Experiment 16. The index combines six measurements:

| Component | Meaning |
|---|---|
| Accuracy | Clean-data performance across repeated runs |
| Noise Robustness | Performance after injected noise |
| Missing Data | Performance when information is unavailable |
| Calibration | Whether confidence matches correctness |
| Distribution Shift | Performance when the test distribution changes |
| Confidence | Alignment between confidence and actual accuracy |

The ranking does not replace the component scores. It gives a top-level answer while keeping the evidence auditable.

![Model Ranking](../figures/model_ranking.png)

## Results

| Rank | Model | Reliability Index | Gap to Leader | Tier |
|---:|---|---:|---:|---|
| 1 | SVM | 96.21 | 0.00 | Leader |
| 2 | Logistic Regression | 95.99 | 0.22 | Leader |
| 3 | Random Forest | 94.82 | 1.39 | Competitive |
| 4 | Decision Tree | 89.42 | 6.79 | Needs Review |

SVM ranks first, but the margin is tiny. Logistic Regression is only 0.22 points behind, which means both models are strong under this benchmark.

Random Forest remains competitive, especially because it performs well under distribution shift. Decision Tree ranks last because several reliability signals expose weaknesses that clean accuracy alone does not show.

## Why Accuracy Alone Can Mislead

Imagine two models with similar clean accuracy.

One model stays stable when values are noisy, produces calibrated confidence, and loses only a little performance under shift.

The other model is accurate on clean data but becomes overconfident, brittle, or inconsistent when the input changes.

Both may look similar in a basic accuracy table. They do not carry the same deployment risk.

## The Research Lesson

The best model is not just the one with the highest clean score.

The best model is the one with the strongest reliability profile across multiple failure modes. Experiment 17 makes that comparison visible by ranking models after accuracy, robustness, calibration, confidence, and distribution-shift behavior have all been considered.
