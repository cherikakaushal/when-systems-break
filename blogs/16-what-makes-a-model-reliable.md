# What Makes a Model Reliable?

Is accuracy enough?

No.

A model can be accurate on clean data and still fail under noise, missing information, distribution shift, or misplaced confidence. Reliability is multi-dimensional.

## Experiment 16

The Reliability Index combines six measurements already produced across the project:

| Component | Weight | Meaning |
|---|---:|---|
| Accuracy | 25% | Mean clean accuracy across 30 seeds |
| Noise Robustness | 20% | Mean noisy accuracy across 30 seeds |
| Missing Data | 15% | Accuracy with missing inputs |
| Calibration | 15% | Agreement between confidence and correctness |
| Distribution Shift | 15% | Accuracy at the largest simulated shift |
| Confidence | 10% | Confidence-accuracy alignment across shift levels |

```text
Reliability = 0.25(Accuracy)
            + 0.20(Noise Robustness)
            + 0.15(Missing Data)
            + 0.15(Calibration)
            + 0.15(Distribution Shift)
            + 0.10(Confidence)
```

## Results

![Model Reliability Index](../figures/reliability_index.png)

| Model | Reliability Index |
|---|---:|
| SVM | 96.21 |
| Logistic Regression | 95.99 |
| Random Forest | 94.82 |
| Decision Tree | 89.42 |

SVM ranks first, but only narrowly. It leads Logistic Regression through stronger missing-data performance, endpoint shift accuracy, and confidence alignment. Logistic Regression performs better on clean accuracy, noisy accuracy, and calibration.

Random Forest has the strongest distribution-shift component at 87.22, but lower calibration and noisy accuracy keep it in third place. Decision Tree ranks last because its missing-data, calibration, shift, and confidence components all expose weaknesses hidden by a respectable 92.92 clean-accuracy score.

## Why Keep the Components?

A composite index is useful for ranking. It is dangerous when the ranking replaces the evidence beneath it.

Two models can receive nearly identical totals for different reasons. The component heatmap therefore matters as much as the headline number: it shows which kinds of reliability each model provides and where each one remains vulnerable.

## Reliability Index vs Reliability Score

Experiment 14's Model Reliability Score includes refusal quality and repeatability inside a unified 30-seed benchmark. Experiment 16's Reliability Index synthesizes six outputs from separate experiments, adding missing-data behavior, calibration, and distribution-shift resistance.

The two metrics answer related but different questions. Neither is a universal safety rating.

## Answer

Accuracy is necessary, but it is not enough.

The best overall model is the one that remains accurate across changing conditions, expresses confidence honestly, and preserves performance when the assumptions behind deployment begin to move.
