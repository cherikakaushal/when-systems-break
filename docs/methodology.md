# Methodology

## Dataset

The experiments use the breast cancer dataset from scikit-learn. It provides numeric tabular features and binary labels, making it useful for controlled robustness testing.

## Models

The project evaluates:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine

Some experiments focus on one model to isolate a failure pattern. Later experiments compare multiple models under the same degradation conditions.

## Degradation Conditions

The project studies three main types of input degradation:

- noise injection
- missing values
- feature removal

The later confidence-collapse experiment also studies how predicted probabilities change as input quality declines.

## Metrics

The primary metrics are:

- clean accuracy
- noisy accuracy
- missing-data accuracy
- feature-removal accuracy
- mean accuracy across seeds
- standard deviation across seeds
- prediction confidence
- wrong prediction count
- coverage
- refusal rate
- Model Reliability Score and its five component scores

## Reliability Score Framework

Experiment 14 proposes a composite score for comparing models within this controlled benchmark:

```text
Reliability = 0.30(Accuracy)
            + 0.25(Robustness)
            + 0.15(Confidence Stability)
            + 0.20(Refusal Quality)
            + 0.10(Repeatability)
```

- **Accuracy** is mean clean-data accuracy across 30 seeded splits.
- **Robustness** is mean degraded accuracy divided by mean clean accuracy, capped at 100.
- **Confidence Stability** is one minus the mean absolute gap between confidence and accuracy across clean, noisy, missing, and feature-removal conditions.
- **Refusal Quality** is the ROC AUC obtained when prediction confidence ranks correct predictions above incorrect predictions. It measures whether confidence can support selective refusal.
- **Repeatability** penalizes mean run-to-run standard deviation against a declared five-percentage-point tolerance.

Every component is scaled from 0 to 100. The weights are explicit design choices, not learned parameters. The score is intended for within-project model comparison and has not been externally validated as a general-purpose safety metric.

The generated `reliability_run_metrics.csv` retains each seed, model, condition, accuracy, and mean confidence observation used to build the summary.

## Confidence Calibration

Experiment 13 tests whether stated confidence corresponds to empirical correctness. Predictions from 30 seeded train-test splits are pooled and grouped into ten equal-width confidence bins under clean and noisy inputs.

Expected Calibration Error is calculated as:

```text
ECE = sum((bin count / total count) * abs(bin accuracy - bin confidence))
```

A perfectly calibrated 90% confidence bin should be correct approximately 90% of the time. The experiment exports every populated bin, pooled ECE, mean and standard deviation of run-level ECE, and the observed accuracy of predictions in the 90-100% confidence bin.

Because ECE depends on binning strategy and sample size, it is reported together with the full reliability diagram rather than treated as a sufficient standalone statistic.

## Interpretation

A robust model should not only perform well on clean data. It should also degrade predictably, show low variance across repeated runs, reduce confidence when input quality becomes unreliable, and refuse predictions when confidence falls below a reliability threshold.
