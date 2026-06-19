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

## Interpretation

A robust model should not only perform well on clean data. It should also degrade predictably, show low variance across repeated runs, reduce confidence when input quality becomes unreliable, and refuse predictions when confidence falls below a reliability threshold.
