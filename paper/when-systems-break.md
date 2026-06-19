# When Systems Break:
# Understanding Model Failure Under Noise, Missing Data, Feature Degradation, and Refusal Thresholds

## 1. Abstract

Machine learning systems are commonly evaluated using clean-data accuracy, but deployed models often operate under degraded input conditions. Data can become noisy, incomplete, corrupted, or partially unavailable, causing models to behave differently from their reported benchmark performance. This paper studies how supervised learning models respond when the assumptions behind their inputs begin to fail.

Using a controlled tabular classification setting, twelve experiments evaluate model behavior under noise injection, missing values, feature removal, repeated random seeds, confidence degradation, and refusal thresholds. The results show that model failure is often gradual rather than immediate. Accuracy can remain stable at low degradation levels before declining more sharply, and different algorithms exhibit different robustness patterns. Confidence scores also provide useful signals, but they must be interpreted alongside coverage and refusal rate.

The central finding is that robustness evaluation should not stop at measuring failure. A stronger system should detect uncertainty and respond to it. Refusal-based reliability provides one such response by allowing a model to abstain from low-confidence predictions.

## 2. Introduction

Most machine learning projects emphasize improving accuracy under ideal conditions. This is useful, but it is incomplete. Real-world systems rarely receive perfectly clean inputs. Features may be missing, sensors may produce noise, users may enter unusual values, and deployment data may differ from training data.

This raises a practical reliability question:

What happens when a machine learning system is asked to make predictions under degraded information?

The goal of this project is to study model failure as a measurable process. Instead of treating failure as a single event, the experiments examine how performance changes as input quality declines. The project also studies whether confidence scores can help identify risky predictions before the system fails visibly.

The research story progresses through three stages:

1. Models fail under degraded information.

2. Some failures can be detected through accuracy, confidence, and variance.

3. Systems can respond by refusing predictions when reliability is too low.

## 3. Experimental Setup

All experiments were implemented in Python using a controlled classification workflow.

**Language:** Python

**Libraries:**

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

**Models:**

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine

**Evaluation Metrics:**

- Accuracy
- Mean accuracy across repeated runs
- Standard deviation across repeated runs
- Prediction confidence using `predict_proba()`
- Coverage
- Refusal rate
- Feature importance

The experiments use clean data as a baseline and then introduce controlled degradation through noise injection, missing values, feature removal, and confidence thresholds. Later experiments compare multiple models and summarize failure patterns in aggregate visualizations.

## 4. Noise Analysis

Noise analysis evaluates how model accuracy changes when random perturbations are added to the input features. This simulates real-world situations where measurements are imprecise, corrupted, or affected by environmental variation.

![Noise Curve](figures/noise_curve.png)

The noise curve shows that degradation is not always immediate. At lower noise levels, the model may continue to perform well. As noise increases, accuracy begins to decline more noticeably. This suggests that model failure can be non-linear: systems may appear stable until a reliability threshold is crossed.

Threshold analysis provides a second view of this behavior.

![Threshold Analysis](figures/threshold_analysis.png)

Together, these results show why robustness testing should evaluate performance across multiple degradation levels rather than relying on a single noisy condition.

## 5. Missing Data Analysis

Missing data analysis studies how the system behaves when part of the input information is unavailable. In real deployments, missing values can come from incomplete forms, sensor failures, data pipeline issues, or unavailable external signals.

Missing information differs from random noise. Noise distorts existing information, while missingness removes information entirely. This can be more harmful when the missing features are important for prediction.

The project also studies feature importance to identify which inputs contribute most strongly to model behavior.

![Feature Importance](figures/feature_importance.png)

The feature-importance results show that not all features contribute equally. A small subset of features has a larger effect on model predictions. This means that removing or corrupting important features can produce sharper performance drops than degrading less influential features.

## 6. Confidence Collapse

Confidence collapse asks whether a model becomes less certain as inputs become less reliable.

The experiment uses `predict_proba()` to track three signals as noise increases:

- accuracy
- confidence in the correct class
- number of wrong predictions

![Confidence Collapse](figures/confidence_collapse.png)

The results show that wrong predictions become more frequent as noise increases. Confidence in the correct class also declines, but the model can still assign high confidence to some incorrect predictions. This is an important reliability risk: a wrong but uncertain model is easier to manage than a wrong and confident one.

Confidence scores are therefore useful, but they should not be treated as perfect guarantees. They are signals that can support reliability decisions, especially when combined with degradation tests and refusal thresholds.

## 7. Refusal-Based Reliability

Refusal-based reliability addresses the question:

When should a machine learning system stop trusting itself?

Standard classifiers usually predict every time:

```text
prediction = model.predict(X)
confidence = max(model.predict_proba(X))
```

A refusal-aware system adds a safety rule:

```text
if confidence < threshold:
    prediction = "REFUSE"
```

This creates a tradeoff between accuracy and coverage. Coverage measures how often the model is willing to make a prediction. Accuracy measures how often accepted predictions are correct. As the confidence threshold increases, the model refuses more low-confidence examples. This can improve the quality of accepted predictions while reducing how often the system answers.

![Accuracy vs Coverage](figures/accuracy_vs_coverage.png)

The refusal experiment evaluates thresholds from 0.50 to 0.90 and stores threshold, coverage, accuracy, refusal rate, accepted predictions, and refused predictions. This turns robustness analysis into a practical system-design question: should the model answer, or should it defer?

## 8. Failure Matrix

The failure matrix compares multiple models across multiple degradation conditions in one visualization. It evaluates clean inputs, noisy inputs, missing data, and feature-removal conditions.

![Failure Matrix](figures/failure_matrix.png)

The matrix shows that failure behavior is not uniform. Some models remain more stable under noise, while others are more affected by missing data or feature removal. This supports the idea that robustness is model-specific and condition-specific.

The project also includes a broader failure-pattern comparison.

![Failure Comparison](figures/failure_comparison.png)

Together, these visualizations provide a compact view of how different forms of degradation affect model reliability.

## 9. Key Findings

1. Clean-data accuracy is not enough to evaluate reliability.

2. Noise does not always cause immediate failure.

3. Performance degradation often appears gradually before a sharper decline.

4. Missing information and feature removal can be more harmful when important features are affected.

5. Different algorithms exhibit different robustness characteristics.

6. Repeated runs reveal variance that a single accuracy score hides.

7. Confidence scores can act as early warning signals, but they are not perfect.

8. Refusal thresholds expose a tradeoff between accepted-prediction accuracy and coverage.

9. A safer machine learning system should know when not to answer.

## 10. Limitations

The experiments were conducted on a limited tabular dataset in a controlled environment. This makes failure patterns easier to isolate, but it does not capture the full complexity of production machine learning systems.

The degradation methods are also simplified. Gaussian noise, missing-value simulation, and feature removal are useful controlled tests, but real-world failure modes may include distribution shift, adversarial perturbations, data drift, semantic corruption, and feedback loops.

Confidence scores are model-dependent and may require calibration before being used as operational reliability signals. The refusal system is therefore a prototype for studying abstention behavior, not a complete production safety mechanism.

## 11. Future Work

Future work includes:

- Explainable AI using SHAP
- Confidence calibration experiments
- Semantic noise analysis
- Distribution shift experiments
- Adversarial perturbation testing
- Human-in-the-loop review workflows
- Cost-sensitive refusal policies
- Interactive robustness dashboard expansion

## 12. References

1. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.

2. McKinney, W. Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 2010.

3. Harris, C. R. et al. Array programming with NumPy. Nature, 2020.

4. Hunter, J. D. Matplotlib: A 2D Graphics Environment. Computing in Science and Engineering, 2007.

5. Geurts, P., Ernst, D., and Wehenkel, L. Extremely randomized trees. Machine Learning, 2006.

6. Cortes, C. and Vapnik, V. Support-vector networks. Machine Learning, 1995.

7. Geifman, Y. and El-Yaniv, R. Selective Classification for Deep Neural Networks. Advances in Neural Information Processing Systems, 2017.
