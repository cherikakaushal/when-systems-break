# When Systems Break

**An experimental framework for studying machine learning failure under noise, missing data, feature degradation, and confidence collapse.**

[Research Report](paper/when-systems-break.pdf) | [Interactive Demo](app.py) | [Experiments](experiments) | [Blogs](blogs)

![Failure Matrix](figures/failure_matrix.png)

---

## Why This Project Exists

Most machine learning projects report a single clean-data accuracy score.

Real systems are messier. Inputs become noisy, features disappear, missing values appear, and models can stay confident even when they are wrong.

This project studies what happens when those assumptions break.

The goal is not only to build a model. The goal is to understand how models fail, how early those failures can be detected, and when a system should stop trusting its own predictions.

---

## Research Questions

1. How does increasing noise affect model accuracy?
2. Which features contribute most to model stability?
3. Do different algorithms fail differently?
4. Can confidence scores warn us before accuracy collapses?
5. Is missing information more harmful than noisy information?
6. When should a model refuse to make a prediction?

---

## Project Highlights

- Controlled robustness experiments across noise, missing data, and feature removal
- Multi-model comparison using Logistic Regression, Decision Tree, Random Forest, and SVM
- Multi-run statistical analysis across 30 random seeds
- Failure matrix heatmap for model-by-condition comparison
- Confidence collapse study using `predict_proba()`
- Streamlit demo for interactively injecting failure conditions
- Research-style paper with generated figures and PDF export

---

## Key Outputs

| Output | File |
|---|---|
| Research paper | [paper/when-systems-break.pdf](paper/when-systems-break.pdf) |
| Failure matrix heatmap | [figures/failure_matrix.png](figures/failure_matrix.png) |
| Confidence collapse plot | [figures/confidence_collapse.png](figures/confidence_collapse.png) |
| Statistical robustness CSV | [experiments/model_statistics.csv](experiments/model_statistics.csv) |
| Failure matrix CSV | [experiments/failure_matrix.csv](experiments/failure_matrix.csv) |
| Confidence collapse CSV | [experiments/confidence_collapse.csv](experiments/confidence_collapse.csv) |

---

## Experiments

| # | Experiment | File | Purpose |
|---:|---|---|---|
| 01 | Baseline noise injection | [experiment1.py](experiments/experiment1.py) | Compare clean performance with noisy-input performance |
| 02 | Missing data simulation | [experiment2.py](experiments/experiment2.py) | Test how incomplete inputs affect predictions |
| 03 | Feature importance | [experiment3.py](experiments/experiment3.py) | Identify which features influence the model most |
| 04 | Model comparison | [experiment4_model_comparison.py](experiments/experiment4_model_comparison.py) | Compare algorithms under clean, noisy, and missing inputs |
| 05 | Feature removal | [experiment5_feature_removal.py](experiments/experiment5_feature_removal.py) | Measure degradation after removing important features |
| 06 | Noise robustness curve | [experiment6_noise_curve.py](experiments/experiment6_noise_curve.py) | Track accuracy as noise increases |
| 07 | Threshold analysis | [experiment7_threshold_analysis.py](experiments/experiment7_threshold_analysis.py) | Find reliability thresholds under stronger perturbation |
| 08 | Failure comparison | [experiment8_comparison.py](experiments/experiment8_comparison.py) | Compare failure conditions in one degradation curve |
| 09 | Multi-run statistics | [experiment9_multi_run_analysis.py](experiments/experiment9_multi_run_analysis.py) | Estimate mean and variance across 30 random seeds |
| 10 | Failure matrix dashboard | [experiment10_failure_matrix.py](experiments/experiment10_failure_matrix.py) | Generate the model-by-condition heatmap |
| 11 | Confidence collapse | [experiment11_confidence_collapse.py](experiments/experiment11_confidence_collapse.py) | Track confidence decline and wrong predictions under noise |

---

## Blogs

| # | Article |
|---:|---|
| 01 | [What happens when data breaks?](blogs/01-what-happens-when-data-breaks.md) |
| 02 | [What happens when data is missing?](blogs/02-when-data-is-missing.md) |
| 03 | [Which features actually matter?](blogs/03-which-features-matter.md) |
| 04 | [Final insights](blogs/04-final-insights.md) |
| 05 | [Do different models break differently?](blogs/05-model-comparison.md) |
| 06 | [What happens when important features disappear?](blogs/06-when-important-features-break.md) |
| 07 | [When models become overconfident](blogs/07-model-confidence-under-noise.md) |
| 08 | [How robust is a model to increasing noise?](blogs/08-robustness-under-noise.md) |
| 09 | [Failure taxonomy](blogs/09-failure-taxonomy.md) |
| 10 | [Comparing failure patterns](blogs/10-comparing-failure-patterns.md) |
| 11 | [Why one accuracy score is not enough](blogs/11-statistical-robustness.md) |
| 12 | [When models become confidently wrong](blogs/12-confidence-collapse.md) |

---

## Interactive Demo

Run the Streamlit app:

```bash
streamlit run app.py
```

The demo lets a user upload data, inject noise, remove features, create missing values, and inspect:

- prediction
- confidence
- failure risk

---

## Reproduce Results

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the newest research experiments:

```bash
python experiments/experiment9_multi_run_analysis.py
python experiments/experiment10_failure_matrix.py
python experiments/experiment11_confidence_collapse.py
```

---

## Repository Structure

```text
when-systems-break/
├── app.py
├── README.md
├── requirements.txt
├── blogs/
├── data/
├── docs/
├── experiments/
├── figures/
├── notebooks/
└── paper/
```

---

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## Core Insight

Machine learning systems often fail gradually before they fail obviously.

By studying noise, missing information, feature degradation, statistical variance, and confidence collapse, this project shows why robustness testing is as important as clean-data accuracy.
