# What happens when data is missing?

## Problem
In real-world systems, missing data is extremely common.

Sensors fail, inputs are incomplete, and users skip fields.

I wanted to understand:
How does a machine learning model behave when some of its input data is missing?

---

## Approach
Using the same dataset and model as before, I introduced missing values into the test data.

- Randomly removed ~10% of input values  
- Filled missing values using mean imputation  
- Evaluated model performance again  

---

## Observations (experiment)

- Baseline accuracy: ~0.96  
- With missing data: ~0.90  
- Drop in accuracy: ~0.05  

The effect was consistent across runs.

Unlike the noise experiment, missing data clearly degraded performance.

---

## Insight
Missing data impacts models more directly than noise.

While noise produced unpredictable behavior, missing data caused a stable and reliable drop in performance.

This suggests:
- models depend heavily on complete feature information  
- missing inputs disrupt predictions more strongly  
- simple imputation is not always sufficient  

---

## Why this matters
In real-world systems, missing data is unavoidable.

If not handled properly:
- predictions become unreliable  
- system performance degrades silently  

Understanding this behavior is essential for building robust systems.

---

## Connection to previous experiment

- Noise → unpredictable impact  
- Missing data → consistent degradation  

Not all imperfections affect systems in the same way.

---

## Next
- Compare different imputation strategies  
- Identify most sensitive features  
- Explore methods to detect unreliable predictions  