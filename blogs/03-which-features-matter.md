# Which features actually matter?

## Problem
Machine learning models take in many features, but not all of them contribute equally.

I wanted to understand:
Which inputs actually drive the model’s decisions?

---

## Approach
Using the same model, I analyzed feature importance through model coefficients.

- Trained a logistic regression model  
- Extracted feature weights  
- Ranked features by their absolute impact  

---

## Observations (experiment)

Top features included:

- worst concavity  
- texture error  
- mean radius  
- worst compactness  

These features had significantly higher influence compared to others.

---

## Insight
The model does not treat all inputs equally.

A small number of features dominate the decision-making process.

This means:
- the model is highly dependent on specific inputs  
- removing or corrupting these features could significantly affect predictions  
- other features contribute very little  

---

## Why this matters
Understanding feature importance helps in:

- identifying critical inputs  
- detecting potential failure points  
- improving model robustness  

If key features become noisy or missing, the system can degrade rapidly.

---

## Connection to previous experiments

- Noise → unpredictable impact  
- Missing data → consistent degradation  
- Feature importance → reveals which inputs control behavior  

Together, these show:

System reliability depends not just on data quality, but also on which data is affected.

---

## Next
- Test model behavior by removing top features  
- Compare robustness across different models  
- Explore feature sensitivity under noise  