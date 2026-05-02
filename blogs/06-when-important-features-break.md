# What happens when important features disappear?

## Problem

Earlier, I observed that not all features contribute equally to a model’s decisions.

Some features dominate predictions.

This raises a question:
What happens if the most important features are removed?

---

## Approach

Using feature importance rankings, I removed the top contributing features from the dataset.

The model was then evaluated again on this modified data.

---

## Observations

- Performance dropped significantly after removing key features  
- The model struggled to maintain accuracy  
- Less important features were not sufficient to compensate  

---

## Insight

Models are not just sensitive to data quality, but also to *which data is affected*.

Losing critical features impacts performance far more than random noise.

---

## Why this matters

In real-world systems:
- key inputs may fail  
- sensors may stop working  
- critical data may be missing  

Understanding feature dependency helps identify system vulnerabilities.

---

## Conclusion

Not all failures are equal.

Removing the wrong piece of information can break the system faster than adding noise.