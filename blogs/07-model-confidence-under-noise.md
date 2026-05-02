# When models become overconfident

## Problem

Accuracy alone does not tell the full story of a model’s behavior.

Even when performance remains stable, predictions may become less reliable.

I wanted to understand:
Do models become overconfident when data quality degrades?

---

## Approach

After introducing noise into the data, I observed model predictions and confidence levels.

Instead of only measuring accuracy, I focused on how certain the model was about its predictions.

---

## Observations

- In some cases, the model remained confident even when predictions were incorrect  
- Noise did not always reduce confidence  
- This creates a gap between accuracy and reliability  

---

## Insight

Models can appear stable in terms of accuracy, but internally become unreliable.

Confidence does not always reflect correctness.

---

## Why this matters

In real-world systems:
- confident but incorrect predictions are dangerous  
- systems may fail silently  
- errors may not be easily detectable  

---

## Conclusion

Understanding confidence is as important as understanding accuracy.

Reliable systems must know when they are uncertain.