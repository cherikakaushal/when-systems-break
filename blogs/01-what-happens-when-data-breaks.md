# What happens when data breaks?

## Problem
In real-world systems, data is rarely perfect.
There can be noise, missing values, or inconsistencies.

I wanted to understand:
What actually happens to a machine learning model when the data it receives is slightly degraded?

---

## Approach
I started with a clean dataset and trained a simple model to establish a baseline.

Then, I introduced controlled changes:
- added random noise to input features
- removed a portion of the data (missing values)

The idea was not to improve performance, but to observe how it fails.

---

## Observations
- Even small amounts of noise caused a noticeable drop in accuracy  
- Missing values had a more uneven impact — some features mattered much more than others  
- The model didn’t “fail loudly”; performance degraded gradually  

---

## Insight
Models are more sensitive to data quality than they appear.

What stood out was:
The model’s behavior didn’t collapse instantly—it degraded quietly.

This makes failure harder to detect in real systems.

---

## Why this matters
In production systems, we often assume models are working fine if they don’t crash.

But degraded input can lead to unreliable outputs without obvious warning.

Understanding this behavior is important for building systems we can trust.

---

## Next
- test different levels of noise  
- identify which features are most sensitive  
- explore ways to detect silent failure
