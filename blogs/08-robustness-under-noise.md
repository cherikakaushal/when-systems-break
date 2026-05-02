# How robust is a model to increasing noise?

## Problem

Earlier experiments showed that noise affects model accuracy.

But instead of testing noise once, I wanted to understand:

How does performance change as noise gradually increases?

---

## Approach

Noise was introduced at different levels (0% → 50%) and model accuracy was measured at each step.

---

## Observations

- Accuracy gradually decreased as noise increased  
- The decline was not always linear  
- Small noise levels had minimal impact  

---

## Insight

Models can tolerate small imperfections, but performance degrades beyond a threshold.

---

## Visualization

![Noise Curve](../experiment/noise_curve.png)

---

## Conclusion

System robustness is not binary.

Understanding how performance degrades over time is more useful than testing single failures.