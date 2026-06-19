# When Should a Model Say "I Don't Know"?

## The research question

When should a machine learning system stop trusting itself?

Most classifiers are designed to always return a prediction. Even when the input is noisy, incomplete, or outside the training distribution, the system still chooses a class.

That behavior can be risky.

A more reliable system should sometimes refuse to answer.

---

## From prediction to refusal

The standard prediction flow is simple:

```python
prediction = model.predict(X)
confidence = max(model.predict_proba(X))
```

This means the model predicts every time.

A refusal-aware system adds one more rule:

```python
if confidence < threshold:
    prediction = "REFUSE"
```

The model still predicts when confidence is high, but it can defer when confidence falls below a safety threshold.

---

## Accuracy and coverage

Refusal creates a tradeoff between accuracy and coverage.

Coverage measures how often the model still makes a prediction.

Accuracy measures how often those accepted predictions are correct.

As the threshold increases:

- more low-confidence predictions are refused
- coverage decreases
- accepted predictions tend to become more accurate

This is useful because a deployed model does not need to answer every input. In high-risk settings, refusing uncertain predictions may be safer than making unreliable ones.

---

## Why this matters

Earlier experiments showed that models fail under noise, missing data, and feature degradation.

The refusal experiment adds the next layer:

Models fail.

We can detect some of that risk.

We can respond by refusing predictions when confidence becomes too low.

---

## Key Insight

A robust machine learning system is not only one that predicts accurately.

It is one that knows when not to predict.
