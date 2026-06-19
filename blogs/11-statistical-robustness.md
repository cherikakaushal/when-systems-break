# Why One Accuracy Score Is Not Enough

## Why repeat experiments?

A single accuracy score can make a machine learning model look more stable than it really is.

Model performance depends on randomness:

- how the data is split
- how noisy inputs are generated
- how model training is initialized
- which examples appear in the test set

Because of this, one run is only one snapshot of behavior.

---

## Variance matters

Two models can have similar average accuracy but very different stability.

A model with low variance behaves consistently across repeated runs.

A model with high variance may perform well once, then drop noticeably when the data split or noise pattern changes.

This is important because real-world systems do not receive the same perfect input every time.

---

## Confidence and reliability

Accuracy tells us whether predictions were correct.

It does not fully explain how reliable the model is under changing conditions.

When accuracy under noise varies strongly across runs, it suggests that the model may be sensitive to small changes in the data. This makes confidence harder to interpret, because a confident prediction may still depend on unstable input conditions.

---

## Repeatability

A robust model should not only perform well once.

It should perform consistently across repeated experiments.

Running each model 30 times gives a clearer picture of:

- average performance
- performance spread
- sensitivity to noisy data
- repeatability across random seeds

---

## Key Insight

One accuracy score answers:

How did the model perform this time?

Statistical robustness asks:

How reliably does the model perform across many possible versions of the same experiment?

That difference is what makes robustness testing more research-oriented than simple model evaluation.
