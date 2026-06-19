# When Models Become Confidently Wrong

## The research question

When should a model refuse to make a prediction?

Accuracy alone does not answer this question. A model can still return a prediction even when the input is noisy, incomplete, or far from the kind of data it was trained on.

The important issue is not only whether the prediction is wrong.

It is whether the model knows that its prediction has become unreliable.

---

## Confidence under noise

Many classifiers produce probability scores using `predict_proba()`.

These scores are often interpreted as confidence.

In a reliable system, increasing noise should reduce confidence in the correct class. If the input becomes less trustworthy, the model should become less certain.

But failure becomes more dangerous when the model remains confident while making more mistakes.

---

## Confidence collapse

The confidence collapse experiment increases noise gradually and tracks three signals:

- noise level
- confidence from predicted probabilities
- number of wrong predictions

As noise increases, the model receives less useful information. Accuracy begins to fall, confidence in the correct class declines, and wrong predictions become more frequent.

This pattern suggests that confidence can act as an early warning signal, but only if it is measured carefully.

---

## Refusal as a safety behavior

A deployed model does not always need to answer.

Sometimes the safest behavior is to refuse, defer, or ask for better input.

A possible refusal rule could combine:

- low confidence in the predicted class
- large confidence drop compared with clean validation data
- high estimated failure risk
- detection of noisy or missing inputs

This turns robustness testing into a practical design question:

Should the system predict, or should it admit uncertainty?

---

## Key Insight

A model that is wrong and uncertain is easier to manage.

A model that is wrong and confident is more dangerous.

Studying confidence collapse helps identify the point where prediction should become refusal.
