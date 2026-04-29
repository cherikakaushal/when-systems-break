# What I learned about systems under imperfect conditions

## The Question
This project started with a simple idea:

What happens when systems face imperfect data?

Real-world data is rarely clean.
It is noisy, incomplete, and uneven.

I wanted to understand how models behave under these conditions.

---

## What I observed

Across the experiments, three different patterns emerged:

### 1. Noise → Unpredictable behavior
Adding noise did not always reduce performance.

Sometimes accuracy dropped.
Sometimes it improved slightly.

This showed that system behavior under noise is not always intuitive.

---

### 2. Missing data → Consistent degradation
When data was missing, performance dropped reliably.

Unlike noise, this effect was stable and predictable.

This indicates that models depend heavily on complete inputs.

---

### 3. Feature importance → Uneven dependency
The model relied heavily on a small subset of features.

Some inputs had strong influence, while others had minimal impact.

This means not all data contributes equally to decisions.

---

## Core Insight

Not all imperfections affect systems in the same way.

- Some introduce instability (noise)
- Some cause predictable failure (missing data)
- Some expose dependency (feature importance)

Understanding these differences is critical.

---

## What this means

Reliable systems are not just about accuracy on clean data.

They must be:
- robust to noise  
- resilient to missing inputs  
- aware of critical dependencies  

Without this, systems can fail in subtle and unexpected ways.

---

## Final Thought

The goal is not just to build models that perform well.

The goal is to understand how they behave when things go wrong.

Because in the real world,
things always do.