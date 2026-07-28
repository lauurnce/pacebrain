# Study notes — feature scaling

Why it matters and which method to use.

## The problem

Features measured on different scales make the loss surface badly conditioned.
A feature ranging 20–120 and another ranging 4–8 produce gradients differing by
an order of magnitude, so a single learning rate cannot suit both. The larger
feature dominates purely because of its units.

Distance-based methods and regularisation are affected even more directly —
both treat a unit of one feature as comparable to a unit of another, which is
only meaningful after scaling.

## Methods

**Standardisation** — `(x - mean) / sd`. Centres at zero, unit variance.
Preserves outliers rather than compressing them. The default.

**Min-max** — `(x - min) / (max - min)`. Maps to [0, 1]. Bounded, which some
architectures like, but a single extreme value squashes everything else into a
narrow band. Fragile against outliers.

**Robust scaling** — subtract the median, divide by the interquartile range.
Standardisation's shape without outlier sensitivity. Worth it when the data is
known to be dirty.

**Log transform** — for right-skewed positive quantities. Not scaling exactly;
it changes the distribution's shape, often turning a multiplicative
relationship into an additive one the model can represent more easily.

## The rule that actually matters

**Fit on training data only, then apply those statistics everywhere else.**

Fitting on the full dataset lets validation statistics influence the
transformation, which is leakage. It usually produces a small optimistic bias
rather than an obvious error, which is what makes it dangerous — nothing fails,
the number is just quietly wrong.

The same fitted statistics have to be reused at inference. Refitting on a
single new row is meaningless — one sample has zero variance.

## Persisting it

Scaling parameters are learned from data, so they are part of the model. A
saved model without its scaler is incomplete, and reconstructing the statistics
later by guessing at how training was configured is how silent errors happen.
Store them together.
