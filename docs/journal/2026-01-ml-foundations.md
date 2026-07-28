# Study notes — supervised learning foundations

Working through the basics properly before touching a framework.

## The setup

Supervised learning: given pairs `(x, y)`, find a function `f` such that
`f(x) ≈ y` for pairs *not* in the training set. That last clause is the whole
problem. Fitting the training data is trivial — a lookup table does it
perfectly and generalises not at all.

## Regression vs classification

Regression predicts a continuous quantity, classification a discrete label.
The distinction is mostly in the output layer and the loss:

| | Output | Typical loss |
|---|---|---|
| Regression | one real number | MSE / MAE |
| Binary classification | one probability | binary cross-entropy |
| Multi-class | probability vector | cross-entropy |

A finish time is continuous, so this is regression. Worth noting because a lot
of tutorial material is classification-first and the habits do not transfer
cleanly — accuracy has no meaning here.

## Bias and variance

- **High bias** — the model is too simple to represent the pattern. Train and
  test error are both high and close together. Underfitting.
- **High variance** — the model has captured noise specific to the training
  set. Train error is low, test error much higher. Overfitting.

The diagnostic is the *gap*, not either number alone. Low training error on its
own says nothing.

More data reduces variance and does nothing for bias. More capacity reduces
bias and increases variance. Knowing which one you have determines which lever
to pull, and guessing wastes a lot of time.

## Why a held-out set is non-negotiable

Any metric computed on data the model trained on is meaningless as an estimate
of future performance. The split has to happen before anything is fitted —
including preprocessing statistics like a mean and standard deviation, which
are themselves learned from the data.

Getting this subtly wrong is the classic way to produce a result that looks
excellent and does not survive contact with anything real.

## Features

The model can only use what it is given. Feature choice usually matters more
than architecture, especially with small data — a well-chosen feature encodes
domain knowledge that the model would otherwise need many examples to discover,
and may never discover at all.

## Next

Neural networks specifically: what makes them different from linear models, and
when the extra capacity is worth its cost.
