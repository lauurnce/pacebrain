# Study notes — interpretability

Knowing a model is accurate is not the same as knowing why. For anything giving
advice to a person, the why matters.

## Coefficients

For a linear model on standardised features, the coefficients *are* the feature
importances — magnitude is effect size, sign is direction. This is the strongest
argument for fitting a linear model first even when planning something more
complex: it is a free explanation of what drives the target.

Only valid on scaled features. Unscaled, coefficient size mostly reflects units.

## Tree importances

Trees report how much each feature reduced impurity across all splits. Cheap,
built in, and biased toward high-cardinality features that offer more possible
split points.

## Permutation importance

Shuffle one feature's values and measure how much performance drops. Big drop
means the model relied on it.

Model agnostic, works on a network, and measures what the model actually uses
rather than how it was built. The caveat is correlated features: if two carry
the same information, shuffling either alone changes little, and both look
unimportant when together they are essential.

## Partial dependence

Vary one feature across its range, hold others fixed, plot the prediction. Shows
the *shape* of a relationship, not just its strength — whether an effect
saturates, reverses, or is linear.

More informative than an importance ranking, and slower.

## SHAP

Attributes a prediction to each feature via a game-theoretic allocation, giving
per-prediction explanations that sum to the output. The most principled option
and the most expensive.

## The sanity check this really buys

The most valuable use is not explaining to users, it is catching nonsense. If a
finish-time model leans hardest on days-since-long-run and barely uses weekly
mileage, something is wrong — with the features, the data, or the training.

A model can be accurate on validation and still be right for the wrong reasons.
Importances catch that; a metric does not.
