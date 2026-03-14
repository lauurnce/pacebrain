# Study notes — trees and gradient boosting

The honest competitor for tabular problems, and the thing a neural network
actually has to beat.

## Decision trees

Recursively split on the feature and threshold that best separates the target.
Predictions are the average of the training rows in each leaf.

Naturally handles non-linear relationships and interactions, needs no scaling
(splits are order-based, so units are irrelevant), and is readable end to end.

Alone, a deep tree overfits badly — taken far enough it isolates every training
row in its own leaf.

## Random forests

Many trees, each on a bootstrap sample with a random subset of features
considered at each split, averaged. The randomisation decorrelates the trees so
their individual errors cancel. Hard to overfit, few knobs, a strong baseline
with almost no tuning.

## Gradient boosting

Trees in sequence, each fitted to the residuals of the ensemble so far. Every
tree corrects what the previous ones got wrong.

More accurate than a forest and more sensitive to hyperparameters, since
mistakes compound rather than average out. Needs a learning rate and enough
regularisation to stop it fitting noise. XGBoost, LightGBM and CatBoost are the
standard implementations.

## Why this matters here

**On small-to-medium tabular data, gradient boosted trees usually beat neural
networks.** This is a consistent, well-documented result, not a niche opinion.

Which makes it the uncomfortable question worth asking early: if the goal is
the best predictions on a table of six features, a neural network may simply be
the wrong tool.

For a learning project that is fine — the point is understanding networks. But
the comparison should be run and reported honestly rather than quietly skipped,
because "the MLP beat a handicapped formula" is a much weaker claim than "the
MLP beat gradient boosting."

## Where networks do win

Very high dimensional input, unstructured data, and anything sequential where
the architecture encodes a real assumption about the structure. A tree cannot
easily represent "segment 7 depends on how segments 1 through 6 were run."
