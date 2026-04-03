# Random Forests — Breiman, 2001

Combines bagged decision trees with random feature subsetting at each split.

The feature subsetting is what decorrelates the trees, and decorrelation is what
makes averaging them worth more than any single one. Bagging alone leaves the
trees too similar.

Still a reasonable default on tabular data two decades later, which is worth
noting given how much has changed elsewhere.
