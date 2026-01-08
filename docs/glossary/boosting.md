# Boosting

Fitting models sequentially, each one trained on the errors the previous ones left behind. Unlike [[bagging]] it reduces bias rather than variance, which is why it can overfit with too many rounds and needs the number of rounds treated as a real hyperparameter.

Boosting interacts strongly with dataset size here; behaviours that are footnote-level on benchmarks are headline-level on a few hundred athlete-weeks.

## See also

- [Dimensionality reduction](dimensionality-reduction.md)
- [K nearest neighbours](k-nearest-neighbours.md)
