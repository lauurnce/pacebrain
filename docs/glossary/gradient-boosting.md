# Gradient boosting

Fitting each new tree to the gradient of the loss left by the ensemble so far. It is the strongest general method on tabular data, and its main risk is that more rounds always fit the training set better whether or not that helps.

## See also

- [Decision tree](decision-tree.md)
- [Boosting](boosting.md)

In PaceBrain, Gradient boosting matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
