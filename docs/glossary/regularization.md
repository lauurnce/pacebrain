# Regularization

Any constraint that trades training fit for generalisation — weight decay, dropout, early stopping, or simply less capacity. All of them work by making the model harder to fit to noise, so the right amount depends on how much noise the data actually has.

## See also

- [Underfitting](underfitting.md)
- [Overfitting](overfitting.md)

In PaceBrain, Regularization matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
