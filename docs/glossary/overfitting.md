# Overfitting

When a model learns the training set's noise as though it were signal, so training loss keeps falling while validation loss flattens or rises. The gap between the two curves is the diagnostic, not the absolute value of either — a model can overfit badly while still scoring well.

In PaceBrain, Overfitting matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
