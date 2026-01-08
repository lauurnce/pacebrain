# Embedding

A learned dense vector representing a discrete item, trained jointly with the rest of the model. It replaces the one-vector-per-category cost of one-hot encoding with a fixed small dimension, and because similar items end up nearby, it generalises across categories rather than treating each as unrelated.

In PaceBrain, Embedding matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.

## See also

- [Attention](attention.md)
- [Sigmoid](sigmoid.md)
