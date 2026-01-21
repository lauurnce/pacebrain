# Data augmentation

Expanding a training set with label-preserving transformations of existing examples. The constraint doing the work is *label-preserving*: an augmentation that changes what the correct answer should be is not adding data, it is adding wrong data.

## See also

- [Bias variance tradeoff](bias-variance-tradeoff.md)
- [Overfitting](overfitting.md)

## See also

- [Weight decay](weight-decay.md)
- [Overfitting](overfitting.md)

In PaceBrain, Data augmentation matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
