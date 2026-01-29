# Min-max scaling

Rescaling a feature to a fixed range, usually 0 to 1. More sensitive to outliers than [[standardization]], since a single extreme value compresses everything else into a narrow band.

In PaceBrain, Min max scaling matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.

## See also

- [Permutation importance](permutation-importance.md)
- [Test set](test-set.md)
