# Outlier

An observation far from the bulk of the data. The decision that matters is not detecting it but classifying it: a measurement error should be removed, while a genuine extreme carries real information and removing it is quietly falsifying the dataset.

## See also

- [Feature engineering](feature-engineering.md)
- [Grouped split](grouped-split.md)

In PaceBrain, Outlier matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
