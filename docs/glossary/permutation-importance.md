# Permutation importance

Measuring a feature's contribution by shuffling it and observing how much performance drops. Model-agnostic and intuitive, but correlated features let the model recover the lost signal from a partner, so both appear unimportant.

In PaceBrain, Permutation importance matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.

## See also

- [Feature engineering](feature-engineering.md)
- [Grouped split](grouped-split.md)
