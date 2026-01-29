# K-nearest neighbours

Predicting from the labels of the closest training examples, with no training step at all. It shifts the entire cost to inference and depends completely on the distance metric, so unscaled features silently make the largest-range column dominate.

In PaceBrain, K nearest neighbours matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.

## See also

- [Pca](pca.md)
- [Decision tree](decision-tree.md)
