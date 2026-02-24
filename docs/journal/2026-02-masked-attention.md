# Masked attention

Preventing positions from attending to later ones, so a model trained in parallel cannot see the future it is meant to predict. Without the mask training and inference disagree, and the model appears excellent then fails completely.
