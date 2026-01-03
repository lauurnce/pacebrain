# Inference

Running a trained model to produce predictions. Requires evaluation mode so dropout is disabled, gradients switched off for speed, and exactly the same input scaling used during training.

In PaceBrain, Inference matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.

## See also

- [Fine tuning](fine-tuning.md)
- [Epoch](epoch.md)
