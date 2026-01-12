# Dropout

Randomly zeroing a fraction of activations during training, which stops the network leaning on any single unit and acts as a cheap ensemble. It must be switched off at inference via `model.eval()`, or predictions become non-deterministic — a bug that looks like model instability rather than a missing line.

In PaceBrain, Dropout matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
