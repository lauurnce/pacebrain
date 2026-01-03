# Noise floor

The best score any predictor can achieve on data carrying irreducible noise, set by the noise distribution alone. Every result should be read against it rather than against zero: with N(0, 2) noise the best possible MAE is E|N(0, 2)| = 1.596 min, so a model scoring 4.12 is 2.6x off the achievable best, which "75% better than baseline" completely hides.

In PaceBrain, Noise floor matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
