# Batch size

How many samples are processed before a parameter update. Larger batches give less noisy gradient estimates and tolerate a higher learning rate; smaller ones add noise that can help escape poor minima.

In PaceBrain, Batch size matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.

## See also

- [Momentum](momentum.md)
- [Fine tuning](fine-tuning.md)
