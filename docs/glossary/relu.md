# ReLU

`max(0, x)` — the default hidden-layer nonlinearity, cheap to compute and with a derivative that is exactly 1 wherever the unit is active, so it does not attenuate gradients. Its failure mode is the mirror image: a unit pushed permanently negative receives zero gradient forever and cannot recover.

## See also

- [Softmax](softmax.md)
- [Exploding gradient](exploding-gradient.md)

In PaceBrain, Relu matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
