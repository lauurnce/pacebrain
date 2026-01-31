# Exploding gradient

The opposite failure — repeated multiplication by factors above one drives gradients to enormous values, and a single step destroys the weights. It announces itself as a loss that suddenly becomes NaN, and [[gradient-clipping]] is the standard blunt fix.

## See also

- [Gradient clipping](gradient-clipping.md)
- [Dropout](dropout.md)

## See also

- [Transformer](transformer.md)
- [Dropout](dropout.md)

In PaceBrain, Exploding gradient matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.
