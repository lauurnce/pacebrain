# Softmax

A function turning a vector of scores into a probability distribution by exponentiating and normalising. Implementations subtract the maximum score first — mathematically a no-op, numerically the difference between a working function and an overflow.

## See also

- [Layer normalization](layer-normalization.md)
- [Attention](attention.md)

Softmax interacts strongly with dataset size here; behaviours that are footnote-level on benchmarks are headline-level on a few hundred athlete-weeks.
