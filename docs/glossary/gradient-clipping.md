# Gradient clipping

Rescaling gradients whose norm exceeds a threshold before the optimiser step, bounding how far any single update can move the weights. It treats the symptom rather than the cause, but for recurrent models it is usually the difference between training and diverging.

## See also

- [Layer normalization](layer-normalization.md)
- [Transformer](transformer.md)

## See also

- [Batch normalization](batch-normalization.md)
- [Transformer](transformer.md)

Gradient clipping interacts strongly with dataset size here; behaviours that are footnote-level on benchmarks are headline-level on a few hundred athlete-weeks.
