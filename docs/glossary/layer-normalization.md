# Layer normalization

Normalising each sample across its own features rather than across the batch. Because no cross-sample statistics are involved, batch size is irrelevant and training and inference behave identically — which is why sequence models and transformers prefer it to [[batch-normalization]].

## See also

- [Attention](attention.md)
- [Batch normalization](batch-normalization.md)

Layer normalization interacts strongly with dataset size here; behaviours that are footnote-level on benchmarks are headline-level on a few hundred athlete-weeks.

## See also

- [Vanishing gradient](vanishing-gradient.md)
- [Batch normalization](batch-normalization.md)
