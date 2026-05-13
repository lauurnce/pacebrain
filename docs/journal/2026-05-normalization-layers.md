# Study notes — batch norm and layer norm

Both rescale intermediate activations to keep their distribution stable during
training. They differ in what they average over, and that difference decides
where each one is usable.

**Batch norm** normalises each feature across the batch. Effective, but it ties
every sample's output to the other samples in its batch, which causes three
practical problems: small batches give noisy statistics, training and inference
behave differently (inference uses running averages), and it does not fit
naturally with variable-length sequences.

**Layer norm** normalises each sample across its own features. No cross-sample
dependence, so batch size is irrelevant and train and inference are identical.
This is why transformers use it and why it is the safer default for sequence
work.

## The explanation that turned out to be wrong

Batch norm was originally justified as reducing "internal covariate shift".
Later work found that explanation does not hold up, and the benefit comes more
from smoothing the optimisation landscape. Worth keeping as a reminder that a
technique can work reliably while the story attached to it is wrong — the
empirical result and its explanation are separate claims.
