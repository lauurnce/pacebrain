# Sequence masking

Hiding padded positions from a model or a loss. When sequences of different lengths are batched they are padded to the longest one, and the padding carries no information. A mask is a boolean tensor, true at real timesteps and false at padding, used to zero the padded contributions and to divide by the count of real ones. Without it the model is trained to predict the padding value, and short sequences, which are padded most, are distorted most.

## See also

- [Batch size](batch-size.md)
- [Loss function](loss-function.md)
