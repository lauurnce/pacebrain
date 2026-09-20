# Sequence masking

Hiding padded positions from a model or a loss. When sequences of different lengths are batched they are padded to the longest one, and the padding carries no information. A mask is a boolean tensor, true at real timesteps and false at padding, used to zero the padded contributions and to divide by the count of real ones. Without it the model is trained to predict the padding value, and short sequences, which are padded most, are distorted most.

## See also

- [Batch size](batch-size.md)
- [Loss function](loss-function.md)

In PaceBrain, per-kilometre races run from 5 to 42 timesteps, so `pad_collate` in `src/pacebrain/seq_data.py` pads a batch and returns the true lengths, `length_mask` in `src/pacebrain/seq_models.py` turns those lengths into a `(batch, max_len)` boolean mask, and `masked_mse_loss` uses it so only real timesteps count. The recurrent models take the lengths too and pack the batch, so the final hidden state is not computed over padding.
