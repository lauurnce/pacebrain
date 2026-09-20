# Masked mse normalisation

`seq_models.masked_mse_loss` averages the squared error over real timesteps only:

```
L = sum_{b,t} m[b,t] * (yhat[b,t] - y[b,t])^2  /  max(1, sum_{b,t} m[b,t])
```

where `m[b,t]` is 1 while `t < lengths[b]` and 0 in the padding (`length_mask`).

The divisor is the count of real timesteps, not `batch * max_len`. For a 5 km race and a marathon padded to 42 steps that is 5 + 42 = 47 against a 2 x 42 = 84 grid. Dividing by 84 would shrink the loss by 47/84, about 0.56, and would also count the 37 padded positions of the short race as error.

Those padded positions are not harmless. `pad_collate` pads targets with 0, and at a padded step the model's output is just the head's bias (see `_forward_recurrent`), so an unmasked loss contains `(b - 0)^2` at every padded step and can be lowered by pushing the bias towards zero: the model is being trained to predict the padding. Short races are padded most, so the pull is strongest on the races with the least real data. The mask multiplies those terms by 0, which also gives the padded outputs a zero gradient.

Two details worth keeping in mind:

- `clamp(min=1)` only matters for a batch with no real timesteps at all; there the loss is 0 rather than 0/0.
- Every real timestep carries the same weight, `1 / sum(m)`. A marathon contributes 42 terms and a 5 km race 5, so within a batch long races count for proportionally more than they would under a per-race average.
