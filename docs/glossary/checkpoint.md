# Checkpoint

A saved snapshot of a model's learned parameters. Holds weights only, not architecture and not preprocessing, so loading one requires rebuilding the model with matching hyperparameters.

## See also

- [Gradient](gradient.md)
- [Learning rate](learning-rate.md)

The failure mode to watch with Checkpoint is silent optimism: results improve on the validation set for reasons that will not survive contact with next month's data.
