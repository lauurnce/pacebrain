# Optimizer

The rule that turns gradients into parameter updates — SGD, Adam, RMSprop and the rest. Adam adapts a per-parameter step size from running estimates of the gradient's mean and variance, which is why it usually converges faster than plain SGD without much tuning.

The failure mode to watch with Optimizer is silent optimism: results improve on the validation set for reasons that will not survive contact with next month's data.
