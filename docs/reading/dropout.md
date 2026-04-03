# Dropout — Srivastava et al., 2014

Randomly zeroes units during training, sampling a different thinned network each
step.

Framed as an efficient approximation to averaging exponentially many networks,
which is why it behaves like an ensemble at the cost of one model.

The implementation detail that matters in practice: it must be disabled at
inference, and the activations rescaled, or train and test disagree. That
mismatch shows up as validation loss lower than training loss, which reads as a
bug elsewhere.
