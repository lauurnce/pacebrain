# Study notes — SGD, momentum, Adam

All of these answer the same question — given a gradient, how far and in what
direction do I step — and differ only in how much history they keep.

**Plain SGD.** `w -= lr * g`. No memory. Sensitive to the learning rate and to
badly scaled features, because one step size has to suit every parameter.

**Momentum.** Keep a running average of past gradients and step along that.
Damps oscillation across a narrow valley and accelerates along its floor. One
extra buffer per parameter.

**RMSprop.** Keep a running average of squared gradients and divide by its
root. Gives each parameter its own effective step size, so parameters with
consistently small gradients still move.

**Adam.** Both at once — momentum in the numerator, RMSprop in the denominator,
plus a bias correction for the first few steps when the averages start at zero.

## Why Adam is the default

It works acceptably without tuning, which matters more than being optimal.
The cost is two extra buffers per parameter and a reputation for generalising
slightly worse than well-tuned SGD on some vision problems.

## The thing worth remembering

None of these fix a bad learning rate; they only make the range of tolerable
learning rates wider. The step size is still the hyperparameter that matters
most.
