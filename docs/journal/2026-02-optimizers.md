# Study notes — optimisers

How parameters actually get updated, and why nobody uses plain gradient descent.

## Batch, stochastic, mini-batch

- **Batch** — compute the gradient over the entire dataset, take one step.
  Accurate direction, very slow, and the whole dataset must fit in memory.
- **Stochastic (SGD)** — one sample at a time. Fast steps, very noisy
  direction.
- **Mini-batch** — a few dozen samples. The compromise everyone actually uses.

The noise in mini-batch gradients is not purely a cost. It helps escape poor
local minima and sharp valleys, which is part of why the compromise wins rather
than merely being a memory concession.

Batch size interacts with learning rate: bigger batches give less noisy
gradients, which tolerates a larger step.

## Momentum

Plain SGD oscillates across narrow ravines — steep in one direction, shallow in
another — making slow progress along the shallow axis.

Momentum accumulates a running average of past gradients and steps along that
instead. Consistent directions accumulate; oscillating ones cancel. The physical
analogy is a ball rolling downhill rather than a point teleporting.

## Adaptive rates

Different parameters may need different step sizes. A feature that appears
rarely should take larger steps when it does appear.

- **AdaGrad** — divides the step by the accumulated sum of squared gradients.
  The denominator only grows, so the learning rate decays monotonically toward
  zero and training eventually stalls.
- **RMSProp** — same idea with an exponentially decaying average instead of a
  sum, so it does not stall.
- **Adam** — RMSProp plus momentum, with bias correction for the initial steps
  when both running averages start at zero.

Adam at `lr=1e-3` is the standard default and a reasonable starting point.

## What Adam does not fix

It is not a substitute for reasonable input scaling. If one feature is measured
in the hundreds and another in single digits, the loss surface is badly
conditioned before the optimiser sees it. Standardise inputs regardless of the
optimiser.

It also will not rescue a wrong learning rate by much. Diverging loss usually
means the rate is too high; a loss that flatlines immediately usually means it
is too low or the gradients are not flowing at all.

## Zeroing gradients

Frameworks accumulate gradients across backward passes rather than replacing
them — deliberately, since it enables splitting a large batch across several
passes. The consequence is that forgetting to zero them does not error, it just
trains on the sum of everything so far. Worth remembering as a cause of
mysterious divergence.

## Next

Regularisation — what to do once the model fits the training data too well.
