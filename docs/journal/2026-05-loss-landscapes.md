# Study notes — what the optimiser is walking on

The loss as a function of millions of parameters cannot be pictured, but a few
facts about its shape explain otherwise mysterious behaviour.

**Local minima are mostly not the problem.** In very high dimensions, a point
where every direction curves upward is vanishingly unlikely. Saddle points —
up in some directions, down in others — are far more common, and momentum is
largely what gets past them.

**Flat minima generalise better than sharp ones.** A wide basin means small
parameter perturbations barely change the loss, which is roughly what
robustness to a slightly different data sample looks like. This is part of why
small batches, which add gradient noise, can generalise better.

**Conditioning matters more than curvature depth.** A long narrow valley makes
plain gradient descent oscillate across the walls while creeping along the
floor. That is the geometry momentum and per-parameter step sizes exist to
handle.

## Why this is worth knowing

It reframes tuning. A learning rate that diverges is not "too big" in the
abstract, it is too big for the sharpest direction in the local geometry, which
is why warmup and schedules help rather than just smaller constants.
