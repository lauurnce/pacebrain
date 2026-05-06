# Study notes — changing the step size over time

A single learning rate has to be small enough not to diverge early and large
enough to make progress late. Those conflict, which is what schedules resolve.

**Step decay.** Multiply by a factor at fixed epochs. Simple, and the drop
points are arbitrary unless tuned.

**Cosine.** Smooth decay to near zero over the run. Widely used, one
hyperparameter, no cliffs.

**Reduce on plateau.** Drop when a monitored metric stops improving. Reactive,
but it fires on noise in the monitored metric, so its patience has to sit well
below any early-stopping patience or the two counters fight.

**Warmup.** Start small and rise for the first few hundred steps. Matters most
with large batches and adaptive optimisers, whose variance estimates are
unreliable before they have seen enough gradients.

## The honest caveat

A schedule is easy to add and hard to attribute. Any gain has to be measured
against several seeds without it, or the schedule gets credit for variance.
