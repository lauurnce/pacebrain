# Study notes — what batch size actually changes

Three separate effects, usually conflated.

**Gradient noise.** Small batches give noisy gradient estimates. That noise is
not purely harmful — it helps escape sharp regions and is part of why small
batches often generalise better.

**Hardware efficiency.** Larger batches use the accelerator better up to a
point, after which memory bandwidth dominates and throughput flattens.

**Effective learning rate.** Larger batches mean less noise per step, so the
step size can rise. The linear scaling rule — double the batch, double the rate
— holds approximately and breaks at large scale without warmup.

## The interaction that surprises

Changing batch size while holding the learning rate fixed changes the outcome,
so a batch size comparison that does not retune the rate is comparing two
things at once and attributing the result to one of them.

## Gradient accumulation

Summing gradients over several small batches before stepping simulates a large
batch within a small memory budget. It buys the statistics of a large batch but
none of the speed.
