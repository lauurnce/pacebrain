# Prior art — race time prediction

Survey done before building anything, mostly to work out what the baselines
should be.

## Power-law formulas

**Riegel (1977)** — `T2 = T1 * (D2/D1)^1.06`. One free parameter, fitted across
a large set of race results. The exponent above 1.0 encodes that longer races
are harder per kilometre.

**Cameron** — a refinement with a distance-dependent correction. More accurate
toward the marathon end, more parameters, same input requirement.

Shared failure mode: they need a recent race, and they assume the runner's
endurance profile matches the population the exponent was fitted on. A runner
with a large aerobic base and no speed is mispredicted in one direction; a fast
5k runner with no long runs in the other.

## VDOT / Daniels

Maps a race performance to an estimated VO2max equivalent, then reads predicted
times for other distances off the same table. Mathematically close to a power
law, with a physiological interpretation attached and calibrated training paces
as the real output.

Same input problem: it needs a performance to start from.

## Critical speed / two-parameter models

Model performance as a hyperbolic relationship between speed and time to
exhaustion, yielding a critical speed plus a finite work capacity above it.
Two parameters means fitting needs at least two performances, which makes the
input problem worse rather than better.

Still worth knowing, because the two-parameter shape describes reality better
than a single exponent. The gap between them is roughly the gap between
"aerobic base" and "speed", which are separately trainable — and a single
exponent cannot represent a runner who has one and not the other.

## Published ML attempts

Most of the literature predicts from prior race results, which means competing
with Riegel on Riegel's own input space and winning by small margins. The work
that does use training data tends to have small cohorts and no public dataset.

The gap: training-history-to-race-outcome at consumer scale. Probably
underexplored because the labels are expensive — you only learn the answer when
someone actually races.

## What this means for baselines

Riegel is the obvious baseline. **But it has to be given a fair input.**
Feeding it something that is not a race time and then reporting a large
improvement proves nothing about the model; it proves the baseline was
handicapped.

Where no real reference time exists in the data, the defensible baselines are
predicting the training-set mean and a linear regression on the same features.
Neither can be accused of being set up to lose.
