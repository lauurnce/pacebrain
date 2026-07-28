# Evaluation plan

Decided up front, so the metric is not chosen after seeing which one flatters
the result.

## Headline metric: MAE in minutes

Mean absolute error, reported in minutes, because "wrong by 4 minutes on
average" is a sentence a runner understands. MSE of 30 is not interpretable by
anyone.

Train on MSE (smoother gradients), report MAE. Different jobs.

## Also worth reporting

- **Per-distance breakdown.** A single averaged number hides everything. A
  model can look fine overall while being useless at the marathon, which is the
  distance people care most about. Any error that scales with race duration
  will be invisible in the average and obvious in the breakdown.
- **Signed mean error**, not just absolute. It separates bias from spread. A
  model consistently 10 minutes optimistic is a different problem from one that
  is randomly wrong by 10 minutes either way — and given the asymmetric cost of
  optimistic predictions, the sign matters.
- **Worst-case error.** The tail is what damages trust.

## Baselines are mandatory

A number with nothing to compare it to means nothing. Minimum set:

1. **Predict the training mean.** The floor. Failing to beat this means the
   model learned nothing at all.
2. **Linear regression on the same features.** The honest test of whether
   nonlinearity is earning its place. If the MLP does not clearly beat this,
   the extra capacity is decoration.
3. **A domain formula.** Riegel, *given a fair input*. A baseline handed an
   input it was not designed for produces a flattering comparison that says
   nothing about the model.

## The noise floor

Synthetic targets will have noise added deliberately. That sets a hard lower
bound on achievable error — for Gaussian noise of standard deviation `s`, the
best possible MAE is `s * sqrt(2/pi)`, about `0.8 * s`.

Compute it and state it alongside results. Without it there is no way to tell
"the model has converged" from "the model has plenty left to learn", and a
result that looks impressive against a baseline may still be far from the floor.

## Validation discipline

- Split before fitting anything, including the scaler.
- Same seed, same split, every run — otherwise comparisons across days are
  measuring the split rather than the change.
- Never tune against the test set. With a project this small, one validation
  split is enough as long as it is not repeatedly optimised against.

## What would falsify this project

If a linear regression matches the MLP on real data, or if the error band is
too wide to change anyone's pacing decision, then the honest conclusion is that
the extra machinery is not earning its place. That has to be a reportable
outcome, not a thing to avoid measuring.
