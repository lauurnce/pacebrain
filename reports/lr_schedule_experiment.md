# Does a learning-rate schedule help? — measured, and mostly no

`ReduceLROnPlateau` is the obvious next thing to add to a training loop that
already early-stops on validation loss: both watch the same signal, so the
schedule can slow the optimiser down before the loop gives up. It is also the
kind of change that is easy to add, easy to believe in, and hard to notice
when it silently makes things worse.

So it was measured before being switched on. It is off by default.

## Setup

`FinishPredictorConfig` defaults — 1000 synthetic rows, 300 epoch budget,
early stopping at `patience=25`, Adam at `lr=1e-3`. Metric is best validation
MSE, the same number the training loop reports and checkpoints on. "OFF" is
`lr_schedule=False`.

## Grid, seed 42

| lr_patience | factor | best val MSE | stopped at | LR drops |
|---|---|---|---|---|
| **OFF** | — | **30.4842** | 189 | 0 |
| 8 | 0.5 | 41.7667 | 191 | 5 |
| 8 | 0.8 | 34.6076 | 189 | 4 |
| 12 | 0.5 | 40.7774 | 191 | 4 |
| 12 | 0.8 | **27.3439** | 298 | 7 |
| 15 | 0.5 | 30.4842 | 189 | 1 |
| 15 | 0.8 | 30.4842 | 189 | 1 |
| 20 | 0.5 | 30.4842 | 189 | 1 |
| 20 | 0.8 | 30.4842 | 189 | 1 |

Three distinct behaviours, and the boring one is the most instructive:

**Aggressive decay actively hurts.** `0.5 / 8` is 37% worse than no schedule
at all. With 200 validation rows the val loss is noisy epoch to epoch, so an
8-epoch patience fires on noise rather than on a real plateau. Five halvings
put the LR 32x below where it started, and the optimiser cannot move far
enough to recover before early stopping ends the run.

**Patience at 15 or 20 does nothing whatsoever.** Those four rows return the
baseline number to four decimal places. The single LR drop happens *after*
the best epoch has already been recorded, so it cannot affect the checkpoint.
The counters race: with early stopping at 25, an LR patience near it means
the schedule never gets to act.

**One setting looked good.** `0.8 / 12` reached 27.34, a 10.3% improvement,
and trained to epoch 298 instead of 189 — the gentler decay kept finding
small improvements rather than stalling.

## The 10.3% did not survive more seeds

| seed | OFF | 0.8 / 12 |
|---|---|---|
| 42 | 30.48 | **27.34** |
| 7 | 25.14 | 25.49 |
| 123 | 23.21 | **17.27** |
| 2024 | 24.15 | 25.12 |
| **mean** | **25.75** | **23.81** |

Better on **2 of 4 seeds**. The mean improves by 7.5%, but that average is
carried almost entirely by seed 123 (23.21 → 17.27) while two seeds got
slightly worse. Two out of four is a coin flip, and one large win among four
runs is the shape you expect from variance, not from an effect.

Reporting the seed-42 grid alone would have made this look like a clean 10%
win. It is the single most misleading number in this document, and it is the
one that would have been quoted.

## Decision

Off by default. The flag and the tuned values stay, because it is one setting
away and worth re-running on real data, where a noisier loss surface and a
less friendly optimisation landscape may genuinely reward it. On this
synthetic problem it does not.

```python
cfg = FinishPredictorConfig(lr_schedule=True)   # 0.8 / 12 by default
```

## What this is really about

The generator here is a smooth closed-form function of six features with mild
gaussian noise. Adam at a fixed 1e-3 has no trouble with a surface like that,
so there is no plateau for the schedule to rescue — only noise for it to
misread. LR scheduling earns its keep on hard optimisation landscapes, and
this is not one.

The transferable part is the method, not the result: four seeds cost a couple
of minutes and turned an apparent 10% improvement into a coin flip. A single
seed is not a measurement.
