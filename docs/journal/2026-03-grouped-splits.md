# Study notes — grouped splits and subtle leakage

Leakage is the fastest way to a great score that means nothing. Most of it is
subtler than "the target is in the features".

## Group leakage

Rows sharing a source are not independent. Several races by one runner share
that runner's physiology, so a random split puts some of their races in train
and some in validation.

The model partly memorises the individual rather than learning the pattern, and
the validation score reflects that memorisation. It will not survive a new
user, which is exactly the case the product needs.

Fix: split by group. All of a runner's rows go entirely to one side.

## Sequence leakage

The same problem one level down. If a race is split into segments and segments
are assigned randomly, segments 1–5 land in train and 6–10 in validation. They
are near-duplicates from the same event.

Split by race, never by segment.

## Preprocessing leakage

Any statistic learned from data belongs to the training split: scaler mean and
standard deviation, imputation medians, category encodings, feature selection.

Computing them on everything and then splitting is the classic error. It rarely
produces an obvious symptom — just a small optimistic bias, which is the worst
kind of bug because nothing signals it.

## Temporal leakage

Training on data recorded after the validation period means using the future to
predict the past. Whether it applies depends on whether the framing is a
forecast, but the moment a feature depends on a prior outcome, ordering starts
to matter.

## Target leakage

A feature that would not exist at prediction time. Anything recorded during or
after the race — average race heart rate, splits, finishing position — cannot
be an input to predicting the finish time. Obvious stated plainly, easy to
include by accident when columns are taken wholesale from an export.

## The tell

Results that look too good. A validation score far better than the problem
plausibly allows is more likely leakage than success. It is worth treating a
suspiciously good number as a bug report rather than an achievement.
