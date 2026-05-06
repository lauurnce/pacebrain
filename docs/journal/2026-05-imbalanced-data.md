# Study notes — when one class dominates

Start by noticing it is mostly a metric problem, not a data problem.

## Metrics first

Accuracy is useless here. Precision, recall, F1 and a confusion matrix say what
is actually happening, and which of those to optimise depends entirely on which
error costs more.

## Then, if needed

**Class weights.** Weight the loss so minority errors count more. Cheap, no
data manipulation, usually the first thing to try.

**Undersampling** the majority discards real data. **Oversampling** the
minority duplicates it and invites memorisation. **Synthetic sampling** creates
interpolated examples, which can invent points in regions where the class does
not actually live.

## The rule that keeps this honest

Resample the training split only. Resampling before splitting puts copies of
the same example on both sides, and the resulting score is measuring
memorisation with a straight face.
