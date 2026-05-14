# Study notes — leakage

Leakage is when information reaches the model that will not exist at prediction
time. It does not produce an error. It produces a validation score that is too
good, which is far worse, because the failure is invisible until deployment.

## The forms I keep finding

**Preprocessing before splitting.** Fitting a scaler or imputer on the full
dataset means validation statistics influence the transform. Split first, fit
on train, transform both.

**Target leakage.** A feature that is a consequence of the target rather than a
cause of it. Available in the training table, absent in reality.

**Temporal leakage.** Training on data from after the prediction point. A
random split of time-ordered data does this automatically.

**Group leakage.** The same entity appearing in both splits — several races by
one runner, say. The model recognises the entity rather than learning the
relationship, and a grouped split is the fix.

## The tell

A validation score that seems too good usually is. The instinct to celebrate it
is exactly backwards; the correct response is to go looking for the leak,
because finding it later costs much more.
