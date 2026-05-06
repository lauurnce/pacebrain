# Study notes — splitting data that has an order

A random split of time-ordered data trains on the future to predict the past.
The score that comes out is excellent and meaningless.

## What to do instead

**Forward chaining.** Train on everything up to time t, validate on the window
after it, then roll forward. Each fold respects the arrow of time, and the
spread across folds says how stable the relationship is.

**Gap.** Leave a hole between train and validation when features use rolling
windows, or the last training rows and the first validation rows share inputs.

## The other trap

Features computed with future information. A rolling mean that is centred
rather than trailing peeks ahead. So does any normalisation fitted on the whole
series. These are invisible in the code and obvious in the score.

## The reality check

If a time series model looks far better than the domain suggests it should,
assume leakage before assuming skill. That prior is right most of the time.
