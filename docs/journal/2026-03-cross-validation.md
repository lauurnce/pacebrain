# Study notes — cross-validation

A single train/validation split gives one estimate from one arbitrary partition.
With little data, that estimate is noisy enough to mislead.

## k-fold

Split into `k` parts, train `k` times, each time holding out a different part.
Average the scores.

Every row is used for validation exactly once, so the estimate is far more
stable. Cost is `k` times the training, which is irrelevant when a run takes
seconds and prohibitive when it takes hours.

`k = 5` or `10` is conventional. Larger `k` means more training data per fold
and higher variance between folds.

## The variance is information

Report the spread across folds, not just the mean. Scores of 4.1 +/- 0.2 and
4.1 +/- 2.0 are very different situations — the second says the result depends
heavily on which rows landed where, and any comparison at that scale is noise.

## Grouped data

If rows are not independent, random k-fold leaks.

Several races from the same runner are correlated. Split randomly and the same
runner appears in both train and validation, so the model can partly memorise
that individual and the score is optimistic in a way that will not survive a
new user.

**Group k-fold** keeps all rows for a group together. For sequence data the
same logic applies at the sequence level — segments from one race must not be
split across folds, or near-duplicates end up on both sides.

## Time ordering

For genuinely temporal problems, random splitting lets the model train on the
future and predict the past. Use forward-chaining splits instead: train on
everything before a cutoff, validate after it.

Whether this applies depends on the framing. Predicting *this* race from *this*
training block is cross-sectional, not a forecast — but the moment a feature
depends on a prior race outcome, ordering matters.

## When one split is enough

A held-out set is simpler and sufficient when data is plentiful and runs are
expensive. The important thing is not which scheme you pick but that the split
respects the structure of the data, which is where the real errors live.
