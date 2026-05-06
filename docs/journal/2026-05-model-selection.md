# Study notes — choosing between models

The comparison is easy to get wrong in a way that always flatters the model you
chose last.

## The rule

Every decision made by looking at a split contaminates it. Selecting a model on
the validation set means the validation score is now optimistic for that model,
because it was chosen partly for fitting that particular sample.

## The structure that avoids it

Three splits: train to fit, validation to choose, test to report once. Or nested
cross-validation, where an inner loop selects and an outer loop estimates, which
is more expensive and the honest version when data is scarce.

## The part that gets skipped

Comparing on one seed. If two models differ by less than the spread across
seeds, there is no result — just noise with a preference. Several runs before
declaring a winner, and report the spread alongside the mean.

## Simplicity as a tiebreak

When scores are within noise, the simpler model wins by default. It is cheaper
to run, easier to debug, and less likely to have found something that will not
survive contact with new data.
