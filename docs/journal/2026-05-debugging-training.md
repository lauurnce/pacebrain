# Study notes — when training does not work

An ordered list, cheapest checks first, because the expensive hypotheses are
almost never right.

1. **Loss is NaN.** Learning rate too high, or a log/divide of zero. Lower the
   rate by 10x first; if it survives, it was the rate.
2. **Loss does not move at all.** Check the optimiser actually received the
   parameters, that `zero_grad` is not clearing after `backward`, and that
   nothing is under `no_grad`.
3. **Loss falls then plateaus high.** Underfitting. More capacity, more
   features, or a longer run.
4. **Train falls, validation rises.** Overfitting. Regularise, stop earlier, or
   get more data.
5. **Validation is better than train.** Almost always dropout being active in
   one and not the other, or a leak.

## The single best test

Overfit one batch deliberately. A model that cannot drive the loss to near zero
on ten examples has a bug, not a tuning problem — and this takes seconds where
diagnosing it on the full set takes hours.
