# Study notes — dropping inputs on purpose

More features is not free. Each adds parameters, dilutes the signal, and gives
the model another opportunity to fit noise.

**Filter methods.** Score each feature independently — correlation, mutual
information — and keep the top ones. Fast, and blind to interactions, so it
discards features that only matter jointly.

**Wrapper methods.** Train repeatedly with subsets, keeping what helps.
Accurate and expensive, and prone to overfitting the selection to the
validation split.

**Embedded methods.** Selection as part of fitting, which is what L1 does.
Usually the best cost/benefit.

## The caution

Selection is a decision made from data, so it belongs inside the
cross-validation loop. Selecting features on the full dataset and then
cross-validating the model reports a score that has already seen the answer,
and it is a very easy mistake to make because the code looks correct.
