# Study notes — categorical features

Numeric models need numbers. How a category becomes a number is a modelling
decision with real consequences.

## Integer encoding

Map each category to an integer. Compact, and wrong for anything unordered —
it asserts that category 3 sits between 2 and 4, and that 4 is twice 2. A model
will happily use that false ordering.

Fine for genuinely ordinal data where the order is real.

## One-hot encoding

One binary column per category. No false ordering, and the standard choice for
low cardinality.

Costs a column per level, so high-cardinality features explode the input width,
and each resulting column is mostly zeros. With more columns than useful signal,
the model has more opportunity to overfit than to learn.

## Target encoding

Replace a category with the mean target for that category. One column
regardless of cardinality, and it carries real signal.

Also **the easiest way to leak**. Computing the mean using rows in the
validation split puts target information directly into the features. It must be
computed on training data only, and even then out-of-fold within training,
because a category appearing once takes that row's own target as its encoding.

Powerful and genuinely dangerous. Worth using carefully or not at all.

## Learned embeddings

Map each category to a small dense vector learned during training. Handles high
cardinality gracefully and can capture similarity between categories — two that
behave alike end up nearby.

This is what makes networks attractive for categorical-heavy tabular data.
Needs enough examples per category to learn anything.

## Unseen categories

Every scheme needs an answer for a value not present at training time. One-hot
gives all zeros, integer encoding gives an index out of range, embeddings need
a reserved unknown slot. Deciding this up front prevents a crash at inference
on the first real user.

## For this project

Race distance is the only categorical-ish feature, and it is genuinely numeric
and ordered — 42.2 really is more than 21.1, and the relationship to finish time
is monotonic. Keeping it numeric is correct here, and a one-hot version would
discard the ordering the model can exploit.
