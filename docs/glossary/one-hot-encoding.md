# One-hot encoding

Representing a categorical value as a vector with a single 1 and zeros elsewhere, so no ordering is implied between categories. It costs one dimension per level, which is why high-cardinality columns are usually given an [[embedding]] instead.

One hot encoding interacts strongly with dataset size here; behaviours that are footnote-level on benchmarks are headline-level on a few hundred athlete-weeks.
