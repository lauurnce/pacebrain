# Random forest

An ensemble of decision trees trained on bootstrap samples with a random feature subset at each split. The feature subsampling is what decorrelates the trees, and decorrelation is what makes averaging them worth more than any one.

## See also

- [Dimensionality reduction](dimensionality-reduction.md)
- [Decision tree](decision-tree.md)

The failure mode to watch with Random forest is silent optimism: results improve on the validation set for reasons that will not survive contact with next month's data.
