# Grid search

Exhaustively evaluating every combination in a hyperparameter grid. Cost grows multiplicatively with each added axis, which is why random search often finds better settings within the same budget.

Grid search interacts strongly with dataset size here; behaviours that are footnote-level on benchmarks are headline-level on a few hundred athlete-weeks.
