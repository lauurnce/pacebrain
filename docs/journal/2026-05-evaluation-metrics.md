# Study notes — picking a metric

The metric is not a reporting decision made after the fact. It defines what
"wrong" means, and the model optimises whatever it is given.

## For regression

**MSE** squares the errors, so one large miss dominates many small ones.
Convenient to differentiate, which is why it is the default training loss.

**RMSE** is MSE square-rooted, back in the units of the target and still
outlier-sensitive.

**MAE** averages absolute errors. Robust to outliers and the easiest to explain
to someone who is not looking at the model — "wrong by about five minutes" is
immediately meaningful.

## The pairing that is actually informative

Report MAE and RMSE together and read the ratio. For purely gaussian errors
RMSE/MAE is `sqrt(pi/2) = 1.253`. Close to that means the misses are uniformly
distributed; well above it means a few large errors dominate and the average is
hiding a tail.

That is one number that says something about the *shape* of the error
distribution, which no single metric does on its own.

## The habit worth keeping

Read every score against the best achievable value rather than against zero. If
the target carries irreducible noise, some error is impossible to remove, and a
percentage improvement quoted without that context can look impressive while
being close to meaningless.
