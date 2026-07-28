# Study notes — loss functions for regression

The loss defines what "wrong" means. Choosing it is a modelling decision, not a
default.

## MSE

`mean((pred - true)^2)`. Penalises quadratically, so one prediction off by 10
costs the same as a hundred off by 1. Clean derivative, converges smoothly.

Its optimum is the conditional **mean**. If the target is skewed, the mean is
not the value you want predicted.

## MAE

`mean(|pred - true|)`. Linear penalty, so outliers do not dominate. Its optimum
is the conditional **median**, which is the more robust summary.

Cost: the gradient is constant regardless of how wrong you are, and undefined
at zero. Convergence is less smooth near the optimum.

## Huber

Quadratic near zero, linear beyond a threshold `delta`. MSE's smooth gradients
where errors are small, MAE's robustness where they are large. One extra
hyperparameter, which is the price.

## Quantile loss

Weights over- and under-prediction differently. Weight `q` on under-prediction
and `1-q` on over-prediction, and the optimum becomes the `q`th quantile.

This is the interesting one for any problem with **asymmetric cost**. If
predicting too high is worse than predicting too low, the loss should say so
rather than being corrected afterwards by hand. It also gives prediction
intervals directly: fit `q=0.1` and `q=0.9` and you have an 80% band.

## Practical pairing

Train on MSE or Huber, report MAE. Optimisation and communication have
different requirements and there is no rule that one metric must serve both.

The thing to avoid is reporting the training loss as though it were a result.
"Val MSE 30.5" means nothing to a reader; the same model at "wrong by 4.1
minutes on average" means something immediately.
