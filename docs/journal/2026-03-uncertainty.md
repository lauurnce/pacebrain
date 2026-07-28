# Study notes — uncertainty and prediction intervals

A single number implies a confidence the model does not have.

## Two sources

**Aleatoric** — noise inherent in the process. The same runner on the same
training runs the same race differently depending on sleep, heat and how the
day goes. No amount of data removes this.

**Epistemic** — uncertainty from not having enough data. A runner unlike
anything in the training set should produce a less confident prediction. This
one shrinks as data grows.

The distinction is practical: aleatoric sets a floor on achievable accuracy,
epistemic tells you where to collect more data.

## Getting an interval

**Quantile regression.** Train separate models with quantile loss at `q=0.1`
and `q=0.9`. Directly gives an 80% band. Simple, and no distributional
assumption.

**Ensembles.** Train several models with different seeds and use the spread of
their predictions. Captures epistemic uncertainty well. Costs n times the
training.

**Dropout at inference.** Leave dropout on, predict many times, take the
variance. Cheap approximation to an ensemble. Requires deliberately overriding
the usual eval-mode discipline, which makes it easy to do by accident and
mistake for a bug.

**Residual quantiles.** Compute the spread of validation errors and apply it as
a fixed band. Crude and assumes constant error, but nearly free and better than
nothing.

## Why it matters for advice

"Your predicted marathon is 3:25" and "3:20 to 3:35, most likely 3:25" lead to
different pacing decisions. The second is honest about what the model knows,
and a user can act on it.

Given asymmetric cost — going out too fast is far worse than too slow — the
sensible recommendation is not the centre of the interval anyway. It is
somewhere on the conservative side of it, which you cannot express at all
without the interval.

## The failure to avoid

Reporting a mean error of 4 minutes and letting a reader assume every
prediction is within 4 minutes. Average error says nothing about the worst
case, and the worst case is what destroys trust.
