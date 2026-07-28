# Study notes — hyperparameters and how to tune them

Follow-on from the regularisation notes, same evening. Those introduced knobs
(patience, dropout rate, weight decay) without saying how to set any of them.

## Parameters vs hyperparameters

Parameters are learned by gradient descent — weights and biases. Hyperparameters
are chosen before training and are not differentiable with respect to the loss,
so there is no gradient to follow. They have to be searched.

## Which ones actually matter

Roughly in order of impact:

1. **Learning rate.** Dominant. Wrong by an order of magnitude and nothing else
   you tune will rescue it.
2. **Model capacity** — depth and width.
3. **Regularisation strength** — dropout rate, weight decay.
4. **Batch size**, which interacts with the learning rate rather than acting
   independently.
5. Everything else, mostly noise at small scale.

Tuning items 3–5 while item 1 is wrong is a common way to waste an afternoon.

## Search strategies

- **Manual.** Perfectly reasonable when there are two or three knobs and
  training takes seconds. Keeping a written log of what was tried matters more
  than the strategy.
- **Grid search.** Every combination. Cost is exponential in the number of
  hyperparameters, and it wastes most of its budget varying things that do not
  matter.
- **Random search.** Sample combinations at random. Counterintuitively better
  than grid at equal budget: if only one hyperparameter matters, grid tries the
  same handful of values for it repeatedly, while random tries a different value
  every time.
- **Bayesian / successive halving.** Worth it when a single run costs hours.
  Overkill here.

## The discipline that actually matters

**Tune against validation, report against a set you did not tune on.** Every
decision made by looking at a score leaks a little information about that data.
Do it enough times and the validation score becomes optimistic in exactly the
way the original train/test split was meant to prevent.

With a small project and few decisions this is a minor effect, but it is worth
knowing that "I tried thirty configurations and picked the best validation
score" makes that score an overestimate.

## Fixing the seed

Set it, and record it. Two runs that differ in both a hyperparameter and the
random seed cannot be compared — you cannot tell which change caused the
difference.

The corollary is worth stating: if changing the seed moves the result as much
as changing the hyperparameter, then the hyperparameter did nothing and the
apparent improvement was noise.

## Practical starting point

Adam at `1e-3`, a small network, dropout off, early stopping on. Get that
working end to end first. Every knob added before the pipeline is correct is a
knob that might be hiding a bug rather than tuning a model.
