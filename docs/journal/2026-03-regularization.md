# Study notes — regularisation

What to do when training error keeps falling and validation error does not.

## The signal

Watch the *gap* between training and validation loss over epochs, not either
curve alone. Training loss falling while validation loss flattens or rises is
overfitting, and the epoch where validation bottoms out is the useful model.

## Early stopping

Stop when validation loss has not improved for `patience` epochs, and keep the
best checkpoint rather than the last one.

This is the cheapest regularisation there is: no hyperparameter beyond patience,
no change to the model, and it also removes the need to guess an epoch count.
The only real subtlety is that the "best" checkpoint must actually be saved when
it occurs — saving at the end of training defeats the entire mechanism.

## Weight decay (L2)

Add a penalty proportional to the sum of squared weights. Large weights are
what let a model produce wild outputs from small input changes, so penalising
them favours smoother functions.

Note that in Adam, naive L2 and true weight decay are not the same thing
because the adaptive denominator scales the penalty too — hence AdamW. A detail
that matters more at scale than at the size I am working at, but worth knowing
the distinction exists.

## Dropout

Randomly zero a fraction of activations during training. Each forward pass
trains a slightly different sub-network, and no single unit can be relied upon,
which discourages brittle co-adaptation between units.

Two things that are easy to get wrong:

- It must be **off at inference**. Frameworks handle this via a train/eval mode
  toggle, but forgetting to flip it produces nondeterministic predictions that
  look like a bug elsewhere.
- Rates around 0.1–0.5 are typical. High dropout on a small network mostly just
  destroys capacity.

## More data

Strictly better than any of the above, and usually unavailable. Augmentation is
the substitute where the domain allows it — obvious for images, much less
obvious for tabular training data, where most transformations would change the
meaning of the row.

## Reducing capacity

The bluntest instrument: fewer layers or narrower ones. Underrated, though —
if a smaller model matches a larger one, the smaller one is simply better. It
trains faster, is easier to reason about, and has less room to memorise.

## Order of operations

1. Get a train/validation split that is honest.
2. Add early stopping.
3. Only then reach for dropout or weight decay, one at a time.

Adding three regularisers simultaneously and observing an improvement teaches
nothing about which one worked.

## Next

Sequence models — where the architecture itself encodes an assumption about
the data.
