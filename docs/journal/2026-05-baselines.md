# Study notes — baselines

A model score means nothing alone. It means something against a reference, and
choosing that reference well is most of the work.

## A ladder, not a single rung

The useful pattern is several baselines ordered by how much each is allowed to
see:

1. Predict the mean — ignores every feature. Anything that cannot beat this has
   learned nothing at all.
2. A domain formula — whatever practitioners already use.
3. Linear regression on the same features the model gets.
4. The model.

The *gaps* between consecutive rungs localise where any advantage comes from.
One baseline gives a number; a ladder gives an explanation.

## The trap

A baseline only says something if it had a fair attempt at the problem. Handing
a formula an input it was never designed for and then reporting how badly it
did measures the handicap, not the model. That comparison feels flattering and
is worthless.

## The other trap

If a linear model on the same features is competitive, the extra capacity is
not earning its place — and if a simple feature transform closes the gap, the
honest conclusion is that the model spent its training rediscovering something
the transform states outright.
