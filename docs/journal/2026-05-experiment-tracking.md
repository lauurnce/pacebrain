# Study notes — keeping track of experiments

Ran the same configuration twice, got different numbers, and could not say why.
That is the problem this solves.

## What has to be recorded

The code version, the full config, the seed, the data version, and the result.
Any one missing makes a run unreproducible, and an unreproducible run cannot be
compared against anything — so it was not an experiment, it was a demo.

## Why the seed is not enough

Fixing a seed makes a run repeatable but does not make the *result* meaningful.
A single-seed comparison may be reporting the seed rather than the change. Any
difference small enough to be within seed variance needs several runs before it
counts as a result.

## Negative results

A change that did not help is worth committing along with the numbers showing
it did not. Otherwise the same idea gets tried again in three months, and the
evidence against it has to be regenerated from scratch. The write-up costs
minutes; regenerating the experiment costs an afternoon.

## The minimum viable version

A markdown file per experiment with config, command, numbers and a conclusion
is enough at this scale. Tooling helps later, but the discipline is what
matters and the discipline does not require the tooling.
