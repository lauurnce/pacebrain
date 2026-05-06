# Study notes — tensors, and why not just numpy

A tensor is an n-dimensional array with two things numpy does not have: it can
live on a GPU, and it can remember how it was computed.

## The shape discipline

Almost every error in this material is a shape error, so the habit worth
building early is writing the expected shape next to every line that changes
one. `(batch, features)` going into a linear layer, `(batch, out)` coming out.
The library will happily broadcast two things that should never have met.

## dtype matters more than expected

Mixing float32 and float64 silently promotes, and on a GPU that is a real cost.
Integer division where a float was meant produces a wrong answer rather than an
error. Being explicit at construction is cheaper than debugging it later.

## The part that is actually new

`requires_grad=True` turns an array into a node in a graph. From then on every
operation records what produced it, which is what makes the backward pass
possible. That is the whole difference, and everything else is bookkeeping.
