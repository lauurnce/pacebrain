# Study notes — making a run repeatable

Setting a seed is the first step and not the whole job.

## What else moves

Library-level nondeterminism in parallel reductions. GPU kernels that pick
algorithms by benchmarking at runtime. Dataloader worker ordering. Dictionary
and set iteration in some contexts. Any of these produces a different result
from identical code and identical seed.

## The checklist that works

Seed every generator in use, not just one. Request deterministic algorithms
explicitly where the framework offers it. Pin dependency versions. Record the
commit hash with the result.

## The cost

Full determinism is usually slower, sometimes considerably, because the fastest
kernel for an operation is often the nondeterministic one. That is a reasonable
trade for a result being published and a bad one for exploration.

## The distinction worth keeping

Reproducible is not the same as robust. A result that only holds for one seed
is reproducible and worthless. Both properties are wanted, and they are
measured differently.
