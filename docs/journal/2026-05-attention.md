# Study notes — attention

A recurrent model compresses everything it has seen into one fixed hidden
state, so information from early in a long sequence has to survive many steps
to still be there at the end. Attention removes the bottleneck by letting the
model look directly at every position.

## The mechanism

Each position produces a query, a key and a value. Scores are the dot product
of one query against all keys, softmaxed into weights, and the output is the
weighted sum of values. That is the entire operation.

The scaling by `sqrt(d_k)` before the softmax is not decoration: for large
dimensions the dot products grow, the softmax saturates, and gradients vanish.

## What it buys

Any position can reach any other in one step rather than many, so long-range
dependencies stop degrading with distance. And because every position is
computed independently, the whole thing parallelises across the sequence —
which is the practical reason transformers displaced recurrence, more than
accuracy alone.

## The cost

Attention is quadratic in sequence length in both time and memory. For the
sequence lengths in this project that is irrelevant, but it is the reason a
large fraction of the literature is about approximating it.
