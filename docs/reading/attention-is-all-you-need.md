# Attention Is All You Need — Vaswani et al., 2017

Replaces recurrence entirely with self-attention plus feed-forward blocks.

The argument that stuck with me is not accuracy, it is parallelism: recurrence
forces sequential computation over the sequence, attention does not, so training
throughput improves by a large factor at the same hardware. Path length between
any two positions becomes constant rather than linear in distance.

Positional encoding is load-bearing rather than incidental — attention is
permutation-invariant, so without it word order carries no information at all.
