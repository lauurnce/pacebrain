# Training Compute-Optimal Large Language Models — Hoffmann et al., 2022

Revises the earlier scaling conclusions: for a fixed compute budget, models had
been made too large and trained on too little data.

Compute-optimal training scales parameters and training tokens together, roughly
in proportion, rather than spending most of the budget on size.

The reason this matters beyond language modelling is the pattern — an accepted
scaling recipe was wrong because one axis had been held fixed while fitting the
others.
