# Decoupled Weight Decay — Loshchilov & Hutter, 2017

Shows that L2 regularisation and weight decay are equivalent for plain SGD but
not for adaptive methods like Adam, where the L2 term gets scaled by the
per-parameter learning rate and stops behaving like decay.

AdamW applies the decay directly to the weights instead, outside the adaptive
scaling.

Useful because it explains a years-long confusion in library defaults rather
than proposing something new — the bug was conceptual, not numerical.
