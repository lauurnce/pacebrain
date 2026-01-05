# Backpropagation

Applying the chain rule backwards through the computation graph to get the loss gradient with respect to every parameter in one pass. It is not an optimiser: it produces the gradients, and a separate update rule decides what to do with them.

Backpropagation interacts strongly with dataset size here; behaviours that are footnote-level on benchmarks are headline-level on a few hundred athlete-weeks.
