# Why Should I Trust You? — Ribeiro et al., 2016

Explains individual predictions by fitting a simple local model to perturbations
around one input.

Local rather than global is the key move: a model can be globally complex and
locally almost linear, so a faithful local explanation is achievable where a
global one is not.

The limitation is that the explanation depends on the perturbation scheme, so
two reasonable choices can produce different explanations of the same
prediction.
