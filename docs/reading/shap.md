# A Unified Approach to Interpreting Model Predictions — Lundberg & Lee, 2017

Frames feature attribution using Shapley values from cooperative game theory,
which gives a set of desirable properties uniquely.

The theoretical grounding is the appeal, and the cost is computational —
exact Shapley values are exponential, so practical use depends on
model-specific approximations.

Correlated features still split credit between themselves in ways that look
like evidence, which is a property of attribution generally rather than of this
method.
