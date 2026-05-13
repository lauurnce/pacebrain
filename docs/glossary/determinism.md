# Determinism

The property that identical inputs and configuration produce identical outputs. Harder than setting a [[seed]] alone — parallel floating-point reductions and some GPU kernels vary between runs, so full determinism usually costs performance and has to be requested explicitly.
