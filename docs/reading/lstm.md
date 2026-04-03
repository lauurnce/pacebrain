# Long Short-Term Memory — Hochreiter & Schmidhuber, 1997

Introduces a gated cell with a separate additive cell state.

The problem being solved is stated precisely: gradients through a recurrent
chain are repeatedly multiplied, so they vanish or explode exponentially in
sequence length. The cell state provides a path where information is added
rather than transformed, so the gradient survives.

Reading the original after using the layer clarified that the gates are the
mechanism and the additive path is the point.
