# Study notes — sequence models

Where the architecture itself encodes an assumption: order matters.

## Why not just flatten

A sequence of 10 steps with 6 features each could be flattened to a 60-length
vector and fed to an MLP. That works, badly, for two reasons:

- **No weight sharing.** The MLP learns a separate parameter for "feature 3 at
  step 7" and another for "feature 3 at step 8", with no notion that they are
  the same measurement at different times. Everything must be relearned per
  position.
- **Fixed length.** A flattened input requires every sequence to be exactly the
  same length.

A recurrent model applies the *same* transformation at every step and carries
state forward, which is a much better match to how the data is actually
generated.

## RNN

Maintain a hidden state `h`, and at each step compute `h_t = f(h_{t-1}, x_t)`.
The state is a running summary of everything seen so far.

The problem is training. Backpropagating through many steps multiplies many
derivatives together, so the gradient either vanishes or explodes. Vanishing is
the common case, and its practical effect is that the model cannot learn
dependencies more than a few steps apart — which defeats the purpose.

## LSTM

Adds a separate cell state alongside the hidden state, plus gates controlling
what enters, what is discarded, and what is emitted.

The key structural difference is that the cell state passes through with only
elementwise multiplication and addition rather than a full nonlinearity at each
step. That gives gradients a path backwards that does not shrink as fast, which
is what makes long-range dependencies learnable.

Mental model: the cell state is a conveyor belt, and the gates decide what gets
placed on it and taken off.

## GRU

Same idea with fewer gates and no separate cell state. Fewer parameters, trains
faster, typically comparable performance. Reasonable default when the LSTM's
extra capacity is not obviously needed.

## Output shapes

Two distinct uses, and confusing them is a common bug:

- **Per-step output** — one prediction per timestep. Output shape
  `(batch, seq_len, out)`.
- **Sequence summary** — the final hidden state only, for one prediction per
  sequence. Shape `(batch, out)`.

Also: frameworks differ on whether the batch dimension comes first. Getting it
wrong does not always error — it can silently treat the batch as the sequence
axis, which trains and produces nonsense.

## Variable length

Real sequences differ in length. Padding to a common length is standard, but
the model must then be told which steps are padding, or it will happily learn
from the padded zeros. Packing utilities exist for exactly this.

## Where this would apply

Anything where step N depends on steps before it and the dependency is the
point rather than a nuisance. Pacing across a race is an obvious candidate —
fatigue accumulates, so how the back half goes depends on how the front half
was run.
