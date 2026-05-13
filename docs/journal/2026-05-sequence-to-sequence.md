# Study notes — sequence models and what they output

Sorting out a confusion I kept hitting: the same recurrent layer supports
several quite different tasks, and the difference is entirely in which outputs
you keep.

**Sequence to one.** Read the whole sequence, use the final hidden state.
Classification, or predicting a single number from a history.

**Sequence to sequence, aligned.** Keep the output at every timestep. One
prediction per input step — per-segment pace across a race, for example.

**Sequence to sequence, unaligned.** Input and output lengths differ, so an
encoder produces a representation and a separate decoder generates from it.
Translation is the standard case.

## The shape consequence

An `nn.LSTM` returns both the full output sequence and the final states. Which
one you take *is* the choice above; the layer itself does not change. That
seems obvious written down and was genuinely not obvious while reading code.

## Variable lengths

Real sequences differ in length, so batching requires padding — and then the
model must be told to ignore the padding, or it learns from timesteps that do
not exist. Packing utilities exist for exactly this. Fixed-length sequences
avoid the problem entirely, which is a reasonable simplification to start with
as long as it is a known one rather than an accident.
