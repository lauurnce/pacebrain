# LSTM

Long short-term memory: a recurrent neural network cell that keeps a separate cell state alongside its hidden state and controls both with three gates. The input gate decides what new information is written, the forget gate what is erased, and the output gate what is exposed as the hidden state. Because the cell state is updated additively rather than rewritten at every step, gradients survive across many timesteps, which is what lets an LSTM learn dependencies that a plain recurrent network loses to the vanishing gradient.

## See also

- [Backpropagation](backpropagation.md)
- [Dropout](dropout.md)
- [Exploding gradient](exploding-gradient.md)
- [Gradient clipping](gradient-clipping.md)
- [Vanishing gradient](vanishing-gradient.md)

In PaceBrain, `PacingLSTM` (`src/pacebrain/seq_models.py`) runs an `nn.LSTM` over a race one segment at a time and puts a linear head on the hidden state at every step, so it predicts a pace per segment rather than a single finish time. The defaults are one layer and a hidden size of 64, which is 18,497 parameters; `dropout` only takes effect between stacked layers.
