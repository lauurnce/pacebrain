# GRU

Gated recurrent unit: a recurrent cell that folds the LSTM's hidden and cell state into a single hidden state and uses only a reset gate and an update gate. The update gate decides how much of the previous state to keep versus replace, and the reset gate decides how much of it to consult when computing the candidate new state. With fewer gates it has fewer parameters than an LSTM of the same width and is often about as accurate, which makes it the usual cheaper alternative to try.

## See also

- [Activation function](activation-function.md)
- [Sigmoid](sigmoid.md)
- [Vanishing gradient](vanishing-gradient.md)

In PaceBrain, `PacingGRU` (`src/pacebrain/seq_models.py`) has the same interface as `PacingLSTM` and is selected with `PacingConfig(cell="gru")`. At hidden size 64 it has 13,889 parameters against the LSTM's 18,497, about 25% fewer. It is a separate class rather than a flag on the LSTM so that a saved state dict records which architecture wrote it.
