# GRU

Gated recurrent unit: a recurrent cell that folds the LSTM's hidden and cell state into a single hidden state and uses only a reset gate and an update gate. The update gate decides how much of the previous state to keep versus replace, and the reset gate decides how much of it to consult when computing the candidate new state. With fewer gates it has fewer parameters than an LSTM of the same width and is often about as accurate, which makes it the usual cheaper alternative to try.

## See also

- [Activation function](activation-function.md)
- [Sigmoid](sigmoid.md)
- [Vanishing gradient](vanishing-gradient.md)
