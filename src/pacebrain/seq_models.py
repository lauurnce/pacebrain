"""
Sequence model definitions for pacing prediction.

An LSTM (Long Short-Term Memory) is an RNN that carries a hidden state h_t
and a cell state c_t from one timestep to the next, with gates that learn
what to remember and what to forget. That running memory is exactly what
lets it model fatigue accumulating over a race.

Why order matters: the prediction for segment 7 can depend on what happened
in segments 1-6. An MLP on flattened features cannot express that with
shared weights -- the LSTM applies the SAME weights at every timestep and
passes context forward through its state.

(GRU is the simpler cousin: it merges the cell and hidden state, has fewer
parameters, and often reaches similar accuracy.)
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn as nn


class PacingLSTM(nn.Module):
    """
    LSTM that predicts a pace value for every segment of a race.

    Input  x: (batch, seq_len, input_size)  one feature vector per segment
    Output  : (batch, seq_len, 1)           one predicted pace per segment

    Args:
        input_size:  number of features per segment
        hidden_size: width of the LSTM's hidden state
        num_layers:  stacked LSTM layers (each feeds the next)
        dropout:     dropout between stacked layers (only applies if num_layers > 1)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # batch_first=True means tensors are (batch, seq, features)
        # instead of PyTorch's default (seq, batch, features).
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch warns if dropout is set with a single layer, since it
            # only applies BETWEEN stacked layers -- so gate it.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # One linear head maps each hidden state to a single pace value.
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch, seq_len, 1)"""
        # nn.LSTM returns (output, (h_n, c_n)):
        #   output: the hidden state at EVERY timestep -- what we want for
        #           per-segment prediction. Shape (batch, seq_len, hidden_size).
        #   h_n, c_n: only the FINAL states -- what you'd use for a single
        #           summary prediction, like Day 4's finish time.
        output, _ = self.lstm(x)

        # nn.Linear broadcasts over all leading dims, so applying it to the
        # full (batch, seq_len, hidden_size) tensor gives a prediction at
        # every timestep in one call: -> (batch, seq_len, 1).
        return self.head(output)


if __name__ == "__main__":
    # Smoke test: random batch of 4 races, 10 segments each, 6 features per segment.
    model = PacingLSTM()
    x = torch.randn(4, 10, 6)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"parameters:   {n_params}")
