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

GRU is the simpler cousin: it merges the cell and hidden state into one and
uses three gates instead of four, so at equal hidden size it needs about 25%
fewer parameters (13,889 vs 18,497 at hidden_size=64).

"Often reaches similar accuracy" is the usual claim, and on this problem it
is too modest — measured over four seeds the GRU wins every one, by 3.6% raw
RMSE and 12.8% once the noise floor is subtracted, while being cheaper and
faster. See reports/gru_vs_lstm.md. The default is still the LSTM, because
every existing checkpoint and write-up describes it; PacingConfig(cell="gru")
switches.
"""

from __future__ import annotations

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

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch, seq_len, 1)"""
        return _forward_recurrent(self.lstm, self.head, x, lengths)


def _forward_recurrent(
    rnn: nn.Module,
    head: nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shared forward pass for the recurrent pacing models.

    Kept as a function rather than a base class on purpose. A base class
    would have to own the submodule under one name (`self.rnn`), which
    renames every key in the state_dict and breaks every checkpoint written
    so far. This way PacingLSTM keeps `lstm.*` and PacingGRU gets `gru.*`,
    and neither duplicates the packing logic.

    Args:
        lengths: optional (batch,) of true, unpadded sequence lengths.
            Required whenever x holds padded sequences of differing length --
            see the note below on why.
    """
    # nn.LSTM returns (output, (h_n, c_n)); nn.GRU returns (output, h_n).
    # Both are unpacked the same way here because only `output` is used:
    #   output: the hidden state at EVERY timestep -- what we want for
    #           per-segment prediction. Shape (batch, seq_len, hidden_size).
    #   the second element: only the FINAL state(s) -- what you'd use for a
    #           single summary prediction, like Day 4's finish time.
    if lengths is None:
        output, _ = rnn(x)
    else:
        # What padding does and does not break, precisely -- because the
        # intuitive answer is wrong and the tests pin down the real one:
        #
        # It does NOT corrupt the per-timestep outputs. These models are
        # unidirectional, so they are causal: the output at step t depends
        # only on steps 1..t, and the padding is all trailing. Feed a
        # padded batch straight in and the first 5 outputs of a 5 km race
        # are identical to running that race alone, to float precision.
        #
        # It DOES corrupt the final hidden state h_n, which ends up
        # describing 37 steps of zeros rather than the race (measured at
        # 0.134 max abs deviation on a 16-unit hidden state). Anything
        # with a summary head reading h_n is silently wrong without
        # packing. It would also corrupt every timestep if the model were
        # bidirectional, since the backward pass starts in the padding.
        # And it wastes the compute either way.
        #
        # So packing here buys correctness for h_n and for any future
        # bidirectional or summary variant, plus the saved steps -- not a
        # fix to the per-segment predictions, which were already right.
        #
        # pack_padded_sequence rewrites the batch as a flat buffer plus
        # per-timestep batch sizes, so the RNN simply stops early on
        # short sequences. enforce_sorted=False lets it handle an
        # unsorted batch by permuting internally.
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),  # must live on the CPU even for a CUDA batch
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = rnn(packed)
        # total_length pins the time dimension back to the padded width.
        # Without it the batch is trimmed to its own longest sequence,
        # which silently changes shape from batch to batch.
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )

    # nn.Linear broadcasts over all leading dims, so applying it to the
    # full (batch, seq_len, hidden_size) tensor gives a prediction at
    # every timestep in one call: -> (batch, seq_len, 1).
    #
    # Note the padded positions are NOT zero in the result: their hidden
    # states are zero, but the head adds its bias, so they come out as
    # b. They are meaningless either way and must be masked out of the
    # loss -- see masked_mse_loss().
    return head(output)


class PacingGRU(nn.Module):
    """
    GRU counterpart to PacingLSTM, with an identical interface.

    A GRU merges the LSTM's hidden and cell states into one, and has three
    gates instead of four. That means roughly 25% fewer parameters at the
    same hidden size, and on many sequence tasks it reaches comparable
    accuracy — which is the claim reports/gru_vs_lstm.md actually tests
    rather than repeats.

    Kept as a separate class rather than a `cell_type` flag on PacingLSTM so
    each state_dict says what architecture wrote it. A flag would let a GRU
    checkpoint load into a model built as an LSTM only to fail on shapes,
    with nothing in the file explaining why.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch, seq_len, 1)"""
        return _forward_recurrent(self.gru, self.head, x, lengths)


def length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Boolean (batch, max_len) mask that is True at real timesteps.

    Built by comparing a row of position indices against each length, which
    vectorises what would otherwise be a Python loop over the batch.
    """
    positions = torch.arange(max_len, device=lengths.device)
    return positions[None, :] < lengths[:, None]


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    MSE over real timesteps only.

    Plain mse_loss would average over padded positions too. That is not
    merely noise: padding is a constant, so the model can drive the loss down
    by predicting it, and because short races are padded most, the gradient
    is biased towards exactly the sequences with the least real data. The
    divisor here is the real timestep count, not batch * max_len.
    """
    mask = length_mask(lengths, pred.size(1)).unsqueeze(-1)
    squared_error = ((pred - target) ** 2) * mask
    return squared_error.sum() / mask.sum().clamp(min=1)


if __name__ == "__main__":
    # Smoke test: random batch of 4 races, 10 segments each, 6 features per segment.
    model = PacingLSTM()
    x = torch.randn(4, 10, 6)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"parameters:   {n_params}")

    # Variable-length path: a 5 km, a 10 km and a marathon in one batch.
    lengths = torch.tensor([5, 10, 42])
    padded = torch.randn(3, 42, 6)
    out = model(padded, lengths=lengths)
    print(f"\npadded input:  {tuple(padded.shape)}  lengths={lengths.tolist()}")
    print(f"packed output: {tuple(out.shape)}")
