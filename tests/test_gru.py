"""
Tests for PacingGRU and the selectable recurrent cell.

The GRU shares its forward pass with the LSTM via _forward_recurrent, so most
of what matters is that the sharing did not change LSTM behaviour and that
the two models really are interchangeable at the interface.
"""

import pytest
import torch

from pacebrain import train_pacing
from pacebrain.config import PacingConfig
from pacebrain.seq_data import SEQ_FEATURES, pad_collate
from pacebrain.seq_models import PacingGRU, PacingLSTM, masked_mse_loss


@pytest.fixture
def ragged_batch():
    torch.manual_seed(1)
    return [
        (torch.randn(5, len(SEQ_FEATURES)), torch.randn(5, 1)),
        (torch.randn(10, len(SEQ_FEATURES)), torch.randn(10, 1)),
        (torch.randn(42, len(SEQ_FEATURES)), torch.randn(42, 1)),
    ]


# ---------------------------------------------------------------------------
# interface parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [PacingLSTM, PacingGRU])
def test_output_shape_matches_input_sequence(cls):
    model = cls(input_size=len(SEQ_FEATURES), hidden_size=16)
    assert model(torch.randn(4, 10, len(SEQ_FEATURES))).shape == (4, 10, 1)


@pytest.mark.parametrize("cls", [PacingLSTM, PacingGRU])
def test_packed_batch_matches_running_each_race_alone(cls, ragged_batch):
    """The property #22 established for the LSTM must hold for the GRU too."""
    torch.manual_seed(0)
    model = cls(input_size=len(SEQ_FEATURES), hidden_size=16)
    model.eval()
    X, _, lengths = pad_collate(ragged_batch)
    batched = model(X, lengths=lengths)

    for i, (X_i, _) in enumerate(ragged_batch):
        alone = model(X_i.unsqueeze(0))[0]
        assert torch.allclose(batched[i, : lengths[i]], alone, atol=1e-6)


@pytest.mark.parametrize("cls", [PacingLSTM, PacingGRU])
def test_works_with_the_masked_loss(cls, ragged_batch):
    model = cls(input_size=len(SEQ_FEATURES), hidden_size=16)
    X, y, lengths = pad_collate(ragged_batch)
    loss = masked_mse_loss(model(X, lengths=lengths), y, lengths)
    assert torch.isfinite(loss) and loss.item() > 0


# ---------------------------------------------------------------------------
# the parameter claim
# ---------------------------------------------------------------------------

def test_gru_has_about_a_quarter_fewer_parameters():
    """
    An LSTM has four gates and two states; a GRU has three gates and one. At
    equal hidden size that is roughly 3/4 of the weights — the claim the
    docstring makes, asserted rather than repeated.
    """
    kw = dict(input_size=len(SEQ_FEATURES), hidden_size=64)
    lstm_n = sum(p.numel() for p in PacingLSTM(**kw).parameters())
    gru_n = sum(p.numel() for p in PacingGRU(**kw).parameters())
    assert gru_n < lstm_n
    assert gru_n / lstm_n == pytest.approx(0.75, abs=0.03)


# ---------------------------------------------------------------------------
# state_dict compatibility — the reason this is two classes, not a flag
# ---------------------------------------------------------------------------

def test_lstm_state_dict_keys_are_unchanged_by_the_refactor():
    """
    Sharing the forward pass must not rename the submodule. A base class
    owning `self.rnn` would have rewritten every key and invalidated every
    pacing checkpoint written so far.
    """
    keys = list(PacingLSTM(input_size=len(SEQ_FEATURES)).state_dict())
    assert all(k.startswith(("lstm.", "head.")) for k in keys), keys


def test_gru_and_lstm_state_dicts_are_not_interchangeable():
    """Distinct prefixes mean a mismatched load fails loudly, not subtly."""
    kw = dict(input_size=len(SEQ_FEATURES), hidden_size=16)
    with pytest.raises(RuntimeError):
        PacingLSTM(**kw).load_state_dict(PacingGRU(**kw).state_dict())


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------

def test_default_cell_is_lstm():
    """Every existing checkpoint and write-up describes the LSTM."""
    assert PacingConfig().cell == "lstm"


@pytest.mark.parametrize("cell,expected", [("lstm", PacingLSTM), ("gru", PacingGRU)])
def test_training_builds_the_requested_cell(cell, expected, tmp_path):
    cfg = PacingConfig(
        cell=cell, n_races=40, epochs=2, batch_size=8,
        checkpoint_path=str(tmp_path / "m.pt"), plot_path=str(tmp_path / "p.png"),
    )
    assert isinstance(train_pacing.train(cfg), expected)


def test_unknown_cell_is_rejected_with_a_useful_message(tmp_path):
    cfg = PacingConfig(
        cell="transformer", n_races=40, epochs=1,
        checkpoint_path=str(tmp_path / "m.pt"), plot_path=str(tmp_path / "p.png"),
    )
    with pytest.raises(ValueError, match="Unknown cell"):
        train_pacing.train(cfg)
