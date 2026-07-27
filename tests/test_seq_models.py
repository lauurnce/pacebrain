"""
Tests for seq_models.py — PacingLSTM output shapes, stacked-layer and
hidden-size wiring, the dropout gate, and the PacingConfig / seq_data.py
feature-count invariant.
"""


import warnings

import pytest
import torch

from pacebrain.config import PacingConfig
from pacebrain.seq_data import N_SEGMENTS, SEQ_FEATURES, make_seq_datasets
from pacebrain.seq_models import PacingLSTM


def test_forward_output_shape():
    """(batch, seq_len, input_size) -> (batch, seq_len, 1), one pace per segment."""
    model = PacingLSTM()
    out = model(torch.randn(4, N_SEGMENTS, len(SEQ_FEATURES)))
    assert out.shape == (4, N_SEGMENTS, 1), f"got {tuple(out.shape)}"


@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_variable_batch_size(batch_size):
    """The same weights handle any batch size, including a single race."""
    model = PacingLSTM()
    out = model(torch.randn(batch_size, N_SEGMENTS, len(SEQ_FEATURES)))
    assert out.shape == (batch_size, N_SEGMENTS, 1)


@pytest.mark.parametrize("seq_len", [1, 5, 25])
def test_variable_sequence_length(seq_len):
    """An LSTM shares weights across timesteps, so seq_len is not fixed."""
    model = PacingLSTM()
    out = model(torch.randn(2, seq_len, len(SEQ_FEATURES)))
    assert out.shape == (2, seq_len, 1)


def test_hidden_size_is_wired_through():
    model = PacingLSTM(hidden_size=32)
    assert model.lstm.hidden_size == 32
    assert model.head.in_features == 32
    assert model.head.out_features == 1


def test_input_size_is_wired_through():
    model = PacingLSTM(input_size=4)
    assert model.lstm.input_size == 4
    assert model(torch.randn(2, N_SEGMENTS, 4)).shape == (2, N_SEGMENTS, 1)


def test_batch_first_is_enabled():
    """Tensors are (batch, seq, features), not PyTorch's default (seq, batch, features)."""
    assert PacingLSTM().lstm.batch_first is True


def test_num_layers_stacks_parameter_sets():
    """Each stacked layer adds its own ih/hh weight and bias tensors."""
    model = PacingLSTM(num_layers=3)
    assert model.lstm.num_layers == 3
    names = [n for n, _ in model.lstm.named_parameters()]
    for layer in range(3):
        assert f"weight_ih_l{layer}" in names
        assert f"weight_hh_l{layer}" in names
    assert "weight_ih_l3" not in names


def test_stacked_layers_have_more_parameters():
    one = sum(p.numel() for p in PacingLSTM(num_layers=1).parameters())
    two = sum(p.numel() for p in PacingLSTM(num_layers=2).parameters())
    assert two > one


def test_dropout_gated_off_for_single_layer():
    """
    LSTM dropout only applies BETWEEN stacked layers, so a single-layer
    model must zero it out rather than let PyTorch warn.
    """
    model = PacingLSTM(num_layers=1, dropout=0.3)
    assert model.lstm.dropout == 0.0


def test_single_layer_dropout_emits_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PacingLSTM(num_layers=1, dropout=0.5)
    assert caught == [], f"unexpected warnings: {[str(w.message) for w in caught]}"


def test_dropout_applied_for_stacked_layers():
    model = PacingLSTM(num_layers=2, dropout=0.3)
    assert model.lstm.dropout == pytest.approx(0.3)


def test_eval_mode_is_deterministic():
    torch.manual_seed(0)
    model = PacingLSTM(num_layers=2, dropout=0.5)
    model.eval()
    x = torch.randn(4, N_SEGMENTS, len(SEQ_FEATURES))
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_train_mode_dropout_perturbs_stacked_output():
    """Between-layer dropout is live in train() for a stacked model."""
    torch.manual_seed(0)
    model = PacingLSTM(num_layers=2, dropout=0.5)
    model.train()
    x = torch.randn(4, N_SEGMENTS, len(SEQ_FEATURES))
    with torch.no_grad():
        assert not torch.equal(model(x), model(x))


def test_single_layer_train_mode_is_deterministic():
    """With dropout gated off, train() and eval() behave identically."""
    torch.manual_seed(0)
    model = PacingLSTM(num_layers=1, dropout=0.5)
    model.train()
    x = torch.randn(4, N_SEGMENTS, len(SEQ_FEATURES))
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_rejects_wrong_feature_width():
    model = PacingLSTM(input_size=len(SEQ_FEATURES))
    with pytest.raises(RuntimeError, match="input_size"):
        model(torch.randn(2, N_SEGMENTS, len(SEQ_FEATURES) - 1))


def test_default_input_size_matches_seq_features():
    """PacingLSTM's default must match the feature vector seq_data.py emits."""
    assert PacingLSTM().lstm.input_size == len(SEQ_FEATURES)


def test_config_input_size_matches_seq_features():
    """
    PacingConfig.input_size carries a "must match len(SEQ_FEATURES)" comment.
    Same drift hazard as FinishTimePredictor.N_FEATURES vs FEATURE_COLS.
    """
    assert PacingConfig().input_size == len(SEQ_FEATURES)


def test_accepts_a_real_dataset_batch():
    """End to end: a batch straight out of make_seq_datasets() runs through."""
    cfg = PacingConfig()
    train_ds, _, _ = make_seq_datasets(n_races=20)
    X = torch.stack([train_ds[i][0] for i in range(4)])
    model = PacingLSTM(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    out = model(X)
    assert out.shape == (4, N_SEGMENTS, 1)
    assert torch.isfinite(out).all()


def test_backward_populates_gradients():
    model = PacingLSTM()
    x = torch.randn(3, N_SEGMENTS, len(SEQ_FEATURES))
    loss = model(x).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
