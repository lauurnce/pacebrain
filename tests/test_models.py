"""
Tests for models.py — MLP layer construction, forward shapes, dropout
behaviour, and the FinishTimePredictor / data.py feature-count invariant.
"""


import pytest
import torch
import torch.nn as nn

from pacebrain.config import FinishPredictorConfig
from pacebrain.data import FEATURE_COLS
from pacebrain.models import MLP, FinishTimePredictor


def linear_layers(model: MLP) -> list:
    return [m for m in model.net if isinstance(m, nn.Linear)]


def module_count(model: MLP, cls) -> int:
    return sum(1 for m in model.net if isinstance(m, cls))


def test_forward_batch_shape():
    model = MLP(input_size=6, hidden_sizes=[16, 8])
    out = model(torch.randn(32, 6))
    assert out.shape == (32, 1), f"expected (32, 1), got {tuple(out.shape)}"


def test_forward_single_row_shape():
    """A batch of one still comes back with the batch dim intact."""
    model = MLP(input_size=6, hidden_sizes=[16, 8])
    out = model(torch.randn(1, 6))
    assert out.shape == (1, 1)


def test_forward_unbatched_row():
    """nn.Linear also accepts a bare (input_size,) vector -> (output_size,)."""
    model = MLP(input_size=6, hidden_sizes=[16, 8])
    out = model(torch.randn(6))
    assert out.shape == (1,)


def test_output_size_is_configurable():
    model = MLP(input_size=6, hidden_sizes=[16], output_size=3)
    assert model(torch.randn(4, 6)).shape == (4, 3)


@pytest.mark.parametrize("hidden_sizes", [[], [8], [16, 8], [32, 16, 8]])
def test_hidden_sizes_produce_expected_linear_count(hidden_sizes):
    """One Linear per hidden layer, plus the output Linear."""
    model = MLP(input_size=6, hidden_sizes=hidden_sizes)
    linears = linear_layers(model)
    assert len(linears) == len(hidden_sizes) + 1
    assert module_count(model, nn.ReLU) == len(hidden_sizes)


def test_layer_widths_chain_through():
    """Each Linear's out_features feeds the next Linear's in_features."""
    model = MLP(input_size=6, hidden_sizes=[16, 8], output_size=1)
    widths = [(lin.in_features, lin.out_features) for lin in linear_layers(model)]
    assert widths == [(6, 16), (16, 8), (8, 1)]


def test_empty_hidden_sizes_is_plain_linear():
    """No hidden layers degenerates to linear regression — one Linear, no ReLU."""
    model = MLP(input_size=6, hidden_sizes=[])
    assert len(model.net) == 1
    assert isinstance(model.net[0], nn.Linear)
    assert model(torch.randn(5, 6)).shape == (5, 1)


def test_no_dropout_layers_when_dropout_zero():
    model = MLP(input_size=6, hidden_sizes=[16, 8], dropout=0.0)
    assert module_count(model, nn.Dropout) == 0


def test_dropout_layer_per_hidden_layer_when_enabled():
    model = MLP(input_size=6, hidden_sizes=[16, 8], dropout=0.25)
    drops = [m for m in model.net if isinstance(m, nn.Dropout)]
    assert len(drops) == 2
    assert all(d.p == 0.25 for d in drops)


def test_finish_time_predictor_defaults():
    """Defaults are hidden_sizes [64, 32] with dropout 0.1."""
    model = FinishTimePredictor()
    widths = [(lin.in_features, lin.out_features) for lin in linear_layers(model)]
    assert widths == [(6, 64), (64, 32), (32, 1)]
    drops = [m for m in model.net if isinstance(m, nn.Dropout)]
    assert len(drops) == 2
    assert all(d.p == pytest.approx(0.1) for d in drops)


def test_finish_time_predictor_accepts_custom_hidden_sizes():
    model = FinishTimePredictor(hidden_sizes=[8], dropout=0.0)
    widths = [(lin.in_features, lin.out_features) for lin in linear_layers(model)]
    assert widths == [(6, 8), (8, 1)]
    assert module_count(model, nn.Dropout) == 0


def test_finish_time_predictor_accepts_six_features():
    model = FinishTimePredictor()
    out = model(torch.randn(12, FinishTimePredictor.N_FEATURES))
    assert out.shape == (12, 1)


@pytest.mark.parametrize("wrong_width", [5, 7])
def test_finish_time_predictor_rejects_wrong_input_width(wrong_width):
    model = FinishTimePredictor()
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        model(torch.randn(4, wrong_width))


def test_n_features_matches_feature_cols():
    """
    The invariant models.py's own comment demands: N_FEATURES must stay in
    sync with len(FEATURE_COLS) in data.py. If a feature is added there and
    not here, every checkpoint silently stops matching its data pipeline.
    """
    assert FinishTimePredictor.N_FEATURES == len(FEATURE_COLS), (
        f"N_FEATURES={FinishTimePredictor.N_FEATURES} but "
        f"len(FEATURE_COLS)={len(FEATURE_COLS)} — models.py and data.py drifted"
    )


def test_config_input_size_matches_feature_cols():
    """FinishPredictorConfig.input_size carries the same must-match comment."""
    assert FinishPredictorConfig().input_size == len(FEATURE_COLS)


def test_config_defaults_match_predictor_defaults():
    """The config's architecture defaults are what FinishTimePredictor builds."""
    cfg = FinishPredictorConfig()
    from_cfg = FinishTimePredictor(hidden_sizes=cfg.hidden_sizes, dropout=cfg.dropout)
    default = FinishTimePredictor()
    assert str(from_cfg.net) == str(default.net)


def test_eval_mode_is_deterministic():
    """Dropout is a no-op in eval(), so repeated forwards are identical."""
    torch.manual_seed(0)
    model = FinishTimePredictor(dropout=0.5)
    model.eval()
    x = torch.randn(8, 6)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_train_mode_dropout_perturbs_output():
    """In train() the same input goes through different dropout masks."""
    torch.manual_seed(0)
    model = FinishTimePredictor(dropout=0.5)
    model.train()
    x = torch.randn(8, 6)
    with torch.no_grad():
        assert not torch.equal(model(x), model(x))


def test_dropout_zero_is_deterministic_even_in_train_mode():
    """With dropout off there is no stochasticity to begin with."""
    torch.manual_seed(0)
    model = FinishTimePredictor(dropout=0.0)
    model.train()
    x = torch.randn(8, 6)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_backward_populates_gradients():
    """Sanity check that the whole stack is differentiable end to end."""
    model = FinishTimePredictor()
    x = torch.randn(4, 6)
    loss = model(x).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
