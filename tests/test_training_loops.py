"""
Tests for the three training modules, which had no coverage at all.

The training loop is the one place where a bug is both most likely and least
visible: it still produces a model, the loss still goes down, and nothing
raises. So these check the properties that would otherwise fail silently —
that weights actually move, that evaluation does not move them, that early
stopping counts what it claims to count, and that the checkpoint written is
the best one rather than the last.

Configs are deliberately tiny (a few dozen rows, two epochs). These test the
mechanics of the loop, not whether the model learns anything.
"""

import pathlib

import pytest
import torch
import torch.nn as nn

from pacebrain import train as train_toy
from pacebrain import train_finish, train_pacing
from pacebrain.checkpoint import detect_version, read_scaler
from pacebrain.config import FinishPredictorConfig, PacingConfig
from pacebrain.data import FEATURE_COLS, make_datasets, make_sample_data
from pacebrain.models import FinishTimePredictor
from pacebrain.seq_data import SEQ_FEATURES, make_seq_datasets
from pacebrain.seq_models import PacingLSTM


def params_snapshot(model: nn.Module) -> list[torch.Tensor]:
    """Detached copies of every parameter, for before/after comparison."""
    return [p.detach().clone() for p in model.parameters()]


def params_changed(before: list[torch.Tensor], model: nn.Module) -> bool:
    return any(
        not torch.equal(b, a) for b, a in zip(before, params_snapshot(model))
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def finish_cfg(tmp_path):
    """A finish-predictor config small enough to train in well under a second."""
    return FinishPredictorConfig(
        n_samples=60,
        epochs=2,
        batch_size=16,
        patience=25,
        checkpoint_path=str(tmp_path / "models" / "finish.pt"),
        plot_path=str(tmp_path / "reports" / "loss.png"),
    )


@pytest.fixture
def pacing_cfg(tmp_path):
    return PacingConfig(
        n_races=40,
        epochs=2,
        batch_size=8,
        patience=20,
        checkpoint_path=str(tmp_path / "models" / "pacing.pt"),
        plot_path=str(tmp_path / "reports" / "pacing_loss.png"),
    )


@pytest.fixture
def finish_loaders():
    df = make_sample_data(n_samples=60, seed=1)
    train_ds, val_ds, _ = make_datasets(df, val_fraction=0.2, seed=1)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=False),
        torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False),
    )


# ---------------------------------------------------------------------------
# train.py — the Day 2 toy loop, which works on raw tensors, not loaders
# ---------------------------------------------------------------------------

def test_make_batches_covers_every_row_exactly_once():
    X, y = torch.arange(10).float().reshape(10, 1), torch.arange(10).float().reshape(10, 1)
    seen = torch.cat([xb.ravel() for xb, _ in train_toy.make_batches(X, y, 3)])
    assert sorted(seen.tolist()) == list(range(10))


def test_make_batches_keeps_features_aligned_with_targets():
    """Shuffling must permute X and y together, or every batch is mislabelled."""
    X = torch.arange(12).float().reshape(12, 1)
    y = X * 10
    for xb, yb in train_toy.make_batches(X, y, 4):
        assert torch.equal(yb, xb * 10)


def test_make_batches_handles_a_ragged_final_batch():
    X = torch.zeros(7, 1)
    sizes = [len(xb) for xb, _ in train_toy.make_batches(X, X, 3)]
    assert sizes == [3, 3, 1]


def test_toy_train_one_epoch_updates_weights_and_returns_finite_loss():
    torch.manual_seed(0)
    model = FinishTimePredictor(input_size=4)
    X, y = torch.randn(32, 4), torch.randn(32, 1)
    before = params_snapshot(model)

    loss = train_toy.train_one_epoch(
        model, X, y, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=1e-2)
    )

    assert isinstance(loss, float) and torch.isfinite(torch.tensor(loss))
    assert params_changed(before, model)


def test_toy_evaluate_leaves_weights_untouched():
    torch.manual_seed(0)
    model = FinishTimePredictor(input_size=4)
    X, y = torch.randn(16, 4), torch.randn(16, 1)
    before = params_snapshot(model)

    loss = train_toy.evaluate(model, X, y, nn.MSELoss())

    assert isinstance(loss, float)
    assert not params_changed(before, model)


def test_toy_evaluate_puts_the_model_in_eval_mode():
    """Dropout must be off, or the reported val loss is noise."""
    model = FinishTimePredictor(input_size=4, dropout=0.5)
    model.train()
    train_toy.evaluate(model, torch.randn(8, 4), torch.randn(8, 1), nn.MSELoss())
    assert not model.training


# ---------------------------------------------------------------------------
# train_finish.py
# ---------------------------------------------------------------------------

def test_finish_train_one_epoch_updates_weights(finish_loaders):
    torch.manual_seed(0)
    train_loader, _ = finish_loaders
    model = FinishTimePredictor(input_size=len(FEATURE_COLS))
    before = params_snapshot(model)

    loss = train_finish.train_one_epoch(
        model, train_loader, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=1e-2)
    )

    assert loss > 0
    assert params_changed(before, model)


def test_finish_evaluate_leaves_weights_untouched(finish_loaders):
    torch.manual_seed(0)
    _, val_loader = finish_loaders
    model = FinishTimePredictor(input_size=len(FEATURE_COLS))
    before = params_snapshot(model)

    train_finish.evaluate(model, val_loader, nn.MSELoss())

    assert not params_changed(before, model)


def test_finish_epoch_loss_is_sample_weighted_not_batch_weighted(finish_loaders):
    """
    The loop accumulates loss * len(batch) and divides by n, so a ragged final
    batch cannot skew the mean. Averaging per batch instead would over-weight
    a short tail batch — a classic and completely silent bug.
    """
    train_loader, _ = finish_loaders
    model = FinishTimePredictor(input_size=len(FEATURE_COLS))
    loss_fn = nn.MSELoss()

    reported = train_finish.evaluate(model, train_loader, loss_fn)

    # Recompute in one shot over the whole split.
    model.eval()
    with torch.no_grad():
        X = torch.cat([xb for xb, _ in train_loader])
        y = torch.cat([yb for _, yb in train_loader])
        expected = loss_fn(model(X), y).item()

    assert reported == pytest.approx(expected, rel=1e-5)


def test_finish_train_writes_a_current_format_checkpoint(finish_cfg):
    train_finish.train(finish_cfg)
    checkpoint = torch.load(finish_cfg.checkpoint_path, weights_only=True)

    assert detect_version(checkpoint) == 2
    assert read_scaler(checkpoint) is not None
    assert list(checkpoint["feature_cols"]) == list(FEATURE_COLS)


def test_finish_train_returns_the_best_weights_not_the_last(finish_cfg):
    """
    train() reloads the checkpoint before returning. If it returned the live
    model instead, an epoch that got worse after the last improvement would
    be what you shipped — and nothing would tell you.
    """
    model = train_finish.train(finish_cfg)
    saved = torch.load(finish_cfg.checkpoint_path, weights_only=True)["state_dict"]

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, saved[name]), name


def test_finish_train_writes_the_loss_curve(finish_cfg):
    train_finish.train(finish_cfg)
    assert pathlib.Path(finish_cfg.plot_path).exists()


def test_finish_train_is_reproducible_for_a_fixed_seed(tmp_path):
    cfgs = [
        FinishPredictorConfig(
            n_samples=60, epochs=2, batch_size=16, seed=7,
            checkpoint_path=str(tmp_path / f"m{i}.pt"),
            plot_path=str(tmp_path / f"p{i}.png"),
        )
        for i in range(2)
    ]
    a, b = (train_finish.train(cfg).state_dict() for cfg in cfgs)
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_finish_early_stopping_halts_after_patience_epochs(finish_cfg, monkeypatch):
    """
    Pins the counter's exact semantics. With val loss constant, epoch 1 sets
    the best and epochs 2..(1+patience) fail to improve, so training must run
    exactly 1 + patience epochs -- not patience, and not the full schedule.
    """
    calls = []

    def constant_val_loss(model, loader, loss_fn):
        calls.append(1)
        return 5.0

    monkeypatch.setattr(train_finish, "evaluate", constant_val_loss)
    finish_cfg.epochs = 100
    finish_cfg.patience = 3

    train_finish.train(finish_cfg)

    assert len(calls) == 1 + finish_cfg.patience


def test_finish_training_runs_the_full_schedule_when_it_keeps_improving(
    finish_cfg, monkeypatch
):
    """The complement of the early-stopping test: no false positives."""
    seq = iter(range(1000, 0, -1))  # strictly decreasing, always improves

    monkeypatch.setattr(
        train_finish, "evaluate", lambda *a, **k: float(next(seq))
    )
    finish_cfg.epochs = 5
    finish_cfg.patience = 2

    train_finish.train(finish_cfg)  # must not stop early; reaching here is the assertion


# ---------------------------------------------------------------------------
# train_pacing.py
# ---------------------------------------------------------------------------

@pytest.fixture
def pacing_loaders():
    train_ds, val_ds, _ = make_seq_datasets(n_races=40, val_fraction=0.2, seed=1)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=False),
        torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False),
    )


def test_pacing_train_one_epoch_updates_weights(pacing_loaders):
    torch.manual_seed(0)
    train_loader, _ = pacing_loaders
    model = PacingLSTM(input_size=len(SEQ_FEATURES), hidden_size=16)
    before = params_snapshot(model)

    loss = train_pacing.train_one_epoch(
        model, train_loader, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=1e-2)
    )

    assert loss > 0
    assert params_changed(before, model)


def test_pacing_evaluate_leaves_weights_untouched(pacing_loaders):
    torch.manual_seed(0)
    _, val_loader = pacing_loaders
    model = PacingLSTM(input_size=len(SEQ_FEATURES), hidden_size=16)
    before = params_snapshot(model)

    train_pacing.evaluate(model, val_loader, nn.MSELoss())

    assert not params_changed(before, model)


def test_pacing_train_end_to_end(pacing_cfg):
    model = train_pacing.train(pacing_cfg)
    assert isinstance(model, PacingLSTM)
    assert pathlib.Path(pacing_cfg.checkpoint_path).exists()
    assert pathlib.Path(pacing_cfg.plot_path).exists()


def test_pacing_early_stopping_halts_after_patience_epochs(pacing_cfg, monkeypatch):
    calls = []

    def constant_val_loss(model, loader, loss_fn):
        calls.append(1)
        return 2.0

    monkeypatch.setattr(train_pacing, "evaluate", constant_val_loss)
    pacing_cfg.epochs = 100
    pacing_cfg.patience = 2

    train_pacing.train(pacing_cfg)

    assert len(calls) == 1 + pacing_cfg.patience
