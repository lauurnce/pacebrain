"""
Tests for the optional ReduceLROnPlateau schedule.

It is off by default, so the first thing worth pinning is that it really is
off — a schedule that quietly switched itself on would change every training
run in the repo without anything in the diff saying so.
"""

import pytest
import torch

from pacebrain import train_finish, train_pacing
from pacebrain.config import FinishPredictorConfig, PacingConfig


@pytest.fixture
def cfg(tmp_path):
    return FinishPredictorConfig(
        n_samples=60,
        epochs=6,
        batch_size=16,
        checkpoint_path=str(tmp_path / "m.pt"),
        plot_path=str(tmp_path / "p.png"),
    )


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

def test_schedule_is_off_by_default():
    """Measured and rejected — see reports/lr_schedule_experiment.md."""
    assert FinishPredictorConfig().lr_schedule is False
    assert PacingConfig().lr_schedule is False


def test_lr_patience_is_below_early_stopping_patience():
    """
    The two counters race. If lr_patience is not clearly the faster, early
    stopping halts training before the schedule ever acts — measured at
    lr_patience 15 and 20, which returned the no-schedule number exactly.
    """
    for c in (FinishPredictorConfig(), PacingConfig()):
        assert c.lr_patience < c.patience


# ---------------------------------------------------------------------------
# off
# ---------------------------------------------------------------------------

def test_lr_is_untouched_when_the_schedule_is_off(cfg, monkeypatch):
    """Constant val loss would trigger any plateau schedule that was active."""
    monkeypatch.setattr(train_finish, "evaluate", lambda *a, **k: 5.0)
    cfg.lr_schedule = False
    cfg.patience = 100

    seen = _lrs_during_training(cfg, monkeypatch)

    assert len(set(seen)) == 1
    assert seen[0] == pytest.approx(cfg.lr)


# ---------------------------------------------------------------------------
# on
# ---------------------------------------------------------------------------

def test_lr_drops_when_enabled_and_val_loss_plateaus(cfg, monkeypatch):
    monkeypatch.setattr(train_finish, "evaluate", lambda *a, **k: 5.0)
    cfg.lr_schedule = True
    cfg.lr_patience = 1
    cfg.lr_factor = 0.5
    cfg.epochs = 8
    cfg.patience = 100

    seen = _lrs_during_training(cfg, monkeypatch)

    assert min(seen) < cfg.lr


def test_lr_holds_while_val_loss_keeps_improving(cfg, monkeypatch):
    """
    A plateau scheduler must react to stalling, not to elapsed time. Feeding
    a strictly decreasing loss is what separates the two.
    """
    seq = iter(range(1000, 0, -1))
    monkeypatch.setattr(train_finish, "evaluate", lambda *a, **k: float(next(seq)))
    cfg.lr_schedule = True
    cfg.lr_patience = 1
    cfg.epochs = 8

    seen = _lrs_during_training(cfg, monkeypatch)

    assert len(set(seen)) == 1


def test_factor_controls_the_size_of_the_drop(cfg, monkeypatch):
    monkeypatch.setattr(train_finish, "evaluate", lambda *a, **k: 5.0)
    cfg.lr_schedule = True
    cfg.lr_patience = 1
    cfg.lr_factor = 0.25
    cfg.epochs = 4
    cfg.patience = 100

    seen = _lrs_during_training(cfg, monkeypatch)

    assert min(seen) == pytest.approx(cfg.lr * 0.25, rel=1e-6)


# ---------------------------------------------------------------------------
# pacing model wiring
# ---------------------------------------------------------------------------

def test_pacing_training_runs_with_the_schedule_enabled(tmp_path):
    pacing_cfg = PacingConfig(
        n_races=40,
        epochs=3,
        batch_size=8,
        lr_schedule=True,
        lr_patience=1,
        checkpoint_path=str(tmp_path / "pm.pt"),
        plot_path=str(tmp_path / "pp.png"),
    )
    assert train_pacing.train(pacing_cfg) is not None


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _lrs_during_training(cfg, monkeypatch):
    """
    Record the LR at every optimiser step by wrapping Adam.step.

    Reading it afterwards would miss the trajectory entirely — the question
    is whether the LR changed *during* the run, not where it ended up.
    """
    seen = []
    real_step = torch.optim.Adam.step

    def recording_step(self, *args, **kwargs):
        seen.append(self.param_groups[0]["lr"])
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)
    train_finish.train(cfg)
    return seen
