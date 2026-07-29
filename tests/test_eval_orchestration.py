"""
Tests for eval.py's plotting and orchestration — the two functions the rest of
the eval suite does not reach.

`mae_minutes`, `riegel_predict` and `get_model_predictions` are pure enough to
test directly, and tests/test_eval.py does. `plot_comparison` and
`run_evaluation` are not: one writes a PNG, the other loads a checkpoint off
disk, runs a model, prints a table and writes that PNG. Left uncovered they
were the largest gap in the package (eval.py sat at 39%), and they are exactly
where a refactor breaks something silently — nothing downstream reads the plot,
so a broken one is invisible until someone opens it.

The approach throughout is to run the real thing in a tmp_path rather than mock
it. `run_evaluation` hardcodes a relative output path, so chdir is what makes
that safe.
"""

from __future__ import annotations

import dataclasses
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from pacebrain.checkpoint import build_checkpoint
from pacebrain.config import FinishPredictorConfig
from pacebrain.data import FEATURE_COLS, make_datasets, make_sample_data
from pacebrain.eval import plot_comparison, run_evaluation
from pacebrain.models import FinishTimePredictor


@pytest.fixture
def small_cfg(tmp_path: pathlib.Path) -> FinishPredictorConfig:
    """
    A config small enough to evaluate quickly, pointed at a tmp checkpoint.

    n_samples is cut from 1000 to 200 because none of these tests assert on
    accuracy — they assert that the orchestration runs, writes what it claims
    to write, and fails loudly when it cannot.
    """
    return dataclasses.replace(
        FinishPredictorConfig(),
        n_samples=200,
        checkpoint_path=str(tmp_path / "finish.pt"),
    )


@pytest.fixture
def trained_checkpoint(small_cfg: FinishPredictorConfig) -> FinishPredictorConfig:
    """
    Write a real (untrained) checkpoint in the current v2 format.

    The weights are random rather than trained: run_evaluation's job is to
    load, predict, score and plot, and it has to do all of that identically
    whether the model is good or not. Training here would cost seconds per
    test and assert nothing extra.
    """
    df = make_sample_data(n_samples=small_cfg.n_samples, seed=small_cfg.seed)
    _, _, scaler = make_datasets(
        df, val_fraction=small_cfg.val_fraction, seed=small_cfg.seed
    )
    model = FinishTimePredictor(
        input_size=small_cfg.input_size,
        hidden_sizes=small_cfg.hidden_sizes,
        dropout=small_cfg.dropout,
    )
    torch.save(
        build_checkpoint(model.state_dict(), scaler, FEATURE_COLS),
        small_cfg.checkpoint_path,
    )
    return small_cfg


# ---------------------------------------------------------------------------
# plot_comparison
# ---------------------------------------------------------------------------

@pytest.fixture
def prediction_arrays():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(20, 300, 40)
    return y_true, y_true + rng.normal(0, 5, 40)


def test_plot_comparison_writes_a_png(tmp_path, prediction_arrays):
    y_true, y_pred = prediction_arrays
    out = tmp_path / "scatter.png"

    plot_comparison(y_true, y_pred, y_true, y_pred * 1.2, save_path=str(out))

    assert out.exists()
    # A PNG signature rather than just a non-empty file: matplotlib writing a
    # zero-byte or truncated file would still pass an exists() check.
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_comparison_creates_the_parent_directory(tmp_path, prediction_arrays):
    """reports/ is gitignored, so on a fresh clone the directory is absent."""
    y_true, y_pred = prediction_arrays
    out = tmp_path / "reports" / "scatter.png"
    assert not out.parent.exists()

    plot_comparison(y_true, y_pred, y_true, y_pred, save_path=str(out))

    assert out.exists()


def test_plot_comparison_closes_its_figure(tmp_path, prediction_arrays):
    """
    Leaked figures are a real cost in a loop — matplotlib warns after 20 open
    and holds every one in memory until then. The close() is easy to drop in a
    refactor and nothing else would notice.
    """
    y_true, y_pred = prediction_arrays
    before = len(plt.get_fignums())

    for i in range(3):
        plot_comparison(
            y_true, y_pred, y_true, y_pred, save_path=str(tmp_path / f"s{i}.png")
        )

    assert len(plt.get_fignums()) == before


def test_plot_comparison_handles_identical_predictions(tmp_path):
    """
    Perfect predictions collapse the axis range to a single point, and the
    function derives its limits from min/max. Degenerate input should still
    produce a plot rather than an axis error.
    """
    y = np.full(10, 100.0)
    out = tmp_path / "flat.png"

    plot_comparison(y, y, y, y, save_path=str(out))

    assert out.exists()


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

def test_run_evaluation_end_to_end(trained_checkpoint, tmp_path, monkeypatch, capsys):
    """
    The whole orchestrator against a real checkpoint.

    chdir into tmp_path because run_evaluation writes to the relative path
    reports/day5_scatter.png — without it the test would litter the repo's
    own reports/ directory.
    """
    monkeypatch.chdir(tmp_path)

    run_evaluation(trained_checkpoint)

    assert (tmp_path / "reports" / "day5_scatter.png").exists()

    out = capsys.readouterr().out
    assert "Evaluation (validation set)" in out
    assert "MLP" in out and "Riegel" in out
    # 20% of 200 rows.
    assert "Samples : 40" in out


def test_run_evaluation_reports_both_mae_and_rmse(
    trained_checkpoint, tmp_path, monkeypatch, capsys
):
    """
    The RMSE/MAE ratio is the point of printing both — it is the shape of the
    error distribution in one number, and a table with only MAE hides it.
    """
    monkeypatch.chdir(tmp_path)

    run_evaluation(trained_checkpoint)

    out = capsys.readouterr().out
    assert "MAE" in out and "RMSE" in out and "ratio" in out
    assert "gaussian errors" in out


def test_run_evaluation_states_which_model_won(
    trained_checkpoint, tmp_path, monkeypatch, capsys
):
    """
    The verdict line has two branches and an untrained model is the case that
    exercises the losing one, which a trained checkpoint never would.
    """
    monkeypatch.chdir(tmp_path)

    run_evaluation(trained_checkpoint)

    out = capsys.readouterr().out
    assert ("MLP beats Riegel by" in out) or ("Riegel beats MLP by" in out)


def test_run_evaluation_without_a_checkpoint_fails_loudly(
    small_cfg, tmp_path, monkeypatch
):
    """
    A missing checkpoint must name the file and say how to make one. Silently
    evaluating random weights would produce a plausible-looking table of
    meaningless numbers.
    """
    monkeypatch.chdir(tmp_path)
    assert not pathlib.Path(small_cfg.checkpoint_path).exists()

    with pytest.raises(FileNotFoundError, match="train_finish.py"):
        run_evaluation(small_cfg)


def test_run_evaluation_scores_baseline_on_the_same_rows_as_the_model(
    trained_checkpoint, tmp_path, monkeypatch, capsys
):
    """
    run_evaluation rebuilds the val DataFrame by replaying make_datasets()'s
    permutation with the same seed, so Riegel is scored on the rows the model
    is scored on. If those drift apart the comparison is meaningless, and the
    row count is the cheapest observable that catches it.
    """
    monkeypatch.chdir(tmp_path)

    run_evaluation(trained_checkpoint)

    expected = int(trained_checkpoint.n_samples * trained_checkpoint.val_fraction)
    assert f"Samples : {expected}" in capsys.readouterr().out
