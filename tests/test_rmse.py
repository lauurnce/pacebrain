"""
Tests for rmse_minutes — and for the relationship between RMSE and MAE that
makes reporting both worthwhile.
"""

import numpy as np
import pytest

from pacebrain.eval import mae_minutes, rmse_minutes


def test_known_value():
    """Errors of 3, 0 and 4 -> sqrt((9 + 0 + 16) / 3) = sqrt(25/3)."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([103.0, 200.0, 296.0])
    assert rmse_minutes(y_true, y_pred) == pytest.approx(np.sqrt(25 / 3))


def test_perfect_prediction_is_zero():
    y = np.array([12.5, 240.0, 61.25])
    assert rmse_minutes(y, y) == 0.0


def test_symmetric_in_sign():
    y_true = np.array([100.0, 100.0])
    over, under = np.array([107.0, 107.0]), np.array([93.0, 93.0])
    assert rmse_minutes(y_true, over) == pytest.approx(rmse_minutes(y_true, under))


def test_returns_plain_float():
    """Callers format this with f-strings, so it must not stay a numpy scalar."""
    result = rmse_minutes(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert type(result) is float


def test_rmse_is_never_below_mae():
    """
    A mathematical guarantee (Jensen's inequality), so it holds for any input.
    Worth asserting because a swapped implementation would break it.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true = rng.normal(180, 40, 50)
        y_pred = y_true + rng.normal(0, 8, 50)
        assert rmse_minutes(y_true, y_pred) >= mae_minutes(y_true, y_pred) - 1e-9


def test_equal_to_mae_when_every_error_is_identical():
    """With no spread in |error|, squaring changes nothing — the ratio is exactly 1."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([15.0, 25.0, 35.0])
    assert rmse_minutes(y_true, y_pred) == pytest.approx(mae_minutes(y_true, y_pred))


def test_one_large_miss_moves_rmse_far_more_than_mae():
    """
    The reason both are reported. Thirty 1-minute misses and one 30-minute
    blow-up have almost the same MAE, but RMSE separates them sharply --
    which is the case that matters when someone is pacing a marathon.
    """
    uniform = np.full(31, 1.0)
    uniform[0] = 1.0
    spiky = np.zeros(31)
    spiky[0] = 30.0

    zeros = np.zeros(31)
    assert mae_minutes(zeros, uniform) == pytest.approx(1.0)
    assert mae_minutes(zeros, spiky) == pytest.approx(30 / 31, rel=1e-6)
    # Near-identical MAE, but RMSE is over 5x apart.
    assert rmse_minutes(zeros, spiky) > 5 * rmse_minutes(zeros, uniform)


def test_ratio_approaches_gaussian_constant_for_normal_errors():
    """
    For errors ~ N(0, s): E|e| = s*sqrt(2/pi) and sqrt(E[e^2]) = s, so the
    ratio tends to sqrt(pi/2) = 1.2533. This is the reference the eval
    output prints the ratio against.
    """
    errors = np.random.default_rng(3).normal(0, 5, 200_000)
    ratio = rmse_minutes(np.zeros_like(errors), errors) / mae_minutes(
        np.zeros_like(errors), errors
    )
    assert ratio == pytest.approx(np.sqrt(np.pi / 2), rel=0.01)
