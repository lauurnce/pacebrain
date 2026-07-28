"""
Tests for baselines.py — the honest baseline set from Day 10.

The point of these baselines is to be *fair*, so most of what is worth
asserting is a property rather than a number: that each baseline is fitted on
train only, that the bias correction actually helps, and that they rank in the
order the design predicts.
"""

import numpy as np
import pytest

from pacebrain.baselines import (
    NOISE_FLOOR_MAE_MIN,
    evaluate_baselines,
    fit_linear,
    fit_riegel_scale,
    mean_prediction,
    riegel_corrected_prediction,
)
from pacebrain.data import FEATURE_COLS, TARGET_COL, make_sample_data
from pacebrain.eval import mae_minutes, riegel_predict


@pytest.fixture
def frames():
    """A deterministic 80/20 split of the synthetic generator."""
    df = make_sample_data(n_samples=500, seed=7)
    n_val = 100
    return df.iloc[n_val:].reset_index(drop=True), df.iloc[:n_val].reset_index(drop=True)


# ---------------------------------------------------------------------------
# noise floor
# ---------------------------------------------------------------------------

def test_noise_floor_is_expected_absolute_normal():
    """E|N(0, 2)| = 2 * sqrt(2/pi) — the best MAE anything can reach here."""
    assert NOISE_FLOOR_MAE_MIN == pytest.approx(1.5957691, rel=1e-6)


def test_noise_floor_matches_a_simulation():
    """Guards the closed form against the generator's actual noise term."""
    draws = np.random.default_rng(0).normal(0, 2.0, 400_000)
    assert np.mean(np.abs(draws)) == pytest.approx(NOISE_FLOOR_MAE_MIN, rel=0.01)


# ---------------------------------------------------------------------------
# mean baseline
# ---------------------------------------------------------------------------

def test_mean_prediction_is_constant_and_correct_length(frames):
    train_df, val_df = frames
    pred = mean_prediction(train_df, len(val_df))
    assert len(pred) == len(val_df)
    assert len(np.unique(pred)) == 1


def test_mean_prediction_uses_train_mean_not_val_mean(frames):
    """A baseline may not peek at the split it is scored on."""
    train_df, val_df = frames
    pred = mean_prediction(train_df, len(val_df))
    assert pred[0] == pytest.approx(train_df[TARGET_COL].mean(), rel=1e-6)
    # The two means differ, so this assertion can actually fail.
    assert pred[0] != pytest.approx(val_df[TARGET_COL].mean(), rel=1e-9)


# ---------------------------------------------------------------------------
# bias-corrected Riegel
# ---------------------------------------------------------------------------

def test_riegel_scale_corrects_downward(frames):
    """
    Day 9 found Riegel overpredicts — it misses three sub-1.0 fitness factors
    whose product averages 0.8345, so the fitted scale should land near there.
    """
    train_df, _ = frames
    assert fit_riegel_scale(train_df) == pytest.approx(0.83, abs=0.05)


def test_bias_correction_beats_raw_riegel(frames):
    """If the error is multiplicative, one constant must remove most of it."""
    train_df, val_df = frames
    scale = fit_riegel_scale(train_df)
    y_val = val_df[TARGET_COL].values

    raw = mae_minutes(y_val, riegel_predict(
        val_df["avg_pace_min_per_km"].values, val_df["race_distance_km"].values,
    ))
    corrected = mae_minutes(y_val, riegel_corrected_prediction(val_df, scale))
    assert corrected < raw


def test_riegel_scale_is_fitted_on_train_only(frames):
    """Refitting on val would be leakage; it must change the number."""
    train_df, val_df = frames
    assert fit_riegel_scale(train_df) != pytest.approx(
        fit_riegel_scale(val_df), abs=1e-9
    )


# ---------------------------------------------------------------------------
# linear regression
# ---------------------------------------------------------------------------

def test_linear_uses_all_six_features(frames):
    train_df, _ = frames
    assert fit_linear(train_df).coef_.shape == (len(FEATURE_COLS),)


def test_linear_beats_the_mean_baseline(frames):
    """The features carry signal a constant cannot, so this must hold."""
    train_df, val_df = frames
    y_val = val_df[TARGET_COL].values
    linear_mae = mae_minutes(y_val, fit_linear(train_df).predict(val_df[FEATURE_COLS].values))
    assert linear_mae < mae_minutes(y_val, mean_prediction(train_df, len(val_df)))


# ---------------------------------------------------------------------------
# evaluate_baselines
# ---------------------------------------------------------------------------

def test_evaluate_baselines_reports_every_baseline(frames):
    train_df, val_df = frames
    scores = evaluate_baselines(train_df, val_df)
    assert set(scores) == {
        "mean", "riegel", "riegel_corrected", "linear", "noise_floor", "riegel_scale",
    }


def test_baselines_rank_in_the_predicted_order(frames):
    """
    Each baseline sees strictly more than the one above it, so each should
    score better: mean (no features) > raw Riegel (two, mis-specified) >
    corrected Riegel (two, calibrated) > linear (all six) > noise floor.
    """
    train_df, val_df = frames
    s = evaluate_baselines(train_df, val_df)
    assert s["mean"] > s["riegel"] > s["riegel_corrected"] > s["linear"] > s["noise_floor"]


def test_no_baseline_beats_the_noise_floor(frames):
    """Beating it would mean the split or the metric is wrong, not that the baseline is good."""
    train_df, val_df = frames
    s = evaluate_baselines(train_df, val_df)
    for name in ("mean", "riegel", "riegel_corrected", "linear"):
        assert s[name] > s["noise_floor"], f"{name} beat the noise floor"
