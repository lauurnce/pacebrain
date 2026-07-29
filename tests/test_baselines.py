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
    LOG_FEATURE_NAMES,
    NOISE_FLOOR_MAE_MIN,
    evaluate_baselines,
    fit_linear,
    fit_log_linear,
    fit_riegel_scale,
    log_features,
    log_linear_prediction,
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
# log-space linear regression (Day 11)
# ---------------------------------------------------------------------------

def test_log_features_has_one_column_per_feature(frames):
    train_df, _ = frames
    assert log_features(train_df).shape == (len(train_df), len(FEATURE_COLS))


def test_log_linear_recovers_the_generator_exponents(frames):
    """
    The sharpest statement this baseline can make.

    make_sample_data() is a product of five factors, so in log space its
    coefficients are known in closed form rather than merely estimated. If the
    fit is doing what the module claims, it must land on them — this asserts
    against the generator's own constants, not against a recorded run.
    """
    train_df, _ = frames
    coef = dict(zip(LOG_FEATURE_NAMES, fit_log_linear(train_df).coef_))

    assert coef["log_race_distance_km"] == pytest.approx(1.06, abs=0.02)
    assert coef["weekly_mileage_km"] == pytest.approx(-0.008, abs=0.001)
    assert coef["long_run_km"] == pytest.approx(-0.015, abs=0.001)
    # log1p(0.005*d) ~= 0.005*d over the generator's 3-21 day range.
    assert coef["days_since_long_run"] == pytest.approx(0.005, abs=0.001)


def test_log_linear_finds_runs_per_week_irrelevant(frames):
    """
    runs_per_week never enters the generator. A fit that recovered the true
    structure should drive it to roughly zero, and it is the one coefficient
    where a large value would mean the model is fitting noise.
    """
    train_df, _ = frames
    coef = dict(zip(LOG_FEATURE_NAMES, fit_log_linear(train_df).coef_))
    assert abs(coef["runs_per_week"]) < 0.01


def test_log_linear_predictions_are_positive(frames):
    """exp() of anything is positive; a negative finish time would mean the
    inverse transform was dropped somewhere."""
    train_df, val_df = frames
    assert (log_linear_prediction(val_df, fit_log_linear(train_df)) > 0).all()


def test_log_linear_beats_plain_linear(frames):
    """
    Day 11's hypothesis: the generator is multiplicative, so the same
    estimator given logged features should close most of the remaining gap.
    """
    train_df, val_df = frames
    y_val = val_df[TARGET_COL].values
    log_mae = mae_minutes(y_val, log_linear_prediction(val_df, fit_log_linear(train_df)))
    plain_mae = mae_minutes(y_val, fit_linear(train_df).predict(val_df[FEATURE_COLS].values))
    assert log_mae < plain_mae


def test_log_linear_fitted_on_train_only(frames):
    """Refitting on val must move the coefficients, or the split is not real."""
    train_df, val_df = frames
    assert not np.allclose(fit_log_linear(train_df).coef_, fit_log_linear(val_df).coef_)


# ---------------------------------------------------------------------------
# evaluate_baselines
# ---------------------------------------------------------------------------

def test_evaluate_baselines_reports_every_baseline(frames):
    train_df, val_df = frames
    scores = evaluate_baselines(train_df, val_df)
    assert set(scores) == {
        "mean", "riegel", "riegel_corrected", "linear", "log_linear",
        "noise_floor", "riegel_scale",
    }


def test_baselines_rank_in_the_predicted_order(frames):
    """
    Each baseline sees strictly more than the one above it, so each should
    score better: mean (no features) > raw Riegel (two, mis-specified) >
    corrected Riegel (two, calibrated) > linear (all six) > log-linear (all
    six, transformed to match the generator) > noise floor.
    """
    train_df, val_df = frames
    s = evaluate_baselines(train_df, val_df)
    assert (
        s["mean"] > s["riegel"] > s["riegel_corrected"]
        > s["linear"] > s["log_linear"] > s["noise_floor"]
    )


def test_no_baseline_beats_the_noise_floor(frames):
    """Beating it would mean the split or the metric is wrong, not that the baseline is good."""
    train_df, val_df = frames
    s = evaluate_baselines(train_df, val_df)
    for name in ("mean", "riegel", "riegel_corrected", "linear", "log_linear"):
        assert s[name] > s["noise_floor"], f"{name} beat the noise floor"
