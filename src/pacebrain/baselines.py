"""
An honest baseline set for the finish-time predictor — Day 10.

Day 9 showed the Day 5 comparison was unfair. `riegel_predict()` is
algebraically the first two factors of the data generator, then handed an
input it was never designed for (easy training pace, not a previous race
time). Its 27.56 min MAE measured that handicap, so "85.5% better than
baseline" said more about the baseline than about the MLP.

A baseline is only informative if it had a fair shot. This module provides
three that do, in increasing order of what they are allowed to see:

    mean        — predicts the training mean, ignoring every feature.
                  Anything that cannot beat this has learned nothing.
    riegel*     — Riegel, rescaled by a single constant fitted on train.
                  Isolates "is the error bias, or shape?"
    linear      — least squares on the same six features the MLP gets.
                  The one that matters: if an MLP cannot clearly beat a
                  linear model on data this well behaved, its extra
                  capacity is not earning its place.

All four numbers are read against the noise floor, not against zero. The
target carries N(0, 2) noise, so no predictor can do better than
E|N(0, 2)| = 1.596 min however good it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pacebrain.data import FEATURE_COLS, TARGET_COL
from pacebrain.eval import mae_minutes, riegel_predict

# make_sample_data() adds rng.normal(0, 2) to every target. For X ~ N(0, s),
# E|X| = s * sqrt(2/pi), so this is the best MAE any predictor can reach here.
NOISE_SIGMA_MIN = 2.0
NOISE_FLOOR_MAE_MIN = float(NOISE_SIGMA_MIN * np.sqrt(2.0 / np.pi))


def mean_prediction(train_df: pd.DataFrame, n: int) -> np.ndarray:
    """
    Predict the training mean for all n rows.

    The floor for "did the model use its features at all". Uses the *training*
    mean rather than the validation mean on purpose: a baseline may not look
    at the split it is scored on.
    """
    return np.full(n, float(train_df[TARGET_COL].mean()), dtype=np.float64)


def fit_riegel_scale(train_df: pd.DataFrame) -> float:
    """
    Fit the single multiplicative constant that best rescales Riegel.

    Day 9 established the error is multiplicative, not additive: Riegel is
    missing the volume, long-run and freshness factors, whose product averages
    0.8345, so it overpredicts by roughly 20% on every row. One constant should
    therefore absorb most of the gap.

    The median ratio is used rather than the mean because the score is MAE:
    the median minimises absolute deviation, and it is not dragged by the
    handful of rows where the noise term dominates a short race.
    """
    pred = riegel_predict(
        train_df["avg_pace_min_per_km"].values,
        train_df["race_distance_km"].values,
    )
    return float(np.median(train_df[TARGET_COL].values / pred))


def riegel_corrected_prediction(df: pd.DataFrame, scale: float) -> np.ndarray:
    """Apply Riegel, then the correction factor fitted by fit_riegel_scale()."""
    pred = riegel_predict(
        df["avg_pace_min_per_km"].values,
        df["race_distance_km"].values,
    )
    return pred * scale


def fit_linear(train_df: pd.DataFrame) -> LinearRegression:
    """
    Least squares on FEATURE_COLS — the MLP's own inputs, no scaling needed.

    Unscaled features are fine here where they are not for the network:
    a closed-form least-squares fit is invariant to per-feature rescaling,
    so a StandardScaler would only add a step that could drift out of sync
    with the one in the checkpoint.
    """
    model = LinearRegression()
    model.fit(train_df[FEATURE_COLS].values, train_df[TARGET_COL].values)
    return model


def evaluate_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Fit every baseline on train_df and score it on val_df.

    Returns MAE in minutes keyed by baseline name, plus the noise floor, so a
    caller can render the whole comparison without refitting anything.
    """
    y_val = val_df[TARGET_COL].values
    scale = fit_riegel_scale(train_df)
    linear = fit_linear(train_df)

    return {
        "mean": mae_minutes(y_val, mean_prediction(train_df, len(val_df))),
        "riegel": mae_minutes(
            y_val,
            riegel_predict(
                val_df["avg_pace_min_per_km"].values,
                val_df["race_distance_km"].values,
            ),
        ),
        "riegel_corrected": mae_minutes(
            y_val, riegel_corrected_prediction(val_df, scale)
        ),
        "linear": mae_minutes(y_val, linear.predict(val_df[FEATURE_COLS].values)),
        "noise_floor": NOISE_FLOOR_MAE_MIN,
        "riegel_scale": scale,
    }
