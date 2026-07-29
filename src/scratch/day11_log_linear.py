"""
Day 11 — does one feature transform beat 189 epochs of gradient descent?

Day 10 ended on a hypothesis: make_sample_data() builds its target as a
product of five factors, so taking logs should turn it into a sum, and a
linear fit on log-transformed features should close most of the gap to the
MLP. This script tests that, and prints the fitted coefficients next to the
generator's own constants so the claim is checkable rather than asserted.

Run:
    python src/scratch/day11_log_linear.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from pacebrain.baselines import LOG_FEATURE_NAMES, evaluate_baselines, fit_log_linear
from pacebrain.config import FinishPredictorConfig
from pacebrain.data import make_datasets
from pacebrain.eval import get_model_predictions, mae_minutes
from pacebrain.inference import load_finish_model
from scratch.day10_baselines import split_frames

# The generator's closed-form constants, in LOG_FEATURE_NAMES order. None
# marks a term whose truth is not a single clean number: the pace exponent is
# 1.0 but is absorbed partly into the intercept, and runs_per_week never
# enters the generator at all.
GENERATOR_TRUTH = {
    "log_avg_pace_min_per_km": 1.0,
    "log_race_distance_km": 1.06,
    "weekly_mileage_km": -0.008,
    "long_run_km": -0.015,
    "days_since_long_run": 0.005,
    "runs_per_week": 0.0,
}


def main() -> None:
    cfg = FinishPredictorConfig()
    df, train_df, val_df = split_frames(cfg)

    scores = evaluate_baselines(train_df, val_df)

    _, val_ds, _ = make_datasets(df, val_fraction=cfg.val_fraction, seed=cfg.seed)
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    y_true, y_pred = get_model_predictions(load_finish_model(cfg), loader)
    mlp_mae = mae_minutes(y_true, y_pred)

    floor = scores["noise_floor"]

    print(f"\n=== Day 11 log-space baseline (validation set, n={len(val_df)}) ===\n")
    print(f"  {'predictor':<24}{'MAE (min)':>11}{'x noise floor':>15}")
    print(f"  {'-' * 48}")
    for name, mae in [
        ("linear regression", scores["linear"]),
        ("MLP", mlp_mae),
        ("log-linear", scores["log_linear"]),
    ]:
        print(f"  {name:<24}{mae:>11.2f}{mae / floor:>15.2f}")
    print(f"  {'-' * 48}")
    print(f"  {'noise floor':<24}{floor:>11.2f}{1.0:>15.2f}")

    # The MAE alone could be luck. The coefficients are the actual evidence:
    # if the fit recovered the generator rather than merely approximating it,
    # they have to land on constants chosen before the fit was ever run.
    model = fit_log_linear(train_df)
    print("\n  fitted coefficients vs the generator's own constants:\n")
    print(f"  {'term':<26}{'fitted':>12}{'generator':>12}")
    print(f"  {'-' * 50}")
    for name, coef in zip(LOG_FEATURE_NAMES, model.coef_):
        print(f"  {name:<26}{coef:>+12.6f}{GENERATOR_TRUTH[name]:>+12.4f}")
    print(f"  {'intercept':<26}{model.intercept_:>+12.6f}{'-':>12}")

    delta = (mlp_mae - scores["log_linear"]) / mlp_mae * 100
    print(
        f"\n  The log-linear baseline beats the MLP by {delta:.1f}% and sits "
        f"{scores['log_linear'] / floor:.2f}x above the noise floor,"
    )
    print(f"  against the MLP's {mlp_mae / floor:.2f}x.\n")


if __name__ == "__main__":
    main()
