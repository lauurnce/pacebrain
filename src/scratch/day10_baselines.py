"""
Day 10 — score the MLP against an honest baseline set.

Day 9 retired the "85.5% better than Riegel" claim after finding the baseline
had been handed an input it was never designed for. This script replaces that
single unfair comparison with four reference points, and reads all of them
against the noise floor rather than against zero.

Run:
    python src/scratch/day10_baselines.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from torch.utils.data import DataLoader

from pacebrain.baselines import evaluate_baselines
from pacebrain.config import FinishPredictorConfig
from pacebrain.data import make_datasets, make_sample_data
from pacebrain.eval import get_model_predictions, mae_minutes
from pacebrain.inference import load_finish_model


def split_frames(cfg: FinishPredictorConfig):
    """
    Rebuild the raw train/val DataFrames for the split make_datasets() uses.

    The baselines need unscaled columns (Riegel reads pace and distance
    directly), so they cannot go through RunningDataset. Reproducing the
    permutation with the same seed is what keeps them scored on exactly the
    rows the MLP is scored on.
    """
    df = make_sample_data(n_samples=cfg.n_samples, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * cfg.val_fraction)
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = df.iloc[idx[n_val:]].reset_index(drop=True)
    return df, train_df, val_df


def main() -> None:
    cfg = FinishPredictorConfig()
    df, train_df, val_df = split_frames(cfg)

    scores = evaluate_baselines(train_df, val_df)

    _, val_ds, _ = make_datasets(df, val_fraction=cfg.val_fraction, seed=cfg.seed)
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    y_true, y_pred = get_model_predictions(load_finish_model(cfg), loader)
    mlp_mae = mae_minutes(y_true, y_pred)

    floor = scores["noise_floor"]
    rows = [
        ("mean of train", scores["mean"]),
        ("Riegel (raw)", scores["riegel"]),
        (f"Riegel x {scores['riegel_scale']:.4f}", scores["riegel_corrected"]),
        ("linear regression", scores["linear"]),
        ("MLP", mlp_mae),
    ]

    print(f"\n=== Day 10 baselines (validation set, n={len(val_df)}) ===\n")
    print(f"  {'predictor':<24}{'MAE (min)':>11}{'x noise floor':>15}")
    print(f"  {'-' * 48}")
    for name, mae in rows:
        print(f"  {name:<24}{mae:>11.2f}{mae / floor:>15.1f}")
    print(f"  {'-' * 48}")
    print(f"  {'noise floor':<24}{floor:>11.2f}{1.0:>15.1f}")

    best_baseline = min(scores["mean"], scores["riegel_corrected"], scores["linear"])
    print(
        f"\n  MLP is {(best_baseline - mlp_mae) / best_baseline * 100:.1f}% better "
        f"than the strongest baseline (linear regression),"
    )
    print(f"  and sits {mlp_mae / floor:.1f}x above the noise floor.\n")


if __name__ == "__main__":
    main()
