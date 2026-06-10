"""
Data utilities — synthetic generator (Day 2) and real RunningDataset (Day 3+).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Features used by RunningDataset.  Target is finish_time_min.
FEATURE_COLS = [
    "weekly_mileage_km",   # avg km per week in training block
    "avg_pace_min_per_km", # easy-run pace (lower = faster runner)
    "long_run_km",         # longest run in the block
    "days_since_long_run", # recency of the long run
    "runs_per_week",       # training frequency
    "race_distance_km",    # 5 / 10 / 21.1 / 42.2
]
TARGET_COL = "finish_time_min"


def make_sample_data(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic running data that mimics a Strava/Garmin export.

    Each row represents one runner's training block plus a race result.
    The target (finish_time_min) is derived from a noisy physics-inspired
    formula so the MLP has a real, learnable signal.

    To swap in your own data later:
        df = pd.read_csv("data/activities.csv")
        # rename columns to match FEATURE_COLS + TARGET_COL, then proceed.
    """
    rng = np.random.default_rng(seed)

    weekly_mileage = rng.uniform(20, 120, n_samples).astype(np.float32)
    avg_pace = rng.uniform(4.5, 7.5, n_samples).astype(np.float32)  # min/km
    long_run = rng.uniform(10, 35, n_samples).astype(np.float32)
    days_since_long = rng.integers(3, 21, n_samples).astype(np.float32)
    runs_per_week = rng.uniform(3, 7, n_samples).astype(np.float32)
    race_distances = rng.choice(
        [5.0, 10.0, 21.1, 42.2], n_samples
    ).astype(np.float32)

    # Base time = pace * distance (minutes)
    base = avg_pace * race_distances

    # Longer races are harder relative to training pace (Riegel effect)
    distance_penalty = (race_distances / 10.0) ** 0.06

    # More weekly volume -> better aerobic base -> faster
    volume_bonus = np.exp(-0.008 * (weekly_mileage - 50))

    # Bigger long run -> better endurance
    long_run_bonus = np.exp(-0.015 * (long_run - 15))

    # Stale training (long gap since long run) -> slightly slower
    freshness_penalty = 1.0 + 0.005 * days_since_long

    finish_time = (
        base * distance_penalty * volume_bonus * long_run_bonus * freshness_penalty
    )
    finish_time += rng.normal(0, 2, n_samples).astype(np.float32)
    finish_time = np.clip(finish_time, 15.0, 360.0).astype(np.float32)

    return pd.DataFrame({
        "weekly_mileage_km": weekly_mileage,
        "avg_pace_min_per_km": avg_pace,
        "long_run_km": long_run,
        "days_since_long_run": days_since_long,
        "runs_per_week": runs_per_week,
        "race_distance_km": race_distances,
        TARGET_COL: finish_time,
    })


def load_running_csv(path: str | pathlib.Path, dropna: bool = True) -> pd.DataFrame:
    """
    Load a real running export and validate it against the expected schema.

    make_sample_data() documents swapping in your own CSV, but read_csv alone
    does not check that the file actually has the columns the model needs. A
    missing column surfaces later as a KeyError deep inside dataset
    construction, and a column present but misnamed surfaces as silently wrong
    predictions -- so validate up front and fail with a message that says what
    is wrong.

    Args:
        path:   CSV containing FEATURE_COLS + TARGET_COL (extra columns are
                ignored, so a raw export can be passed through unchanged)
        dropna: drop rows with missing values in the required columns. Real
                exports routinely have gaps; a NaN reaching the model produces
                a NaN loss and silently poisons training.

    Returns:
        DataFrame with exactly FEATURE_COLS + TARGET_COL, as float32, with the
        index reset so positional splitting downstream stays predictable.
    """
    csv_path = pathlib.Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV at {csv_path}.")

    df = pd.read_csv(csv_path)

    required = FEATURE_COLS + [TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {missing}. "
            f"Expected all of: {required}"
        )

    out = df[required].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    if dropna:
        before = len(out)
        out = out.dropna()
        dropped = before - len(out)
        if dropped:
            print(f"load_running_csv: dropped {dropped} of {before} rows with missing values")

    if out.empty:
        raise ValueError(f"{csv_path} has no usable rows after validation.")

    return out.reset_index(drop=True)


class RunningDataset(Dataset):
    """
    PyTorch Dataset wrapping a running DataFrame.

    Subclassing Dataset requires two methods:
        __len__  — total number of samples
        __getitem__ — return one (features, target) pair by index

    DataLoader calls these under the hood to build shuffled mini-batches.

    Args:
        df:         DataFrame containing FEATURE_COLS + TARGET_COL
        scaler:     a pre-fitted StandardScaler to apply (or None)
        fit_scaler: if True, fit a new scaler on this data.
                    Only ever set True on the training split — see make_datasets().
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = scaler.transform(X)
        else:
            self.scaler = None

        # Store as tensors so __getitem__ is a direct index, no conversion each call
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_datasets(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple["RunningDataset", "RunningDataset", StandardScaler]:
    """
    Split df and return train/val RunningDatasets with a shared scaler.

    Key rule: fit the StandardScaler on training rows only, then apply
    it to val.  Fitting on the full dataset would leak val statistics
    into the scaler (mean/std), which means the model sees val info
    during training — a subtle but real form of data leakage.

    Returns:
        train_ds, val_ds, scaler
        (scaler is exposed so predict.py can normalize new inputs the same way)
    """
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)

    train_df = df.iloc[idx[n_val:]].reset_index(drop=True)
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)

    train_ds = RunningDataset(train_df, fit_scaler=True)
    val_ds = RunningDataset(val_df, scaler=train_ds.scaler)

    return train_ds, val_ds, train_ds.scaler


def make_synthetic_data(
    n_samples: int = 500,
    n_features: int = 4,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic regression dataset.

    The true relationship is a weighted sum of the features plus a
    nonlinear interaction term, so the MLP has something non-trivial to fit.

    Returns:
        X: shape (n_samples, n_features)  — input features, standardized
        y: shape (n_samples, 1)           — regression target
    """
    rng = np.random.default_rng(seed)

    X_np = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # True target: linear combo + one nonlinear term + noise
    weights = np.array([2.0, -1.5, 0.8, 1.2], dtype=np.float32)[:n_features]
    y_np = X_np @ weights + 0.5 * X_np[:, 0] ** 2
    y_np += rng.standard_normal(n_samples).astype(np.float32) * noise
    y_np = y_np.reshape(-1, 1)

    return torch.from_numpy(X_np), torch.from_numpy(y_np)


def train_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split tensors into train and validation sets."""
    n = len(X)
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=rng)

    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
