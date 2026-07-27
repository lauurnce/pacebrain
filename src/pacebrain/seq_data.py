"""
Sequence data utilities for the pacing model (Day 7).

Tabular data (data.py) gives one row per race. Here each race becomes a
SEQUENCE: 10 distance segments in order, each with features and a pace
target. Order matters — an LSTM can use "what happened in segments 1-5"
to predict the fade in segments 6-10, which a plain MLP cannot.
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset


# Every race is split into N_SEGMENTS equal-distance segments, regardless
# of total distance (segment 1 of a 5K is 0.5 km, of a marathon 4.22 km).
# Fixed length keeps every sequence the same shape, so batching is a simple
# stack. Variable-length sequences (padding + pack_padded_sequence) are the
# realistic extension — deferred until the fixed-length version works.
N_SEGMENTS = 10

# Features per timestep. segment_fraction is the only TIME-VARYING feature:
# the midpoint of each segment, i.e. (i + 0.5) / N_SEGMENTS, telling the
# model "how far into the race are we". The other five are the runner's
# static training features repeated at every timestep. Repeating static
# features like this is a simple, standard way to condition an RNN: the
# network sees the runner's profile at every step, so its hidden state can
# combine "who is running" with "where in the race we are".
SEQ_FEATURES = [
    "segment_fraction",     # midpoint of segment, 0.05 .. 0.95 (time-varying)
    "race_distance_km",     # 5 / 10 / 21.1 / 42.2 (static, repeated)
    "weekly_mileage_km",    # avg km per week in training block (static)
    "avg_pace_min_per_km",  # easy-run pace, lower = faster (static)
    "long_run_km",          # longest run in the block (static)
    "runs_per_week",        # training frequency (static)
]

# Sane bounds for per-segment pace (min/km). Generated paces are clipped here.
PACE_MIN = 3.8
PACE_MAX = 9.0


def make_sample_sequences(
    n_races: int = 600,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic per-segment pacing data.

    Runner features use the same distributions as make_sample_data() in
    data.py, so both pipelines describe the same population of runners.

    Per-segment pace is built from a physics-inspired story:
      1. Race pace is a bit faster than easy training pace, scaled up
         slightly for longer races (you can't hold 5K effort for 42 km).
      2. Pace FADES in the back half. The fade is worse for longer races
         and for runners with an endurance deficit (small long run relative
         to the race, low weekly mileage). The front half is roughly flat.
      3. Small gaussian noise per segment, then clip to a sane range.

    Returns:
        X: float32 (n_races, N_SEGMENTS, 6)  — SEQ_FEATURES per timestep
        y: float32 (n_races, N_SEGMENTS, 1)  — pace in min/km per segment
    """
    rng = np.random.default_rng(seed)

    # Same runner-feature distributions as data.make_sample_data()
    weekly_mileage = rng.uniform(20, 120, n_races).astype(np.float32)
    avg_pace = rng.uniform(4.5, 7.5, n_races).astype(np.float32)  # min/km
    long_run = rng.uniform(10, 35, n_races).astype(np.float32)
    runs_per_week = rng.uniform(3, 7, n_races).astype(np.float32)
    race_distance = rng.choice(
        [5.0, 10.0, 21.1, 42.2], n_races
    ).astype(np.float32)

    # Segment midpoints: 0.05, 0.15, ..., 0.95 — shared by every race
    fracs = ((np.arange(N_SEGMENTS) + 0.5) / N_SEGMENTS).astype(np.float32)

    # 1. Base race pace: ~8% faster than easy pace, but longer races are
    #    run at a relatively slower pace (mild Riegel-style exponent).
    base_pace = avg_pace * 0.92 * (race_distance / 10.0) ** 0.05  # (n,)

    # 2. Fade. Endurance deficit has two parts:
    #    - long run small relative to race distance (never ran far enough)
    #    - low weekly mileage (thin aerobic base)
    #    Both are clipped to [0, 1] so well-trained runners get ~zero fade.
    long_run_deficit = np.clip(1.0 - long_run / race_distance, 0.0, 1.0)
    mileage_deficit = np.clip((60.0 - weekly_mileage) / 60.0, 0.0, 1.0)
    # Longer races punish a deficit harder: a marathon exposes weak
    # endurance, a 5K barely does.
    fade_strength = 1.5 * (long_run_deficit + mileage_deficit) * (
        race_distance / 42.2
    )  # (n,)

    # Fade multiplier per (race, segment): 1.0 through the front half, then
    # quadratic growth after halfway — slow at 60%, steep by 90%, which is
    # how a real blow-up feels. Broadcasting (n,1) * (T,) -> (n, T).
    fade = 1.0 + fade_strength[:, None] * np.maximum(0.0, fracs - 0.5) ** 2

    # 3. Per-segment noise (GPS jitter, terrain, wind) and a sanity clip
    pace = base_pace[:, None] * fade
    pace += rng.normal(0, 0.08, (n_races, N_SEGMENTS))
    pace = np.clip(pace, PACE_MIN, PACE_MAX).astype(np.float32)

    # Assemble X in SEQ_FEATURES order: segment_fraction varies along the
    # time axis, the five static features are repeated at every timestep.
    static = np.stack(
        [race_distance, weekly_mileage, avg_pace, long_run, runs_per_week],
        axis=1,
    )  # (n, 5)
    static_rep = np.repeat(static[:, None, :], N_SEGMENTS, axis=1)  # (n, T, 5)
    frac_rep = np.broadcast_to(fracs, (n_races, N_SEGMENTS))[..., None]  # (n, T, 1)
    X = np.concatenate([frac_rep, static_rep], axis=2).astype(np.float32)

    # Target keeps a trailing dim of 1 so it matches an LSTM head that
    # outputs one value per timestep: (batch, seq_len, 1).
    y = pace[..., None]

    return torch.from_numpy(X), torch.from_numpy(y)


class PacingSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping sequence tensors (X, y).

    Optionally normalizes features with externally computed per-feature
    mean/std tensors of shape (6,) — computed on the TRAINING split only,
    see make_seq_datasets(). Broadcasting handles the rest: (n, T, 6)
    minus (6,) normalizes every timestep of every race.

    Targets stay in raw min/km. Keeping the target unscaled means the
    training loss and any error metric read directly in pace units
    (e.g. "off by 0.2 min/km"), which is far easier to sanity-check than
    an error in standardized units.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        if mean is not None and std is not None:
            X = (X - mean) / std
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_seq_datasets(
    n_races: int = 600,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[PacingSequenceDataset, PacingSequenceDataset, tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate sequences and return train/val datasets plus the scaler stats.

    The split is BY RACE, never by segment. Putting segments 1-5 of a race
    in train and segments 6-10 in val would leak: the model would be graded
    on the back half of races it already saw the front half of, and the
    static features of that race appear in both splits.

    As in make_datasets() (data.py), mean/std are computed on the training
    races only, then applied to both splits — fitting on all data would
    leak val statistics into the scaler.

    Returns:
        train_ds, val_ds, (mean, std)
        (mean/std are returned so inference can normalize new inputs
        exactly the same way)
    """
    X, y = make_sample_sequences(n_races=n_races, seed=seed)

    rng = np.random.default_rng(seed)
    idx = torch.from_numpy(rng.permutation(n_races))
    n_val = int(n_races * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Per-feature stats over all training timesteps: flatten (n, T, 6) to
    # (n*T, 6) so each of the 6 features gets one mean and one std.
    flat = X_train.reshape(-1, len(SEQ_FEATURES))
    mean = flat.mean(dim=0)
    std = flat.std(dim=0)

    train_ds = PacingSequenceDataset(X_train, y_train, mean=mean, std=std)
    val_ds = PacingSequenceDataset(X_val, y_val, mean=mean, std=std)

    return train_ds, val_ds, (mean, std)
