"""
Tests for seq_data.py — sequence generation, fade behavior, normalization,
DataLoader batching.
"""


import torch
from torch.utils.data import DataLoader

from pacebrain.seq_data import (
    N_SEGMENTS,
    PACE_MAX,
    PACE_MIN,
    SEQ_FEATURES,
    make_sample_sequences,
    make_seq_datasets,
)


def test_make_sample_sequences_shapes_and_dtypes():
    X, y = make_sample_sequences(n_races=50)
    assert X.shape == (50, N_SEGMENTS, len(SEQ_FEATURES))
    assert y.shape == (50, N_SEGMENTS, 1)
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32


def test_make_sample_sequences_deterministic():
    X1, y1 = make_sample_sequences(n_races=50, seed=7)
    X2, y2 = make_sample_sequences(n_races=50, seed=7)
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_pace_values_in_clipped_range():
    _, y = make_sample_sequences(n_races=50)
    assert (y >= PACE_MIN).all()
    assert (y <= PACE_MAX).all()


def test_low_endurance_marathoners_fade():
    """
    The core property the LSTM is supposed to learn: marathoners whose
    longest run is short (< 20 km) should slow down in the back half.
    Compare mean pace of the last 3 segments vs the first 3 per race.
    """
    # 300 races so the deterministic seed yields plenty of weak marathoners
    X, y = make_sample_sequences(n_races=300)

    race_distance = X[:, 0, SEQ_FEATURES.index("race_distance_km")]
    long_run = X[:, 0, SEQ_FEATURES.index("long_run_km")]
    weak_marathon = (race_distance == 42.2) & (long_run < 20.0)
    assert weak_marathon.any(), "seed produced no low-endurance marathoners"

    paces = y[weak_marathon, :, 0]  # (k, N_SEGMENTS)
    first_3 = paces[:, :3].mean(dim=1)
    last_3 = paces[:, -3:].mean(dim=1)
    assert (last_3 > first_3).all(), "expected fade in low-endurance marathons"


def test_scaler_stats_from_train_only():
    """
    Leakage guard: mean/std must come from the training races alone, so
    they differ from stats over the full dataset, and the normalized
    training features end up ~zero mean / ~unit std.
    """
    train_ds, _, (mean, std) = make_seq_datasets(n_races=100, val_fraction=0.2)

    X_full, _ = make_sample_sequences(n_races=100)
    full_mean = X_full.reshape(-1, len(SEQ_FEATURES)).mean(dim=0)
    assert not torch.allclose(mean, full_mean), (
        "train mean equals full-data mean — scaler leaked val statistics"
    )

    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    flat = X_train.reshape(-1, len(SEQ_FEATURES))
    assert torch.allclose(flat.mean(dim=0), torch.zeros(len(SEQ_FEATURES)), atol=1e-4)
    assert torch.allclose(flat.std(dim=0), torch.ones(len(SEQ_FEATURES)), atol=1e-3)


def test_targets_stay_in_raw_units():
    """Normalization applies to features only — y stays in min/km."""
    train_ds, _, _ = make_seq_datasets(n_races=50)
    _, y = train_ds[0]
    assert (y >= PACE_MIN).all() and (y <= PACE_MAX).all()


def test_train_val_split_sizes():
    train_ds, val_ds, _ = make_seq_datasets(n_races=50, val_fraction=0.2)
    assert len(train_ds) == 40
    assert len(val_ds) == 10


def test_dataloader_batch_shapes():
    train_ds, _, _ = make_seq_datasets(n_races=50)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    X_batch, y_batch = next(iter(loader))
    assert X_batch.shape == (8, N_SEGMENTS, len(SEQ_FEATURES))
    assert y_batch.shape == (8, N_SEGMENTS, 1)
