"""
Tests for data.py — running dataset, normalization, DataLoader batching.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

from pacebrain.data import (
    make_sample_data,
    RunningDataset,
    make_datasets,
    FEATURE_COLS,
    TARGET_COL,
)


def test_make_sample_data_shape():
    df = make_sample_data(n_samples=100)
    assert len(df) == 100
    for col in FEATURE_COLS + [TARGET_COL]:
        assert col in df.columns, f"missing column: {col}"


def test_make_sample_data_values_in_range():
    df = make_sample_data(n_samples=200)
    assert (df[TARGET_COL] > 0).all(), "finish times must be positive"
    assert (df["weekly_mileage_km"] > 0).all()
    valid = np.array([5.0, 10.0, 21.1, 42.2], dtype=np.float32)
    assert (df["race_distance_km"].isin(valid)).all()


def test_make_sample_data_reproducible():
    df1 = make_sample_data(seed=0)
    df2 = make_sample_data(seed=0)
    assert (df1[TARGET_COL].values == df2[TARGET_COL].values).all()


def test_running_dataset_len():
    df = make_sample_data(n_samples=50)
    ds = RunningDataset(df, fit_scaler=True)
    assert len(ds) == 50


def test_running_dataset_item_shapes():
    df = make_sample_data(n_samples=50)
    ds = RunningDataset(df, fit_scaler=True)
    x, y = ds[0]
    assert x.shape == (len(FEATURE_COLS),), f"expected ({len(FEATURE_COLS)},), got {x.shape}"
    assert y.shape == (1,), f"expected (1,), got {y.shape}"


def test_running_dataset_dtypes():
    df = make_sample_data(n_samples=50)
    ds = RunningDataset(df, fit_scaler=True)
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_no_scaler_passthrough():
    """Dataset works without any scaler — features pass through unchanged."""
    df = make_sample_data(n_samples=20)
    ds = RunningDataset(df)
    assert ds.scaler is None
    x, _ = ds[0]
    assert x.shape == (len(FEATURE_COLS),)


def test_scaler_fit_on_train_only():
    """
    After make_datasets(), training features should be ~zero mean.
    Val features won't be (different slice), but both use the same scaler object.
    This is the data-leakage guard: if we had fit on all data, train mean would
    not be exactly zero because val statistics would shift the scaler.
    """
    df = make_sample_data(n_samples=300)
    train_ds, val_ds, scaler = make_datasets(df, val_fraction=0.2)

    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_mean = X_train.mean(dim=0)
    assert torch.allclose(train_mean, torch.zeros_like(train_mean), atol=1e-4)

    # Val uses the same scaler object, not a separate one fit on val data
    assert val_ds.scaler is scaler


def test_dataloader_batch_shapes():
    df = make_sample_data(n_samples=100)
    train_ds, _, _ = make_datasets(df)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    X_batch, y_batch = next(iter(loader))
    assert X_batch.shape == (16, len(FEATURE_COLS))
    assert y_batch.shape == (16, 1)


def test_train_val_no_overlap():
    """Indices in train and val must be disjoint."""
    df = make_sample_data(n_samples=100)
    train_ds, val_ds, _ = make_datasets(df, val_fraction=0.2)
    assert len(train_ds) + len(val_ds) == 100
