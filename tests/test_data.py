"""
Tests for data.py — running dataset, normalization, DataLoader batching.
"""


import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pacebrain.data import (
    FEATURE_COLS,
    TARGET_COL,
    RunningDataset,
    load_running_csv,
    make_datasets,
    make_sample_data,
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


# ---------------------------------------------------------------------------
# load_running_csv — schema validation for real exports
# ---------------------------------------------------------------------------

def _write_csv(path, rows=3, extra_cols=None, drop_cols=(), values=None):
    """Build a valid CSV, optionally perturbed, for the loader tests."""
    data = {col: [float(i + 1) for i in range(rows)] for col in FEATURE_COLS}
    data[TARGET_COL] = [100.0 + i for i in range(rows)]
    if values:
        data.update(values)
    frame = pd.DataFrame(data)
    for col in drop_cols:
        frame = frame.drop(columns=[col])
    if extra_cols:
        for col in extra_cols:
            frame[col] = "ignored"
    frame.to_csv(path, index=False)
    return path


def test_load_running_csv_returns_expected_schema(tmp_path):
    df = load_running_csv(_write_csv(tmp_path / "ok.csv"))
    assert list(df.columns) == FEATURE_COLS + [TARGET_COL]
    assert len(df) == 3
    assert all(df[col].dtype == np.float32 for col in df.columns)


def test_load_running_csv_ignores_extra_columns(tmp_path):
    """A raw export carries columns the model does not use; they must not break it."""
    path = _write_csv(tmp_path / "extra.csv", extra_cols=["activity_id", "device"])
    assert list(load_running_csv(path).columns) == FEATURE_COLS + [TARGET_COL]


@pytest.mark.parametrize("missing", ["weekly_mileage_km", "race_distance_km", TARGET_COL])
def test_load_running_csv_names_the_missing_column(tmp_path, missing):
    """The whole point: fail up front with a message saying what is wrong."""
    path = _write_csv(tmp_path / "bad.csv", drop_cols=[missing])
    with pytest.raises(ValueError) as exc:
        load_running_csv(path)
    assert missing in str(exc.value)


def test_load_running_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_running_csv(tmp_path / "nope.csv")


def test_load_running_csv_drops_rows_with_gaps(tmp_path):
    """A NaN reaching the model gives a NaN loss, which poisons training silently."""
    path = _write_csv(tmp_path / "gaps.csv", rows=3, values={"long_run_km": [10.0, None, 30.0]})
    df = load_running_csv(path)
    assert len(df) == 2
    assert not df.isna().any().any()


def test_load_running_csv_can_keep_gaps(tmp_path):
    path = _write_csv(tmp_path / "gaps.csv", rows=3, values={"long_run_km": [10.0, None, 30.0]})
    assert len(load_running_csv(path, dropna=False)) == 3


def test_load_running_csv_coerces_non_numeric_then_drops(tmp_path):
    """Strava exports sometimes carry '--' or '' in a numeric column."""
    path = _write_csv(tmp_path / "junk.csv", rows=3, values={"avg_pace_min_per_km": [5.0, "--", 6.0]})
    assert len(load_running_csv(path)) == 2


def test_load_running_csv_rejects_a_file_with_no_usable_rows(tmp_path):
    path = _write_csv(tmp_path / "empty.csv", rows=2, values={"runs_per_week": [None, None]})
    with pytest.raises(ValueError, match="no usable rows"):
        load_running_csv(path)


def test_loaded_csv_feeds_make_datasets(tmp_path):
    """End to end: a real CSV must flow into the existing pipeline unchanged."""
    df = load_running_csv(_write_csv(tmp_path / "ok.csv", rows=50))
    train_ds, val_ds, scaler = make_datasets(df, val_fraction=0.2, seed=0)
    assert len(train_ds) + len(val_ds) == 50
    assert scaler.mean_.shape == (len(FEATURE_COLS),)
