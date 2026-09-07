"""
Direct tests for seq_data.PacingSequenceDataset.

The class is reached through make_seq_datasets(), which always hands it
training statistics. That leaves the no-statistics path and the constructor's
own contract unpinned, so they are checked here on tensors small enough to
verify by hand.
"""

from __future__ import annotations

import torch

from pacebrain.seq_data import PacingSequenceDataset


def _races(n: int = 3, t: int = 4, f: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.arange(n * t * f, dtype=torch.float32).reshape(n, t, f)
    y = torch.linspace(4.0, 6.0, n * t).reshape(n, t, 1)
    return X, y


def test_without_statistics_features_pass_through_unscaled():
    X, y = _races()
    ds = PacingSequenceDataset(X, y)
    assert torch.equal(ds.X, X)
    assert torch.equal(ds.y, y)


def test_statistics_standardise_every_timestep_of_every_race():
    X, y = _races()
    mean = torch.arange(6, dtype=torch.float32)
    std = torch.full((6,), 2.0)
    ds = PacingSequenceDataset(X, y, mean=mean, std=std)
    assert torch.allclose(ds.X, (X - mean) / std)


def test_targets_stay_in_raw_pace_units():
    """Scaling y would turn every loss and error into standardised units."""
    X, y = _races()
    ds = PacingSequenceDataset(X, y, mean=torch.zeros(6), std=torch.full((6,), 10.0))
    assert torch.equal(ds.y, y)


def test_scaling_does_not_modify_the_callers_tensor():
    X, y = _races()
    before = X.clone()
    PacingSequenceDataset(X, y, mean=torch.ones(6), std=torch.full((6,), 3.0))
    assert torch.equal(X, before)


def test_len_counts_races_and_getitem_returns_one_race():
    X, y = _races(n=3, t=4)
    ds = PacingSequenceDataset(X, y)
    assert len(ds) == 3
    x_i, y_i = ds[1]
    assert x_i.shape == (4, 6)
    assert y_i.shape == (4, 1)
    assert torch.equal(x_i, X[1])
    assert torch.equal(y_i, y[1])
