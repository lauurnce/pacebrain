"""
Direct tests for seq_data.VariableLengthPacingDataset.

The class is reached through make_variable_seq_datasets(), which always hands
it training statistics. That leaves the no-statistics path unpinned, and the
per-race handling of differing lengths is only implied, so both are checked
here on races small enough to verify by hand.
"""

from __future__ import annotations

import torch

from pacebrain.seq_data import VariableLengthPacingDataset, make_variable_length_sequences

LENGTHS = (5, 10, 3)


def _ragged() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.arange(t * 6, dtype=torch.float32).reshape(t, 6),
            torch.full((t, 1), 5.0),
        )
        for t in LENGTHS
    ]


def test_without_statistics_races_pass_through_unscaled():
    races = _ragged()
    ds = VariableLengthPacingDataset(races)
    assert len(ds) == len(LENGTHS)
    for (x, y), (x_in, y_in) in zip(ds.races, races):
        assert torch.equal(x, x_in)
        assert torch.equal(y, y_in)


def test_statistics_standardise_each_race_whatever_its_length():
    races = _ragged()
    mean = torch.arange(6, dtype=torch.float32)
    std = torch.full((6,), 2.0)
    ds = VariableLengthPacingDataset(races, mean=mean, std=std)
    for i, t in enumerate(LENGTHS):
        x, _ = ds[i]
        assert x.shape == (t, 6)
        assert torch.allclose(x, (races[i][0] - mean) / std)


def test_targets_stay_in_raw_pace_units():
    races = _ragged()
    ds = VariableLengthPacingDataset(races, mean=torch.zeros(6), std=torch.full((6,), 10.0))
    for i, (_, y_in) in enumerate(races):
        assert torch.equal(ds[i][1], y_in)


def test_scaling_leaves_the_callers_list_alone():
    races = _ragged()
    before = [(x.clone(), y.clone()) for x, y in races]
    VariableLengthPacingDataset(races, mean=torch.ones(6), std=torch.full((6,), 3.0))
    for (x, y), (x_before, y_before) in zip(races, before):
        assert torch.equal(x, x_before)
        assert torch.equal(y, y_before)


def test_generated_races_keep_their_own_lengths():
    ds = VariableLengthPacingDataset(make_variable_length_sequences(n_races=12, seed=0))
    lengths = {ds[i][0].shape[0] for i in range(len(ds))}
    assert len(ds) == 12
    assert len(lengths) > 1
