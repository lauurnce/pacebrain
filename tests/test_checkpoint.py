"""
Tests for the versioned checkpoint format.

Two things need holding down: every historical format still loads, and a
format this code does not understand fails loudly instead of being guessed
at. The feature-order check gets its own section because it is the one
failure that produces wrong numbers rather than an exception.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from pacebrain.checkpoint import (
    CHECKPOINT_VERSION,
    build_checkpoint,
    check_feature_cols,
    detect_version,
    read_scaler,
    read_state_dict,
)
from pacebrain.data import FEATURE_COLS
from pacebrain.models import FinishTimePredictor


@pytest.fixture
def scaler():
    s = StandardScaler()
    s.fit(np.random.default_rng(0).normal(size=(50, len(FEATURE_COLS))))
    return s


@pytest.fixture
def state_dict():
    return FinishTimePredictor(input_size=len(FEATURE_COLS)).state_dict()


@pytest.fixture
def v0(state_dict):
    """Days 4-7: a bare state_dict."""
    return state_dict


@pytest.fixture
def v1(state_dict, scaler):
    """Day 8: dict with the scaler, but no version key."""
    return {
        "state_dict": state_dict,
        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
        "feature_cols": list(FEATURE_COLS),
    }


@pytest.fixture
def v2(state_dict, scaler):
    return build_checkpoint(state_dict, scaler, FEATURE_COLS)


# ---------------------------------------------------------------------------
# version detection
# ---------------------------------------------------------------------------

def test_detects_each_format(v0, v1, v2):
    assert detect_version(v0) == 0
    assert detect_version(v1) == 1
    assert detect_version(v2) == 2


def test_new_checkpoints_carry_the_current_version(v2):
    assert v2["version"] == CHECKPOINT_VERSION


def test_a_future_format_is_an_error_not_a_guess(v2):
    """
    The whole point of the version key: an unknown format must fail rather
    than silently fall through to the oldest branch.
    """
    v2["version"] = 99
    with pytest.raises(ValueError, match="not supported"):
        read_state_dict(v2)
    with pytest.raises(ValueError, match="not supported"):
        read_scaler(v2)


# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------

def test_every_format_yields_loadable_weights(v0, v1, v2):
    for checkpoint in (v0, v1, v2):
        model = FinishTimePredictor(input_size=len(FEATURE_COLS))
        model.load_state_dict(read_state_dict(checkpoint))  # must not raise


def test_v1_and_v2_return_the_same_scaler_statistics(v1, v2, scaler):
    for checkpoint in (v1, v2):
        loaded = read_scaler(checkpoint)
        assert np.allclose(loaded.mean_, scaler.mean_)
        assert np.allclose(loaded.scale_, scaler.scale_)


def test_v0_has_no_scaler_and_says_so(v0):
    """None means 'absent', so the caller can fall back deliberately."""
    assert read_scaler(v0) is None


def test_loaded_scaler_transforms_identically_to_the_original(v2, scaler):
    """The statistics round-tripping is only useful if transform() matches."""
    rows = np.random.default_rng(1).normal(size=(4, len(FEATURE_COLS)))
    assert np.allclose(read_scaler(v2).transform(rows), scaler.transform(rows))


def test_checkpoint_survives_weights_only_load(v2, tmp_path):
    """
    weights_only=True restricts torch.load to tensors and simple containers.
    Storing the scaler as numpy arrays would break this, which is why
    build_checkpoint converts them.
    """
    path = tmp_path / "ckpt.pt"
    torch.save(v2, path)
    reloaded = torch.load(path, weights_only=True)
    assert detect_version(reloaded) == CHECKPOINT_VERSION
    assert read_scaler(reloaded) is not None


# ---------------------------------------------------------------------------
# feature-order validation
# ---------------------------------------------------------------------------

def test_matching_feature_order_passes(v2):
    check_feature_cols(v2, FEATURE_COLS)  # must not raise


def test_reordered_features_are_rejected(v2):
    """
    The failure that motivates this check: swapping two columns keeps every
    shape valid, so load_state_dict succeeds and the model reads the wrong
    feature at each position. Nothing else in the pipeline would notice.
    """
    swapped = list(FEATURE_COLS)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="feature order"):
        check_feature_cols(v2, swapped)


def test_renamed_feature_is_rejected(v2):
    renamed = list(FEATURE_COLS)
    renamed[-1] = "distance_km"
    with pytest.raises(ValueError, match="feature order"):
        check_feature_cols(v2, renamed)


def test_v0_skips_the_check_rather_than_assuming_it_passed(v0):
    """A format with no stored list cannot be validated, so it is not."""
    check_feature_cols(v0, FEATURE_COLS)  # must not raise
