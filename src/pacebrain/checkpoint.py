"""
Checkpoint format — one place that knows how a saved model is laid out.

Day 8 closed with: "The dict format is self-describing enough to sniff today,
but a `version` key would beat key-presence checks once there are three
formats." There are now three, so this is that key.

Format history:

    v0  bare state_dict                                    (Days 4-7)
    v1  dict: state_dict + scaler_mean/scale + feature_cols (Day 8)
    v2  v1 plus an explicit "version" key                   (this module)

Why sniffing stops working: every check is a guess about what the *absence*
of a key means. `"scaler_mean" not in checkpoint` currently means "written
before Day 8, rebuild the scaler from the seed" — but it would equally match
a future format that moved the scaler somewhere else, and the fallback would
then silently substitute synthetic statistics for the real ones. A version
number says what a checkpoint *is* instead of inferring it from what it lacks,
so an unknown format can fail loudly rather than take the oldest branch.

Reading stays backward compatible: v0 and v1 checkpoints have no version key,
so detect_version() infers theirs from structure. Only new files carry it.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

CHECKPOINT_VERSION = 2

# Versions this code can load. A checkpoint from the future is an error, not
# something to guess at — it was written by code that knew more than this does.
SUPPORTED_VERSIONS = (0, 1, 2)


def detect_version(checkpoint: dict) -> int:
    """
    Determine which format a loaded checkpoint uses.

    v2 says so directly. v0 and v1 predate the key, so they are told apart by
    structure — which is exactly the sniffing this module exists to retire,
    kept only for files already on disk.
    """
    if "version" in checkpoint:
        return int(checkpoint["version"])
    if "state_dict" in checkpoint:
        return 1
    return 0  # a bare state_dict is itself a dict, so this is the fallthrough


def build_checkpoint(
    state_dict: dict,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> dict:
    """
    Assemble a v2 checkpoint.

    Scaler statistics are stored as tensors rather than numpy arrays so the
    file still loads under weights_only=True, which restricts torch.load to
    tensors and simple containers instead of running the full pickle
    machinery. Keeping that flag on is worth the conversion: loading an
    untrusted checkpoint should never be able to execute code.
    """
    return {
        "version": CHECKPOINT_VERSION,
        "state_dict": state_dict,
        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
        "feature_cols": list(feature_cols),
    }


def read_state_dict(checkpoint: dict) -> dict:
    """Pull the weights out of any supported format."""
    version = detect_version(checkpoint)
    _require_supported(version)
    return checkpoint if version == 0 else checkpoint["state_dict"]


def read_scaler(checkpoint: dict) -> StandardScaler | None:
    """
    Rebuild the fitted StandardScaler from a checkpoint, or None if absent.

    None means "this format does not carry one" (v0), and the caller decides
    what to do about it — see inference.load_scaler(). It never means "use
    defaults", which would silently normalise with the wrong statistics.
    """
    version = detect_version(checkpoint)
    _require_supported(version)
    if version == 0 or "scaler_mean" not in checkpoint:
        return None

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"].numpy().astype(np.float64)
    scaler.scale_ = checkpoint["scaler_scale"].numpy().astype(np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler


def check_feature_cols(checkpoint: dict, expected: list[str]) -> None:
    """
    Verify the checkpoint was trained on the feature order in use now.

    This is the failure the version key cannot catch on its own and the one
    that actually costs you: reorder FEATURE_COLS, and every tensor still has
    the right shape, so nothing raises — the model just reads long_run_km
    where it was trained to read weekly_mileage_km and returns confident
    nonsense. Checkpoints that predate the stored list are skipped rather
    than assumed correct.
    """
    saved = checkpoint.get("feature_cols")
    if saved is None:
        return
    if list(saved) != list(expected):
        raise ValueError(
            "Checkpoint feature order does not match the current FEATURE_COLS.\n"
            f"  checkpoint: {list(saved)}\n"
            f"  current   : {list(expected)}\n"
            "Retrain, or restore the original column order — the shapes match "
            "either way, so this would otherwise produce silently wrong predictions."
        )


def _require_supported(version: int) -> None:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Checkpoint format v{version} is not supported by this code "
            f"(known versions: {list(SUPPORTED_VERSIONS)}). It was probably "
            "written by a newer version of pacebrain."
        )
