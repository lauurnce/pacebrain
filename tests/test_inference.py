"""
Tests for inference.py and the predict.py CLI formatters — checkpoint
loading, scaler reconstruction, single-runner prediction, H:MM:SS / M:SS output.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from pacebrain.config import FinishPredictorConfig
from pacebrain.data import FEATURE_COLS, make_sample_data, make_datasets
from pacebrain.inference import load_finish_model, predict_finish_time, rebuild_scaler
from pacebrain.models import FinishTimePredictor
from pacebrain.predict import format_hms, format_pace


# Small config so the synthetic-data rebuild stays fast in tests.
def _small_cfg(**overrides) -> FinishPredictorConfig:
    cfg = FinishPredictorConfig(n_samples=200, hidden_sizes=[8, 4], dropout=0.0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _fitted_scaler():
    df = make_sample_data(n_samples=200, seed=1)
    _, _, scaler = make_datasets(df, val_fraction=0.2, seed=1)
    return scaler


def _tiny_model(seed: int = 0) -> FinishTimePredictor:
    """A small dropout-free predictor with deterministic weights."""
    torch.manual_seed(seed)
    return FinishTimePredictor(hidden_sizes=[8, 4], dropout=0.0)


# ---------------------------------------------------------------------------
# load_finish_model
# ---------------------------------------------------------------------------

def test_load_finish_model_missing_checkpoint_raises(tmp_path):
    """
    models/ is gitignored, so a fresh clone has no checkpoint. The error must
    name the path and point the user at the training script.
    """
    missing = tmp_path / "nope" / "finish_predictor.pt"
    cfg = _small_cfg(checkpoint_path=str(missing))

    with pytest.raises(FileNotFoundError) as excinfo:
        load_finish_model(cfg)

    message = str(excinfo.value)
    assert str(missing) in message, "error should name the missing path"
    assert "train_finish.py" in message, "error should tell the user how to fix it"


def test_load_finish_model_round_trip(tmp_path):
    """Weights saved as a state_dict must come back identical."""
    cfg = _small_cfg(checkpoint_path=str(tmp_path / "finish_predictor.pt"))
    saved = FinishTimePredictor(hidden_sizes=cfg.hidden_sizes, dropout=cfg.dropout)
    torch.save(saved.state_dict(), cfg.checkpoint_path)

    loaded = load_finish_model(cfg)

    assert isinstance(loaded, FinishTimePredictor)
    for name, tensor in saved.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), f"weight mismatch: {name}"


def test_load_finish_model_returns_eval_mode(tmp_path):
    """Docstring promises the model comes back ready for inference."""
    cfg = _small_cfg(checkpoint_path=str(tmp_path / "finish_predictor.pt"))
    torch.save(
        FinishTimePredictor(hidden_sizes=cfg.hidden_sizes, dropout=cfg.dropout).state_dict(),
        cfg.checkpoint_path,
    )

    assert not load_finish_model(cfg).training


def test_load_finish_model_architecture_mismatch_raises(tmp_path):
    """
    A state_dict holds weights, not architecture. Building with different
    hidden_sizes than the checkpoint must fail loudly, not load garbage.
    """
    path = tmp_path / "finish_predictor.pt"
    torch.save(FinishTimePredictor(hidden_sizes=[16]).state_dict(), path)
    cfg = _small_cfg(checkpoint_path=str(path), hidden_sizes=[64, 32])

    with pytest.raises(RuntimeError):
        load_finish_model(cfg)


# ---------------------------------------------------------------------------
# rebuild_scaler
# ---------------------------------------------------------------------------

def test_rebuild_scaler_is_deterministic():
    """
    Load-bearing invariant: the scaler is NOT in the checkpoint, so inference
    refits it from seeded synthetic data. Two calls with the same config must
    produce bit-identical statistics or every CLI prediction silently drifts.
    """
    cfg = _small_cfg()

    first = rebuild_scaler(cfg)
    second = rebuild_scaler(cfg)

    np.testing.assert_array_equal(first.mean_, second.mean_)
    np.testing.assert_array_equal(first.scale_, second.scale_)
    np.testing.assert_array_equal(first.var_, second.var_)


def test_rebuild_scaler_matches_training_pipeline():
    """
    The docstring claims the refit scaler is identical to the one fitted during
    training. train_finish.train() builds it as make_datasets(make_sample_data(
    n_samples=cfg.n_samples, seed=cfg.seed), val_fraction=cfg.val_fraction,
    seed=cfg.seed) — reproduce that here and compare exactly.
    """
    cfg = _small_cfg()
    df = make_sample_data(n_samples=cfg.n_samples, seed=cfg.seed)
    _, _, training_scaler = make_datasets(df, val_fraction=cfg.val_fraction, seed=cfg.seed)

    rebuilt = rebuild_scaler(cfg)

    np.testing.assert_array_equal(rebuilt.mean_, training_scaler.mean_)
    np.testing.assert_array_equal(rebuilt.scale_, training_scaler.scale_)


def test_rebuild_scaler_shape_matches_feature_cols():
    scaler = rebuild_scaler(_small_cfg())
    assert scaler.mean_.shape == (len(FEATURE_COLS),)
    assert scaler.scale_.shape == (len(FEATURE_COLS),)


def test_rebuild_scaler_differs_across_seeds():
    """Sanity check that the determinism above isn't just a constant scaler."""
    a = rebuild_scaler(_small_cfg(seed=42))
    b = rebuild_scaler(_small_cfg(seed=7))
    assert not np.array_equal(a.mean_, b.mean_)


# ---------------------------------------------------------------------------
# predict_finish_time
# ---------------------------------------------------------------------------

MARATHON_FEATURES = {
    "weekly_mileage_km": 60.0,
    "avg_pace_min_per_km": 5.5,
    "long_run_km": 28.0,
    "days_since_long_run": 7.0,
    "runs_per_week": 4.0,
    "race_distance_km": 42.2,
}


def test_predict_finish_time_returns_plain_float():
    """The CLI formats this with f-strings, so a tensor or numpy scalar won't do."""
    result = predict_finish_time(_tiny_model(), _fitted_scaler(), dict(MARATHON_FEATURES))

    assert type(result) is float
    assert not isinstance(result, torch.Tensor)
    assert np.isfinite(result)


def test_predict_finish_time_ignores_dict_insertion_order():
    """
    predict_finish_time must index the dict by FEATURE_COLS, never rely on
    insertion order. A caller building the dict in any order must get the
    same answer — otherwise features silently swap columns.
    """
    model, scaler = _tiny_model(), _fitted_scaler()

    in_order = {col: MARATHON_FEATURES[col] for col in FEATURE_COLS}
    reversed_order = {col: MARATHON_FEATURES[col] for col in reversed(FEATURE_COLS)}
    shuffled_order = {
        col: MARATHON_FEATURES[col]
        for col in ["race_distance_km", "long_run_km", "runs_per_week",
                    "avg_pace_min_per_km", "days_since_long_run", "weekly_mileage_km"]
    }

    assert list(reversed_order) != list(in_order), "test setup: orders must differ"
    assert list(shuffled_order) != list(in_order), "test setup: orders must differ"

    baseline = predict_finish_time(model, scaler, in_order)
    assert predict_finish_time(model, scaler, reversed_order) == baseline
    assert predict_finish_time(model, scaler, shuffled_order) == baseline


def test_predict_finish_time_is_sensitive_to_feature_identity():
    """
    Guards the ordering test above from being vacuous: if the model ignored
    its inputs, any column permutation would trivially "pass". Swapping two
    feature *values* must move the prediction.
    """
    model, scaler = _tiny_model(), _fitted_scaler()

    swapped = dict(MARATHON_FEATURES)
    swapped["weekly_mileage_km"] = MARATHON_FEATURES["race_distance_km"]
    swapped["race_distance_km"] = MARATHON_FEATURES["weekly_mileage_km"]

    baseline = predict_finish_time(model, scaler, dict(MARATHON_FEATURES))
    assert predict_finish_time(model, scaler, swapped) != baseline


def test_predict_finish_time_missing_feature_raises():
    """A dict missing a required column must fail loudly, not predict nonsense."""
    incomplete = dict(MARATHON_FEATURES)
    del incomplete["race_distance_km"]

    with pytest.raises(KeyError):
        predict_finish_time(_tiny_model(), _fitted_scaler(), incomplete)


def test_predict_finish_time_ignores_extra_keys():
    """Only FEATURE_COLS are read, so unrelated keys are harmless."""
    model, scaler = _tiny_model(), _fitted_scaler()
    padded = dict(MARATHON_FEATURES, resting_hr=48, shoe="carbon plate")

    assert predict_finish_time(model, scaler, padded) == predict_finish_time(
        model, scaler, dict(MARATHON_FEATURES)
    )


def test_predict_finish_time_is_deterministic_with_dropout():
    """model.eval() inside the function must disable Dropout."""
    torch.manual_seed(0)
    model = FinishTimePredictor(hidden_sizes=[16, 8], dropout=0.5)
    model.train()  # deliberately start in training mode
    scaler = _fitted_scaler()

    first = predict_finish_time(model, scaler, dict(MARATHON_FEATURES))
    second = predict_finish_time(model, scaler, dict(MARATHON_FEATURES))

    assert first == second
    assert not model.training


def test_predict_finish_time_uses_supplied_scaler():
    """
    Never refit on one row — the model only understands training-set statistics.
    Two different scalers must therefore give two different answers.
    """
    model = _tiny_model()
    scaler_a = _fitted_scaler()
    _, _, scaler_b = make_datasets(make_sample_data(n_samples=200, seed=99), seed=99)

    assert predict_finish_time(model, scaler_a, dict(MARATHON_FEATURES)) != (
        predict_finish_time(model, scaler_b, dict(MARATHON_FEATURES))
    )


# ---------------------------------------------------------------------------
# format_hms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "minutes,expected",
    [
        (245.3, "4:05:18"),   # docstring example
        (0.0, "0:00:00"),
        (1.0, "0:01:00"),
        (59.5, "0:59:30"),    # sub-hour: no hour padding, minutes stay two-digit
        (60.0, "1:00:00"),    # exact hour
        (120.0, "2:00:00"),   # exact hour, multi-hour
        (90.0, "1:30:00"),
        (125.25, "2:05:15"),
        (600.0, "10:00:00"),  # hours are not zero-padded or truncated
    ],
)
def test_format_hms_known_values(minutes, expected):
    assert format_hms(minutes) == expected


def test_format_hms_zero_pads_minutes_and_seconds():
    assert format_hms(65.05) == "1:05:03"


def test_format_hms_rounds_to_nearest_second():
    assert format_hms(0.008) == "0:00:00"    # 0.48 s rounds down
    assert format_hms(0.02) == "0:00:01"     # 1.2 s rounds down to 1
    assert format_hms(0.0158) == "0:00:01"   # 0.948 s rounds up


def test_format_hms_rounding_carries_across_boundaries():
    """3599.94 s must round to 3600 and roll over into a whole hour."""
    assert format_hms(59.999) == "1:00:00"
    assert format_hms(1.9999) == "0:02:00"


def test_format_hms_returns_string():
    assert isinstance(format_hms(245.3), str)


# ---------------------------------------------------------------------------
# format_pace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "min_per_km,expected",
    [
        (5.82, "5:49"),   # docstring example
        (5.0, "5:00"),    # exact minute
        (4.5, "4:30"),
        (0.5, "0:30"),
        (10.5, "10:30"),  # minutes are not padded
        (7.25, "7:15"),
    ],
)
def test_format_pace_known_values(min_per_km, expected):
    assert format_pace(min_per_km) == expected


def test_format_pace_zero_pads_seconds():
    """Single-digit seconds must render as 5:03, never 5:3."""
    assert format_pace(5.05) == "5:03"
    assert format_pace(6.0166667) == "6:01"


def test_format_pace_rounding_carries_into_the_next_minute():
    assert format_pace(5.9999) == "6:00"


def test_format_pace_returns_string():
    assert isinstance(format_pace(5.82), str)
