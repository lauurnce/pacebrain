"""
Tests for eval.py — MAE metric, Riegel baseline formula, model prediction
collection over a DataLoader.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from torch.utils.data import DataLoader

from pacebrain.data import make_sample_data, make_datasets, RunningDataset
from pacebrain.eval import mae_minutes, riegel_predict, get_model_predictions
from pacebrain.models import FinishTimePredictor


# ---------------------------------------------------------------------------
# mae_minutes
# ---------------------------------------------------------------------------

def test_mae_minutes_known_value():
    """Errors of 10, 0 and 2 minutes average to 4.0."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 200.0, 298.0])
    assert mae_minutes(y_true, y_pred) == pytest.approx(4.0)


def test_mae_minutes_perfect_prediction_is_zero():
    y = np.array([12.5, 240.0, 61.25])
    assert mae_minutes(y, y) == 0.0


def test_mae_minutes_is_symmetric_in_sign():
    """Over- and under-predicting by the same amount must cost the same."""
    y_true = np.array([100.0, 100.0])
    over = np.array([107.0, 107.0])
    under = np.array([93.0, 93.0])
    assert mae_minutes(y_true, over) == pytest.approx(mae_minutes(y_true, under))
    # ...and swapping the arguments changes nothing either
    assert mae_minutes(y_true, over) == pytest.approx(mae_minutes(over, y_true))


def test_mae_minutes_returns_plain_float():
    """Callers format this with f-strings, so it must not stay a numpy scalar."""
    result = mae_minutes(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert type(result) is float
    assert result == pytest.approx(1.5)


def test_mae_minutes_single_element():
    assert mae_minutes(np.array([50.0]), np.array([44.5])) == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# riegel_predict
# ---------------------------------------------------------------------------

def test_riegel_at_reference_distance_is_exact_pace_times_distance():
    """
    At race_distance == ref_distance the (D2/D1)^1.06 term is exactly 1.0,
    so the prediction collapses to the 10 km proxy avg_pace * ref_distance.
    Exact equality, not approx — the exponent term must contribute nothing.
    """
    assert riegel_predict(5.0, 10.0, ref_distance_km=10.0) == 50.0
    assert riegel_predict(6.25, 21.1, ref_distance_km=21.1) == 6.25 * 21.1


def test_riegel_longer_race_means_slower_implied_pace():
    """
    The whole point of the 1.06 exponent: per-kilometre pace degrades as the
    race gets longer. 5 km must imply a faster pace than 10 km, which must
    imply a faster pace than a marathon.
    """
    avg_pace = 5.0
    implied = [
        riegel_predict(avg_pace, d) / d for d in (5.0, 10.0, 21.1, 42.2)
    ]
    assert implied == sorted(implied), f"implied paces not monotonic: {implied}"
    assert implied[0] < implied[1] < implied[2] < implied[3]


def test_riegel_shorter_race_is_faster_than_reference_pace():
    """Below the reference distance the runner is predicted faster than proxy pace."""
    avg_pace = 5.0
    assert riegel_predict(avg_pace, 5.0) / 5.0 < avg_pace
    assert riegel_predict(avg_pace, 42.2) / 42.2 > avg_pace


def test_riegel_total_time_grows_with_distance():
    """Longer race, longer total time — superlinearly, thanks to the exponent."""
    avg_pace = 5.5
    t_half = riegel_predict(avg_pace, 21.1)
    t_full = riegel_predict(avg_pace, 42.2)
    assert t_full > t_half
    assert t_full > 2 * t_half, "marathon must cost more than 2x the half"


def test_riegel_is_vectorised_over_numpy_arrays():
    paces = np.array([4.5, 5.0, 6.0, 7.5])
    distances = np.array([5.0, 10.0, 21.1, 42.2])

    out = riegel_predict(paces, distances)

    assert isinstance(out, np.ndarray)
    assert out.shape == paces.shape
    expected = [riegel_predict(float(p), float(d)) for p, d in zip(paces, distances)]
    np.testing.assert_allclose(out, expected)


def test_riegel_scales_linearly_with_pace():
    """T1 is linear in avg_pace, so doubling pace doubles the predicted time."""
    single = riegel_predict(5.0, 42.2)
    double = riegel_predict(10.0, 42.2)
    assert double == pytest.approx(2 * single)


def test_riegel_custom_reference_distance():
    """A 5 km proxy and a 10 km proxy of the same runner disagree as expected."""
    from_5k = riegel_predict(5.0, 42.2, ref_distance_km=5.0)
    from_10k = riegel_predict(5.0, 42.2, ref_distance_km=10.0)
    # Holding the same pace for only 5 km is a weaker performance than holding
    # it for 10 km, so extrapolating from the 5 km proxy gives a slower marathon.
    assert from_5k > from_10k


# ---------------------------------------------------------------------------
# get_model_predictions
# ---------------------------------------------------------------------------

def _tiny_model():
    """A small, dropout-free FinishTimePredictor — no checkpoint needed."""
    return FinishTimePredictor(hidden_sizes=[4], dropout=0.0)


def test_get_model_predictions_shapes():
    df = make_sample_data(n_samples=37)
    ds = RunningDataset(df, fit_scaler=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    y_true, y_pred = get_model_predictions(_tiny_model(), loader)

    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.ndim == 1 and y_pred.ndim == 1, "both must be flattened to 1-D"
    assert len(y_true) == len(y_pred) == len(ds) == 37


def test_get_model_predictions_recovers_targets_in_order():
    """With shuffle=False, y_true must be the dataset targets in dataset order."""
    df = make_sample_data(n_samples=24)
    ds = RunningDataset(df, fit_scaler=True)
    loader = DataLoader(ds, batch_size=5, shuffle=False)

    y_true, _ = get_model_predictions(_tiny_model(), loader)

    np.testing.assert_allclose(y_true, ds.y.numpy().ravel())


def test_get_model_predictions_leaves_model_in_eval_mode():
    """model.eval() is what makes these predictions deterministic."""
    df = make_sample_data(n_samples=20)
    ds = RunningDataset(df, fit_scaler=True)
    loader = DataLoader(ds, batch_size=10, shuffle=False)

    model = FinishTimePredictor(hidden_sizes=[4], dropout=0.5)
    model.train()  # deliberately start in training mode

    get_model_predictions(model, loader)

    assert not model.training


def test_get_model_predictions_is_deterministic_with_dropout():
    """Dropout is active in train() but must be disabled here — same run twice."""
    df = make_sample_data(n_samples=30)
    ds = RunningDataset(df, fit_scaler=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model = FinishTimePredictor(hidden_sizes=[8, 4], dropout=0.5)

    _, first = get_model_predictions(model, loader)
    _, second = get_model_predictions(model, loader)

    np.testing.assert_array_equal(first, second)


def test_get_model_predictions_over_val_split():
    """The realistic call site: the val DataLoader built by make_datasets()."""
    df = make_sample_data(n_samples=100)
    _, val_ds, _ = make_datasets(df, val_fraction=0.2)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    y_true, y_pred = get_model_predictions(_tiny_model(), loader)

    assert len(y_true) == len(val_ds) == 20
    # An untrained model is bad, but MAE must still be a finite number.
    assert np.isfinite(mae_minutes(y_true, y_pred))
