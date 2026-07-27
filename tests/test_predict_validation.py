"""
Tests for predict.py input validation — the error bucket vs the warn bucket.

Two behaviours are being pinned down here:
    impossible input  -> exit 2, nothing runs
    unusual input     -> warning on stderr, prediction still happens
Mixing those up is the actual bug this file guards against.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest

from pacebrain import predict
from pacebrain.predict import check_training_range, validate_inputs


# A runner sitting comfortably inside every training range, used as the
# baseline that individual tests perturb one field at a time.
VALID = {
    "weekly_mileage": 60.0,
    "avg_pace": 5.5,
    "long_run": 28.0,
    "days_since_long_run": 7.0,
    "runs_per_week": 4.0,
    "race_distance": 42.2,
}


def args_with(**overrides):
    """Copy VALID with the given fields replaced."""
    merged = dict(VALID)
    merged.update(overrides)
    return merged


def cli_args(**overrides):
    """Build an argv list for predict.py from VALID plus any overrides."""
    merged = args_with(**overrides)
    argv = ["predict.py"]
    for key, value in merged.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return argv


# --- bucket 1: impossible input is rejected -------------------------------

def test_valid_inputs_produce_no_errors():
    assert validate_inputs(**VALID) == []


@pytest.mark.parametrize("field", [
    "weekly_mileage",
    "avg_pace",
    "long_run",
    "runs_per_week",
    "race_distance",
])
def test_negative_magnitude_rejected(field):
    errors = validate_inputs(**args_with(**{field: -5.0}))
    assert len(errors) == 1
    assert field.replace("_", "-") in errors[0]


@pytest.mark.parametrize("field", [
    "weekly_mileage",
    "avg_pace",
    "long_run",
    "runs_per_week",
    "race_distance",
])
def test_zero_magnitude_rejected(field):
    """Zero is as meaningless as negative for every magnitude field."""
    errors = validate_inputs(**args_with(**{field: 0.0}))
    assert len(errors) == 1
    assert field.replace("_", "-") in errors[0]


def test_zero_race_distance_rejected():
    """
    The regression case: main() divides minutes by race distance to report the
    implied pace, so a zero used to reach the model and then crash with
    ZeroDivisionError.
    """
    errors = validate_inputs(**args_with(race_distance=0.0))
    assert errors and "--race-distance" in errors[0]


def test_negative_days_since_long_run_rejected():
    errors = validate_inputs(**args_with(days_since_long_run=-1.0))
    assert len(errors) == 1
    assert "--days-since-long-run" in errors[0]


def test_zero_days_since_long_run_allowed():
    """Zero days is a long run done today — unusual, not impossible."""
    assert validate_inputs(**args_with(days_since_long_run=0.0)) == []


def test_multiple_bad_fields_all_reported():
    errors = validate_inputs(**args_with(weekly_mileage=-5.0, race_distance=0.0))
    assert len(errors) == 2


def test_nan_rejected():
    """nan compares False against every bound, so it needs its own check."""
    errors = validate_inputs(**args_with(avg_pace=float("nan")))
    assert len(errors) == 1
    assert "--avg-pace" in errors[0]


# --- bucket 1 through the CLI: exit code and ordering ---------------------

def test_cli_exits_2_on_invalid_input(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", cli_args(weekly_mileage=-5.0))
    with pytest.raises(SystemExit) as exc:
        predict.parse_args()
    assert exc.value.code == 2
    assert "--weekly-mileage" in capsys.readouterr().err


def test_validation_runs_before_model_is_loaded(monkeypatch):
    """
    Bad input must cost an error message, not a checkpoint load.  If ordering
    ever regresses, the load_finish_model stub below fires instead of the exit.
    """
    def boom(*_args, **_kwargs):
        raise AssertionError("model was loaded despite invalid input")

    monkeypatch.setattr(predict, "rebuild_scaler", boom)
    monkeypatch.setattr(predict, "load_finish_model", boom)
    monkeypatch.setattr(sys, "argv", cli_args(race_distance=0.0))

    with pytest.raises(SystemExit) as exc:
        predict.main()
    assert exc.value.code == 2


# --- bucket 2: out-of-distribution input warns but still predicts ---------

def test_valid_typical_inputs_produce_no_warnings():
    assert check_training_range(**VALID) == []


@pytest.mark.parametrize("field,value", [
    ("weekly_mileage", 130.0),   # above the 20-120 the model trained on
    ("weekly_mileage", 10.0),
    ("avg_pace", 3.5),           # faster than any training runner
    ("avg_pace", 9.0),
    ("long_run", 45.0),
    ("long_run", 5.0),
    ("days_since_long_run", 60.0),
    ("days_since_long_run", 0.0),
    ("runs_per_week", 10.0),
    ("runs_per_week", 1.0),
])
def test_out_of_range_field_warns(field, value):
    warnings = check_training_range(**args_with(**{field: value}))
    assert len(warnings) == 1
    assert field.replace("_", "-") in warnings[0]


def test_unseen_race_distance_warns():
    """15 km is a real race, just not one in the training set."""
    warnings = check_training_range(**args_with(race_distance=15.0))
    assert len(warnings) == 1
    assert "--race-distance" in warnings[0]


@pytest.mark.parametrize("distance", [5.0, 10.0, 21.1, 42.2])
def test_trained_race_distances_do_not_warn(distance):
    assert check_training_range(**args_with(race_distance=distance)) == []


def test_out_of_range_input_is_not_an_error():
    """The whole point of the split: a 130 km/week runner is not rejected."""
    assert validate_inputs(**args_with(weekly_mileage=130.0)) == []


def stub_model(monkeypatch, minutes=240.0):
    """Replace the checkpoint-backed helpers so main() can run end to end."""
    monkeypatch.setattr(predict, "rebuild_scaler", lambda cfg: None)
    monkeypatch.setattr(predict, "load_finish_model", lambda cfg: None)
    monkeypatch.setattr(
        predict, "predict_finish_time", lambda model, scaler, features: minutes
    )


def test_cli_warns_on_stderr_but_still_predicts(monkeypatch, capsys):
    stub_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", cli_args(weekly_mileage=130.0))

    predict.main()

    captured = capsys.readouterr()
    assert "--weekly-mileage" in captured.err
    assert "unreliable" in captured.err
    # The prediction itself still lands on stdout, unchanged in format.
    assert "Predicted time" in captured.out
    assert "4:00:00" in captured.out


def test_cli_stays_silent_for_typical_input(monkeypatch, capsys):
    stub_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", cli_args())

    predict.main()

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Predicted time" in captured.out
