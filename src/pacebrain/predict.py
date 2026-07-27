"""
Predict a race finish time from recent training stats — Day 6 inference CLI.

Takes the same six features the model was trained on, scales them with the
rebuilt training scaler, and runs one forward pass through the checkpointed
FinishTimePredictor. No training happens here; this is pure inference.

Run:
    python src/pacebrain/predict.py --weekly-mileage 60 --avg-pace 5.5 \
        --long-run 28 --race-distance 42.2
"""

from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import math

from pacebrain.config import FinishPredictorConfig
from pacebrain.inference import load_finish_model, predict_finish_time, rebuild_scaler


# The ranges make_sample_data() draws each feature from in data.py.  The model
# has literally never seen a value outside these, so anything beyond them is
# extrapolation rather than prediction.  Kept as (low, high) per CLI flag so the
# warning can name the flag the user actually typed.
TRAINING_RANGES = {
    "--weekly-mileage": (20.0, 120.0),
    "--avg-pace": (4.5, 7.5),
    "--long-run": (10.0, 35.0),
    "--days-since-long-run": (3.0, 21.0),
    "--runs-per-week": (3.0, 7.0),
}

# make_sample_data() only ever picks from these four, so a 15 km race is an
# unseen distance even though it is a perfectly sensible thing to run.
TRAINING_RACE_DISTANCES = (5.0, 10.0, 21.1, 42.2)


def validate_inputs(
    weekly_mileage: float,
    avg_pace: float,
    long_run: float,
    days_since_long_run: float,
    runs_per_week: float,
    race_distance: float,
) -> list[str]:
    """
    Return one message per physically impossible input; empty list means OK.

    These are hard errors rather than warnings because there is no runner they
    could describe — a negative weekly mileage or a zero-length race isn't an
    unusual athlete, it's a typo.  Predicting on them would hand back a
    confident-looking number with nothing behind it.

    --race-distance of 0 is the sharp case: main() computes
    `minutes / race_distance` for the implied pace, so a zero raced through the
    model and then blew up with ZeroDivisionError.  Catching it here means it
    never reaches either.

    Pure function on purpose (floats in, strings out) so the rules can be
    tested without spawning the CLI.
    """
    errors = []

    # Every one of these is a magnitude — zero is as meaningless as negative.
    for flag, value in (
        ("--weekly-mileage", weekly_mileage),
        ("--avg-pace", avg_pace),
        ("--long-run", long_run),
        ("--runs-per-week", runs_per_week),
        ("--race-distance", race_distance),
    ):
        # nan/inf slip past `<= 0` (nan compares False against everything) and
        # would poison the forward pass silently, so screen them first.
        if not math.isfinite(value):
            errors.append(f"{flag} must be a finite number (got {value})")
        elif value <= 0:
            errors.append(f"{flag} must be greater than 0 (got {value})")

    # Zero is fine here and nowhere else: the long run could have been today.
    if not math.isfinite(days_since_long_run):
        errors.append(
            f"--days-since-long-run must be a finite number (got {days_since_long_run})"
        )
    elif days_since_long_run < 0:
        errors.append(
            f"--days-since-long-run cannot be negative (got {days_since_long_run})"
        )

    return errors


def check_training_range(
    weekly_mileage: float,
    avg_pace: float,
    long_run: float,
    days_since_long_run: float,
    runs_per_week: float,
    race_distance: float,
) -> list[str]:
    """
    Return one message per input that is valid but outside the training data.

    Deliberately separate from validate_inputs(): these values describe real
    runners, they're just runners the model never trained on.  A 130 km/week
    athlete deserves a number plus a caveat, not a refusal — the caller stays
    in charge of what to do with the answer.  Blocking them would make the CLI
    useless for exactly the people most curious about it.

    Empty list means the inputs sit inside the distribution the model learned.
    """
    warnings = []

    for flag, value in (
        ("--weekly-mileage", weekly_mileage),
        ("--avg-pace", avg_pace),
        ("--long-run", long_run),
        ("--days-since-long-run", days_since_long_run),
        ("--runs-per-week", runs_per_week),
    ):
        low, high = TRAINING_RANGES[flag]
        if value < low or value > high:
            warnings.append(
                f"{flag}={value:g} is outside the training range {low:g}-{high:g}"
            )

    if race_distance not in TRAINING_RACE_DISTANCES:
        seen = ", ".join(f"{d:g}" for d in TRAINING_RACE_DISTANCES)
        warnings.append(
            f"--race-distance={race_distance:g} km was never in the training data "
            f"(trained on {seen} km only)"
        )

    return warnings


def format_hms(minutes: float) -> str:
    """Convert minutes (e.g. 245.3) to H:MM:SS (e.g. 4:05:18)."""
    total_seconds = int(round(minutes * 60))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def format_pace(min_per_km: float) -> str:
    """Convert pace in min/km (e.g. 5.82) to M:SS per km (e.g. 5:49)."""
    total_seconds = int(round(min_per_km * 60))
    m, s = divmod(total_seconds, 60)
    return f"{m}:{s:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a race finish time from recent training stats."
    )
    parser.add_argument(
        "--weekly-mileage", type=float, required=True,
        help="average weekly mileage in km",
    )
    parser.add_argument(
        "--avg-pace", type=float, required=True,
        help="easy-run pace in min per km (e.g. 5.5)",
    )
    parser.add_argument(
        "--long-run", type=float, required=True,
        help="longest recent run in km",
    )
    parser.add_argument(
        "--days-since-long-run", type=float, default=7,
        help="days since that long run (default: 7)",
    )
    parser.add_argument(
        "--runs-per-week", type=float, default=4,
        help="training runs per week (default: 4)",
    )
    parser.add_argument(
        "--race-distance", type=float, required=True,
        help="race distance in km (common: 5, 10, 21.1, 42.2)",
    )
    args = parser.parse_args()

    # argparse only guarantees "this parses as a float", not "this describes a
    # runner".  parser.error() prints usage plus the message to stderr and
    # exits 2, matching how argparse reports a bad flag.
    errors = validate_inputs(
        args.weekly_mileage,
        args.avg_pace,
        args.long_run,
        args.days_since_long_run,
        args.runs_per_week,
        args.race_distance,
    )
    if errors:
        parser.error("; ".join(errors))

    return args


def main() -> None:
    # Validation runs first, before the checkpoint is touched: bad input should
    # cost the user an error message, not a model load.
    args = parse_args()

    # Warnings go to stderr so `predict.py ... > times.txt` keeps the caveat
    # visible instead of burying it in the captured output.
    range_warnings = check_training_range(
        args.weekly_mileage,
        args.avg_pace,
        args.long_run,
        args.days_since_long_run,
        args.runs_per_week,
        args.race_distance,
    )
    for message in range_warnings:
        print(f"warning: {message}", file=sys.stderr)
    if range_warnings:
        print(
            "warning: prediction is an extrapolation beyond the training data "
            "and is unreliable",
            file=sys.stderr,
        )

    cfg = FinishPredictorConfig()

    # Rebuild the training scaler (deterministic — see inference.rebuild_scaler)
    # and load the trained weights from the checkpoint.
    scaler = rebuild_scaler(cfg)
    model = load_finish_model(cfg)

    # Keys must match FEATURE_COLS in data.py — same names, same order matters
    # inside predict_finish_time().
    features = {
        "weekly_mileage_km": args.weekly_mileage,
        "avg_pace_min_per_km": args.avg_pace,
        "long_run_km": args.long_run,
        "days_since_long_run": args.days_since_long_run,
        "runs_per_week": args.runs_per_week,
        "race_distance_km": args.race_distance,
    }

    minutes = predict_finish_time(model, scaler, features)
    race_pace = minutes / args.race_distance

    print(f"Race distance      : {args.race_distance} km")
    print(f"Predicted time     : {minutes:.1f} min  ({format_hms(minutes)})")
    print(f"Implied race pace  : {format_pace(race_pace)} /km")


if __name__ == "__main__":
    main()
