"""
Predict a race finish time from recent training stats — Day 6 inference CLI.

Takes the same six features the model was trained on, scales them with the
rebuilt training scaler, and runs one forward pass through the checkpointed
FinishTimePredictor. No training happens here; this is pure inference.

Run:
    python src/pacebrain/predict.py --weekly-mileage 60 --avg-pace 5.5 \
        --long-run 28 --race-distance 42.2
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse

from pacebrain.config import FinishPredictorConfig
from pacebrain.inference import load_finish_model, predict_finish_time, rebuild_scaler


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
