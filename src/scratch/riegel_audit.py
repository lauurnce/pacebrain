"""
Audit the Riegel baseline — Day 9.

Day 5 reported the MLP beating Riegel by 85.5% and explained the gap as "easy
training pace overshoots race pace". That explanation is incomplete. This
script works out what the baseline is actually doing.

The finding: riegel_predict() is algebraically identical to the first two
factors of make_sample_data()'s target. Pulling the constant through,

    10 * (D/10) ** 1.06  ==  D * (D/10) ** 0.06

so the whole formula reduces to avg_pace * D * (D/10)**0.06, which is exactly
`base * distance_penalty` in the generator. Riegel is not a bad model of this
data; it is a strict subset of it, missing the three fitness factors.

Run:
    python src/scratch/riegel_audit.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np

from pacebrain.data import TARGET_COL, make_sample_data
from pacebrain.eval import mae_minutes, riegel_predict

N_SAMPLES = 5000
SEED = 42


def main() -> None:
    df = make_sample_data(n_samples=N_SAMPLES, seed=SEED)
    pace = df["avg_pace_min_per_km"].values
    dist = df["race_distance_km"].values
    y = df[TARGET_COL].values

    riegel = riegel_predict(pace, dist)

    # 1. Riegel == base * distance_penalty, exactly.
    base = pace * dist
    distance_penalty = (dist / 10.0) ** 0.06
    algebraic_gap = float(np.max(np.abs(riegel - base * distance_penalty)))

    # 2. The three factors Riegel cannot see.
    volume_bonus = np.exp(-0.008 * (df["weekly_mileage_km"].values - 50))
    long_run_bonus = np.exp(-0.015 * (df["long_run_km"].values - 15))
    freshness_penalty = 1.0 + 0.005 * df["days_since_long_run"].values
    omitted = volume_bonus * long_run_bonus * freshness_penalty

    print(f"Rows: {N_SAMPLES}  seed: {SEED}\n")

    print("1. Is Riegel algebraically inside the generator?")
    print(f"   max |riegel - base*distance_penalty| = {algebraic_gap:.2e}")
    print("   (float32 rounding only, so yes)\n")

    print("2. The factors Riegel omits")
    print(f"   mean  {omitted.mean():.4f}   range {omitted.min():.3f} - {omitted.max():.3f}")
    print("   Mean is not 1.0, so the baseline carries systematic bias,")
    print("   not just spread.\n")

    print("3. Consequences")
    print(f"   Riegel MAE overall  : {mae_minutes(y, riegel):6.2f} min")
    print(f"   mean(true / riegel) : {np.mean(y / riegel):6.4f}")
    print(f"   mean signed error   : {np.mean(riegel - y):+6.2f} min (positive = too slow)\n")

    print("4. Error scales with race duration (multiplicative bias)")
    for d in (5.0, 10.0, 21.1, 42.2):
        mask = dist == d
        print(f"   {d:5.1f} km   MAE {mae_minutes(y[mask], riegel[mask]):6.2f} min")

    # The irreducible error: finish_time has N(0, 2) noise added, and the
    # expected absolute value of a N(0, s) variable is s * sqrt(2/pi).
    print(f"\n5. Noise floor: {2 * np.sqrt(2 / np.pi):.3f} min")
    print("   No model can beat this on synthetic data. The MLP's 4.12 min")
    print("   is 2.6x above it, so it has not converged as well as the")
    print("   85.5% headline suggests.")


if __name__ == "__main__":
    main()
