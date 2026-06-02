# Feature design

The six columns in `FEATURE_COLS`, and why each earns its place.

| Feature | Unit | Range to generate | Rationale |
|---|---|---|---|
| `weekly_mileage_km` | km/week | 20-120 | Best single proxy for aerobic base |
| `avg_pace_min_per_km` | min/km | 4.5-7.5 | Speed proxy; lower is faster |
| `long_run_km` | km | 10-35 | Endurance proxy, dominates long races |
| `days_since_long_run` | days | 3-21 | Freshness/decay |
| `runs_per_week` | count | 3-7 | Consistency |
| `race_distance_km` | km | 5 / 10 / 21.1 / 42.2 | The question being asked |

Target: `finish_time_min`.

## Why min/km and not min/mile

Everything internal stays metric. Unit conversion at the boundary if ever
needed, never in the middle — mixed units inside a model is the kind of bug
that produces plausible-looking wrong answers rather than a crash.

## Why pace rather than speed

Pace (min/km) is what runners actually think in. Speed (km/h) is its
reciprocal, so the information is identical, but the model's inputs should
match the units of the thing being predicted — a time. Keeping both in
time-per-distance means the relationship to the target stays roughly linear
rather than reciprocal.

## Why race distance is a feature, not a separate model

Four distances could be four models. One model with distance as an input is
better here: the underlying physiology is shared, so a single model can learn
from all the data at once rather than splitting it four ways. It also
generalises to distances not in the training set, though the audit should check
whether that generalisation actually holds.

## Synthetic target design

The target needs to be learnable but not trivial. Plan: a physics-inspired
product of factors rather than a linear sum, so the model has to learn
interactions.

- Base time from pace times distance.
- A distance penalty reproducing the Riegel effect.
- Multiplicative bonuses for volume and long-run length.
- A small penalty for staleness.
- Gaussian noise so the model cannot reach zero error.

Using exponentials for the bonuses gives diminishing returns, which matches
reality — going from 20 to 40 km/week helps far more than 100 to 120.

**A caution to check later:** if the baseline is Riegel, and the target is built
starting from a Riegel-shaped term, then the baseline is partly inside the
generating function. That would make any comparison between them flattering to
the model. Worth auditing once there are numbers to look at.

## Known limitation

The target is a closed-form function of exactly these six inputs plus noise.
That is the friendliest possible learning problem, and success on it says
nothing about real data. Synthetic results are a check that the pipeline works,
not evidence that the model is good.
