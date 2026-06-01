# Running science background

Domain notes gathered before modelling, so the features are chosen for reasons
rather than convenience.

## Riegel's formula

    T2 = T1 * (D2 / D1) ^ 1.06

Given a known race time `T1` over distance `D1`, predict time `T2` over `D2`.
The 1.06 exponent is empirical, fitted by Pete Riegel across a large set of race
results. It encodes that longer races are harder per kilometre — if pace held
constant the exponent would be 1.0.

**The input requirement is strict and easy to violate.** `T1` must be an actual
race performance. Substituting easy training pace gives a reference that is far
too slow, and the error propagates through the exponent. Any implementation that
takes training pace as `T1` is not really running Riegel; it is running a
distorted version that will overpredict.

Worth remembering when it is used as a baseline: a baseline given the wrong
input is not measuring what you think it is.

## Where the exponent breaks down

- Below ~1500 m, anaerobic contribution dominates and the power-law
  underestimates.
- Beyond the marathon, fuelling and gut tolerance matter more than aerobic
  capacity, so the true exponent rises above 1.06 for ultras.
- For undertrained runners the marathon specifically is worse than 1.06
  predicts, because glycogen depletion is a cliff rather than a curve.

## Fade

Pace almost never holds flat. The realistic shape is a modest positive split,
with the back half slower than the front. Magnitude depends on:

- **Endurance base** relative to race distance. A runner whose longest run is
  20 km fades badly at 42 km and barely at 10 km.
- **Starting discipline.** Going out fast makes fade worse, non-linearly.
- **Fuelling**, which mainly matters past ~90 minutes.

This is the argument for a sequence model rather than a single number: fade is
a trajectory, and segment N depends on how hard segments 1..N-1 were run.

## Training-load factors worth modelling

| Factor | Effect | Notes |
|---|---|---|
| Weekly volume | Strongest single predictor of aerobic fitness | Diminishing returns at high mileage |
| Longest recent run | Endurance proxy; dominates marathon outcomes | More predictive than volume for long races |
| Run frequency | Consistency proxy | Correlated with volume, so partly redundant |
| Recency of long run | Fitness decays | Weak effect over a few weeks |

Deliberately excluded: heart rate and elevation. Both are meaningful, but HR is
device-dependent and drifts with heat and fatigue, and elevation needs a course
profile the model will not have at prediction time.

## Implication for the feature set

Six features: weekly mileage, easy pace, longest run, days since long run, runs
per week, race distance. Enough to carry real signal, small enough that a
laptop-sized MLP has a chance of fitting them.
