# Day 11 — One transform beats the network

Day 10 ended on a prediction:

> The generator is multiplicative, so logs turn it into a sum and a linear fit
> on log-transformed features should close most of the gap to the MLP. If it
> does, the honest claim weakens from "the MLP beats linear" to "the MLP
> rediscovered a log-linear relationship that one line of feature engineering
> would have handed it".

It does not close most of the gap. It closes all of it and keeps going.

## The setup

`make_sample_data()` builds the target as a product:

```
finish = (pace * distance)
         * (distance / 10) ** 0.06
         * exp(-0.008 * (mileage - 50))
         * exp(-0.015 * (long_run - 15))
         * (1 + 0.005 * days_since_long)
         + N(0, 2)
```

Take logs of the product and every factor becomes an additive term:

```
log finish = log pace + 1.06 * log distance
             - 0.008 * mileage
             - 0.015 * long_run
             + log1p(0.005 * days_since_long)
             + const
```

So the transform is not a guess. Pace and distance enter logged because they
are multiplicative; mileage and long_run enter raw because they already sit
inside an `exp`. That is the entire feature engineering — six columns, chosen
to match a structure that was known in advance.

The one term that is not exactly linear is freshness: `log1p(0.005 * d)` is
concave in `d`. Over the generator's 3–21 day range it tracks `0.005 * d` to
within 0.005, so a raw column absorbs it and the residual curvature disappears
under the noise.

## Results

Validation set, n=200, same seeded split the MLP is scored on.

```
  predictor                 MAE (min)  x noise floor
  ------------------------------------------------
  mean of train                 67.97          42.60
  Riegel (raw)                  28.36          17.77
  Riegel x 0.8135               21.82          13.67
  linear regression             16.89          10.59
  MLP                            4.12           2.58
  log-linear                     1.98           1.24
  ------------------------------------------------
  noise floor                    1.60           1.00
```

The log-linear baseline beats the MLP by **51.9%**, and sits **1.24x** above
the noise floor against the network's 2.58x. Of the 2.52 min of headroom the
MLP left on the table, the transform recovers 2.14 of it.

## Why this is recovery, not fitting

An MAE that good could be luck, or a subtle leak. The coefficients settle it.
They are the generator's own constants:

| term | fitted | generator |
|---|---|---|
| log(avg_pace) | +0.958945 | +1.0 |
| log(race_distance) | +1.055595 | +1.06 |
| weekly_mileage | -0.007863 | -0.008 |
| long_run | -0.014568 | -0.015 |
| days_since_long_run | +0.005083 | +0.005 |
| runs_per_week | +0.001746 | 0 (unused) |

Four land within 3% of a constant chosen before the fit ran. The fifth,
`runs_per_week`, is the one the generator never uses — and the fit drives it to
0.0017, correctly finding it irrelevant rather than inventing a use for it.

The pace exponent is the loosest at 0.959 against a true 1.0, which is what the
additive noise term does under a log transform: `log(y + e)` is not
`log(y) + log(e)`, so the noise becomes heteroscedastic in log space and biases
the fit slightly toward the mean. That is a real limitation of the approach and
worth naming rather than rounding away.

## What this costs the project's headline

The README claimed the MLP beats the strongest honest baseline by 75.6%. That
was true of the baselines Day 10 had. With this one, the MLP does not beat the
strongest baseline at all — it loses to it by a factor of two, and the whole
comparison inverts:

- **Day 5:** MLP beats baseline by 85.5% (measuring the baseline's handicap)
- **Day 10:** MLP beats baseline by 75.6% (measuring against a fair but
  mis-specified linear model)
- **Day 11:** the strongest baseline beats the MLP by 51.9%

Each revision made the baseline stronger and the MLP's advantage smaller. That
is the direction honest baselining goes.

## The lesson

The network spent 189 epochs of gradient descent discovering an approximation
to a relationship that one `np.log` hands over exactly, in closed form, with no
hyperparameters, no seed, and no training run. Its 4.12 min is the cost of
learning the shape of the problem from scratch instead of being told it.

The generalisation is the part worth keeping: **model capacity substitutes for
knowledge about the problem, and it substitutes badly.** An MLP is universal in
the sense that it can approximate this target, not in the sense that it should
have to. Where the structure is known — and multiplicative structure in
physical quantities is very often known — encoding it beats learning it, on
accuracy and determinism and cost simultaneously.

The honest framing of this repo's result is therefore not "the MLP works". It
is: this target is a closed-form product of six features, that is close to the
friendliest learning problem there is, and on it a linear model with the right
transform is the correct tool. The MLP earns its place only where the
transform is *not* known in advance — which is the real case, and not the case
this synthetic generator tests.

## What this does not show

The log-linear model was handed the generator's structure by someone who had
read the generator. On real Strava data nobody knows the true functional form,
the noise is not Gaussian, features are correlated and partly missing, and the
"right" transform is exactly the thing in question. The MLP's advantage is
supposed to be that it does not need that knowledge. Nothing here tests that,
because the synthetic target was never a test of it.

## Reproduce

```bash
python src/scratch/day11_log_linear.py
```

## Open question for Day 12

Break the assumption the transform depends on. Add an interaction the log form
cannot represent — a term where volume matters more at marathon distance than
at 5 km, say — and rerun both. If the MLP overtakes the log-linear model there,
that is the first evidence in this project that the capacity is worth its cost,
and it would be measured rather than assumed.
