# Riegel power law in log space

Riegel's formula predicts a race time from another race time: `T2 = T1 * (D2 / D1) ** k`, with `k = 1.06` in `eval.riegel_predict`. Take logs and the power becomes a slope:

```
log T2 = log T1 + k * (log D2 - log D1)
```

On log-log axes the formula is a straight line whose slope is the exponent. That turns "what is k for these runners?" into a linear regression: regress log time on log distance and read the coefficient.

`baselines.fit_log_linear` is that regression with the other features added. `make_sample_data` builds finish time as `pace * distance * (distance / 10) ** 0.06 * ...`, and since

```
distance * (distance / 10) ** 0.06 = 10 * (distance / 10) ** 1.06
```

the generator's distance exponent is 1 + 0.06 = 1.06, the same number Riegel uses. That is why `log_features` puts pace and distance in as logs (they multiply) and mileage and long run in raw (they already sit inside an `exp`).

Fitted on `make_sample_data(n_samples=1000, seed=42)`, the coefficients on `log_avg_pace_min_per_km` and `log_race_distance_km` come out at 0.967 and 1.055, against the generator's 1 and 1.06. Close but not exact: the target carries N(0, 2) noise and the freshness term is only approximately linear in `log_features`, and the two sources were not separated.

Exponentiating the fitted log prediction (`log_linear_prediction`) gives the conditional median rather than the mean, which suits the MAE this baseline is scored on.
