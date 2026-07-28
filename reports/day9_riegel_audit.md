# Day 9 — Auditing the Riegel baseline

Day 5 reported the MLP at 4.12 min MAE against Riegel's 28.36 min and called it
an 85.5% improvement. The explanation given was that Riegel "uses easy training
pace as the reference time proxy, which overshoots race pace."

That is directionally right and mechanically wrong. Reproduce with
`python src/scratch/riegel_audit.py` (5000 rows, seed 42).

## 1. Riegel is a strict subset of the data generator

`riegel_predict()` computes `t1 * (D/10)**1.06` where `t1 = avg_pace * 10`.
Pull the constant through:

```
10 * (D/10)**1.06  ==  D * (D/10)**0.06
```

so the formula reduces to `avg_pace * D * (D/10)**0.06`.

Now `make_sample_data()`:

```python
base = avg_pace * race_distances
distance_penalty = (race_distances / 10.0) ** 0.06
```

`base * distance_penalty` is the same expression. Measured maximum absolute
difference across 5000 rows: **6.10e-05**, which is float32 rounding.

Riegel is not a weak model of this data. It reproduces two of the five factors
exactly.

## 2. The error is entirely the three omitted factors

The generator continues:

```python
volume_bonus      = exp(-0.008 * (weekly_mileage - 50))
long_run_bonus    = exp(-0.015 * (long_run - 15))
freshness_penalty = 1.0 + 0.005 * days_since_long
```

Product over the sampled ranges: **mean 0.8345**, range 0.435 to 1.452.

The mean is the important number. It is not centred on 1.0, so this is
systematic bias rather than noise:

| Metric | Value |
|---|---|
| Riegel MAE (overall) | 27.56 min |
| mean(true / riegel) | 0.8364 |
| mean signed error | +20.70 min (Riegel too slow) |

## 3. The bias is multiplicative

| Distance | Riegel MAE |
|---|---|
| 5.0 km | 6.55 min |
| 10.0 km | 13.72 min |
| 21.1 km | 30.42 min |
| 42.2 km | 61.82 min |

Error scales with race duration, which is what a ~20% multiplicative bias
predicts. A constant offset would look flat across distances.

## 4. Why the comparison flatters the MLP

1. **Riegel is used off-label.** It predicts a race time from *another race
   time*. Feeding it `easy_pace * 10 km` supplies an input it was never designed
   to take — that is not a 10K race time.
2. **The target is a closed-form function of exactly the six input features**,
   plus `N(0, 2)` noise. The MLP is fitting a formula it has every input for.
3. **The noise floor is 1.596 min** (`E|N(0,2)| = 2*sqrt(2/pi)`). The MLP's
   4.12 min is 2.6x above it, so it has not converged nearly as well as "85.5%
   better" implies.

The headline mostly measures *"the generator has fitness terms and Riegel
doesn't."*

## 5. What would make it a fair comparison

- Add a `recent_10k_time_min` feature and give Riegel a real reference time.
- Or fit a single scalar correction to Riegel on the training split, removing
  the 0.834 bias — that is the honest dumb baseline.
- Or report against predicting the training-set mean, the standard floor.
- Ultimately: test on real data, where the target is not a formula the MLP can
  memorise.

## Takeaway

A baseline is only meaningful if it is given a fair shot at the problem. This
one was constructed from a subset of the generating function and then handed an
input it was not designed for. Both numbers currently say more about
`make_sample_data()` than about either model.
