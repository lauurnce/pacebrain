# RMSE/MAE ratio for gaussian errors

`eval.run_evaluation` prints the RMSE/MAE ratio next to both metrics and calls 1.25 the gaussian value. The constant is `sqrt(pi / 2)`.

For an error `e ~ N(0, s^2)`:

```
E[e^2]  = s^2                  so RMSE = s
E[|e|]  = s * sqrt(2 / pi)     so MAE  = s * sqrt(2 / pi)
```

The second line is the integral `2 * integral_0^inf x * phi(x / s) / s dx`, which evaluates to `s * sqrt(2 / pi)`. Dividing:

```
RMSE / MAE = 1 / sqrt(2 / pi) = sqrt(pi / 2) = 1.2533
```

independent of `s`. A million simulated N(0, 2) errors give MAE 1.597, RMSE 2.001 and a ratio of 1.2533.

The same expression sets the noise floor in `baselines.py`: the target carries N(0, 2) noise, so no predictor can beat `2 * sqrt(2 / pi) = 1.596` minutes of MAE.

Reading the ratio:

- It is never below 1, because `(E|e|)^2 <= E[e^2]`, with equality only when every error has the same magnitude.
- Near 1.25: the errors look gaussian.
- Well above 1.25: a few large misses dominate RMSE while MAE stays low, a heavier tail than gaussian.
- Between 1 and 1.25: the errors are more uniform in size than gaussian ones.
