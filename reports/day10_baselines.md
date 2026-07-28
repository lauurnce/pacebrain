# Day 10 — An honest baseline set

Day 9 retired the claim that the MLP was "85.5% better than baseline". It did
not replace it. This is the replacement.

## Why one baseline was not enough

The Day 5 comparison used a single reference point, and Day 9 showed it was
handicapped: `riegel_predict()` is algebraically the first two factors of
`make_sample_data()`, and it was then asked to predict a race time from easy
training pace — an input Riegel was never designed for. Its 27.56 min error was
mostly that handicap.

Beating a handicapped baseline by 85% says nothing. So rather than fix the one
baseline, this adds three more, ordered by how much each is allowed to see:

| Baseline | Sees | Answers |
|---|---|---|
| Mean of train | nothing | did the model use its features at all? |
| Riegel, raw | 2 features, mis-specified | the Day 5 reference, kept for continuity |
| Riegel x fitted constant | 2 features, calibrated | is the error bias, or shape? |
| Linear regression | all 6 features | is the MLP's extra capacity earning its place? |

Every one is fitted on the training split and scored on validation, using the
same seeded permutation the MLP is scored on. The mean baseline uses the *train*
mean deliberately — a baseline that peeks at the split it is scored on is not a
baseline.

## Results

Validation set, n=200.

```
  predictor                 MAE (min)  x noise floor
  ------------------------------------------------
  mean of train                 67.97           42.6
  Riegel (raw)                  28.36           17.8
  Riegel x 0.8135               21.82           13.7
  linear regression             16.89           10.6
  MLP                            4.12            2.6
  ------------------------------------------------
  noise floor                    1.60            1.0
```

**The MLP is 75.6% better than the strongest baseline.** That is the honest
headline, and it is a good deal less flattering than 85.5% — while being worth
considerably more, because linear regression is a real contender rather than a
mis-specified formula.

## What each row settles

**Bias correction recovers 6.5 min.** The fitted scale is 0.8135, and Day 9
predicted 0.8345 from first principles (the mean product of the three fitness
factors Riegel cannot see). Those agreeing to within 0.02 is a good check on
both: the residual gap is the difference between a median-of-ratios fit and a
mean-of-products, plus noise. Confirms the error is multiplicative bias, exactly
as Day 9 argued — one constant removes nearly a quarter of it without the
formula learning anything new.

**Linear regression at 16.89 min answers Day 9's open question.** The worry was
that an MLP might not beat a linear model on data this well behaved, in which
case the extra capacity would be decoration. It beats it by 75.6%. The reason is
structural: the generator is a *product* of five terms, and no additive model can
represent `base * distance_penalty * volume_bonus * long_run_bonus *
freshness_penalty`. The nonlinearity is real, and the MLP is capturing it.

**The mean baseline at 67.97 min** is the sanity floor. Nothing here is close to
failing it, which is the expected result and worth exactly one line.

## Reading against the floor, not against zero

Every figure above carries a second column, and it matters more than the first.
The target has `N(0, 2)` noise added, so `E|N(0,2)| = 2*sqrt(2/pi) = 1.596 min`
is the best MAE any predictor can achieve — a perfect model scores 1.60, not 0.

The MLP sits 2.6x above that floor. So there is still signal it is not
extracting, and "75.6% better than linear" should not be read as "nearly
solved". The gap between 4.12 and 1.60 is the honest measure of remaining
headroom.

## One thing to review

**A log-space linear baseline.** The generator is multiplicative, so taking logs
turns it into a *sum* — `log(finish_time)` is very nearly linear in
`log(pace) + log(distance) + weekly_mileage + long_run + days_since_long`. A
linear fit on log-transformed features should therefore close most of the gap to
the MLP, and if it does, the interesting claim changes from "the MLP beats
linear" to "the MLP rediscovered a log-linear relationship that one line of
feature engineering would have handed it". That is the strongest honest baseline
available here and the natural Day 11.
