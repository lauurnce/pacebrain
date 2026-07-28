# Study notes — missing data

Real datasets have gaps. What you do about them is a modelling decision.

## Why it is not just a cleanup step

Missingness often carries information. A blank heart rate might mean the runner
had no strap, which correlates with casual sessions. Dropping those rows
discards a real signal and biases what remains.

Worth asking first: is this missing at random, or is the absence itself
meaningful?

## Options

**Drop rows.** Simple and correct when gaps are rare and unsystematic. With
5% missing scattered randomly, fine. With 30% concentrated in one group, you
have just deleted that group from the model.

**Drop the column.** Reasonable when most of a feature is absent. A feature
present 20% of the time cannot carry much.

**Impute with mean/median.** Fills the hole and shrinks the variance, since
every imputed value sits exactly at the centre. It also invents data, and the
model cannot tell invented from observed.

**Impute plus an indicator column.** Fill the value *and* add a boolean saying
it was missing. The model can then learn from the absence itself. Usually the
best of the simple options.

**Model-based imputation.** Predict the missing feature from the others. More
accurate, more machinery, and a second model to validate.

## The trap

Imputation statistics are learned from data, so they belong to the training
split. Computing a median over the full dataset and using it to fill training
rows leaks validation information — the same mistake as fitting a scaler on
everything, and just as quiet.

## For a numeric pipeline specifically

`NaN` propagates through arithmetic rather than raising. A single missing value
reaches the loss, makes it `NaN`, and every gradient becomes `NaN` — the model
does not error, it just stops learning while continuing to train.

Check for missing values explicitly at load time. Failing loudly at the
boundary is far cheaper than diagnosing a silently dead training run.
