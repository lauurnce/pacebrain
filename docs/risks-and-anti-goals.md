# Risks and anti-goals

Written before starting, so the failure modes are named while it is still easy
to be honest about them.

## Data leakage

The most likely way to get a great number that means nothing.

- **Scaler fitted on the full dataset.** Validation statistics end up baked into
  the normalisation, so the model has seen information it should not have.
  Fit on train only.
- **Splitting after any transformation that looks across rows.** Split first.
- **Sequence data split by segment rather than by race.** Segments from the same
  race in both train and validation is near-duplicate leakage and will produce
  a validation score that cannot be reproduced on anything new.

## Impressive on synthetic, useless in reality

The trap this project is most exposed to.

If the synthetic target is a closed-form function of exactly the input features
plus noise, then a sufficiently flexible model can approach the noise floor —
and that says nothing whatsoever about real data, where the relationship is
messier and the features are incomplete.

**Synthetic results validate the pipeline, not the model.** Worth writing at
the top of any results table, because it is easy to forget once the number
looks good.

## Baseline theatre

A related and subtler failure: beating a baseline that was never given a fair
chance. If the baseline is fed an input it was not designed for, or is a
degenerate special case of however the data was generated, the comparison
measures the handicap rather than the model.

Any large reported improvement deserves the question "is this baseline actually
trying?" before it is believed.

## Extrapolation

The model will be trained on a range of mileage, pace and distance. Inputs
outside that range are extrapolation, and a neural network extrapolates
confidently and badly — no error, just a wrong answer delivered with the same
certainty as a right one.

Needs an explicit range check at the inference boundary, warning rather than
refusing. A 130 km/week runner is real, just unrepresented.

## Overfitting

Small dataset, flexible model. Mitigations, in order of effort: train/val split
with early stopping, dropout, then reducing capacity. Watch the gap between
train and validation loss rather than the absolute value of either.

## Silent wrongness generally

The recurring theme in all of the above. The dangerous failures here do not
raise exceptions — a mismatched scaler, a leaked split, an extrapolated input
and a misnamed column all produce a confident number that is simply wrong.

Prefer designs that fail loudly. Validate inputs at boundaries, and when
something cannot be validated, say so in the output rather than hoping.

## Anti-goals

- Chasing a better metric on synthetic data past the point of learning
  something. Once the pipeline is proven, more epochs are procrastination.
- Adding model capacity before checking a linear model's performance.
- Building a UI before the model is worth showing.
- Optimising for training speed. The runs take seconds; the bottleneck is
  understanding.
