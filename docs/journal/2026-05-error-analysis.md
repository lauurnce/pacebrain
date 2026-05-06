# Study notes — looking at the errors

An aggregate metric says how much the model is wrong. It says nothing about
where or why, and those are the actionable questions.

## The procedure

Sort predictions by error and read the worst ones. Not summary statistics of
them — the actual rows. Twenty minutes of this reliably finds things no metric
surfaces.

## What tends to turn up

Systematic bias inside a subgroup that averages out globally. Label errors that
the model is being penalised for getting right. A feature that is missing or
mis-scaled for a particular slice. A range of the target where the model has
almost no training data.

## Slicing

Break the metric down by any grouping that exists — distance, season, source,
whatever the domain has. A model with acceptable overall error can be
unusable on one slice, and the aggregate is exactly the thing that hides it.

## The recurring lesson

Average error describes the typical case. The worst case is what determines
whether anyone trusts the thing, and no mean tells you about it.
