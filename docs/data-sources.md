# Data sources

What running data is actually obtainable, and what it does not contain.

## Strava API

The obvious source, with real constraints:

- **OAuth per user.** No bulk access; every runner authorises individually.
- **Rate limits.** Low enough that backfilling years of history for many users
  needs queueing rather than a loop.
- **Summary vs detailed activities.** The list endpoint gives distance, moving
  time and average speed cheaply. Splits and streams cost an extra call per
  activity, which is what the pacing model will eventually need.

## Garmin / bulk export

A full account export arrives as a zip of `.fit` files plus CSV summaries.
Better for a one-off dataset than an ongoing product, and useful precisely
because it needs no API quota while prototyping.

Formats worth knowing:

| Format | Notes |
|---|---|
| `.fit` | Binary, richest. Needs a parser library |
| `.tcx` | XML, has per-lap data, verbose |
| `.gpx` | XML, points and timestamps only. No HR or cadence |

## The label problem

This is the hard part, and it is not a tooling issue.

The target is a *race* finish time, but an export is a list of activities with
no reliable flag saying which were races. Heuristics available:

- Activity title matching, which depends on people naming things consistently.
  They do not.
- A hard effort at a standard distance (5, 10, 21.1, 42.2 km) in a single
  continuous run.
- Explicit `workout_type` where the platform records it, which is the most
  reliable and the least often set.

Even with a perfect flag, each runner contributes very few labelled rows.
Someone racing four times a year gives four labels — against months of training
input per label. That ratio is the core difficulty of the whole project.

## Cold start

A new user has no race history at all. Whatever gets built has to say something
useful from training data alone, which reinforces the decision to make training
history the input space rather than a prior race.

## Practical implication

Start synthetic. A generator with known structure means model failures are
attributable to the model rather than to the data pipeline, and the label
problem above means a real dataset large enough to train on is weeks of work by
itself. Real data comes after the pipeline is proven, not before.

`data/` stays gitignored regardless — real exports are personal data and do not
belong in a public repository.
