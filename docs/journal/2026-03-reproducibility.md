# Study notes — reproducibility and experiment tracking

Being unable to reproduce your own result from last week is a common and
entirely avoidable failure.

## Seeds

Random initialisation, shuffling, dropout masks and data splits all draw from
RNGs. Fix and record every seed, or two runs differ for reasons unrelated to
whatever was changed.

The corollary matters more than the practice: if re-running with a different
seed moves the metric as much as the change being tested, the change did
nothing measurable. Seed variation is the natural noise floor for any
comparison, and it is worth measuring once so later differences can be judged
against it.

Full determinism on GPU needs more than a seed — some kernels are
nondeterministic by default. On CPU at this scale, seeds are enough.

## What to record per run

- The seed
- Every hyperparameter, not just the changed one
- The data version or generator parameters
- The code version, ideally a commit hash
- The resulting metrics

A hyperparameter absent from the record is a hyperparameter you will
misremember.

## Tooling

For a project this size a text log or a CSV row per run is sufficient. Tracking
frameworks earn their keep when runs are long and numerous; before that they
are setup that produces no insight.

The discipline is what matters, not the tool.

## Config in one place

Scattered magic numbers make a run impossible to describe. A single config
object means the full specification of a run is one printable thing, and
comparing two runs is comparing two objects rather than reading two files
line by line.

## Environment

Pin dependency versions. A library upgrade changing a default can move results,
and diagnosing that without a version record is miserable.

## The real payoff

Not scientific rigour for its own sake — it is the ability to answer "did that
change actually help?" A pile of untracked runs makes that question
unanswerable, and answering it is the entire point of running experiments.
