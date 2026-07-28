# Contributing

PaceBrain is a learning project with a day-by-day journal in `PROGRESS.md`.
That shapes most of the conventions below: the history is meant to be read,
not just merged.

## Setup

Python 3.9 or newer (CI tests 3.9, 3.11 and 3.12).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Install the hooks — they run the same checks CI does, minutes earlier:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## The checks

```bash
pytest                # test suite
ruff check .          # lint
mypy                  # type-check src/pacebrain
pytest --cov          # coverage (CI enforces a floor)
```

All four are blocking in CI. `pre-commit run --all-files` runs the lint and
hygiene hooks against everything without committing.

## Where code goes

| Directory | What belongs there | Held to |
|---|---|---|
| `src/pacebrain/` | the package: models, data, training, inference | tests, ruff, mypy |
| `src/scratch/` | day-by-day exploration, one-off experiments | ruff only |
| `tests/` | unit tests mirroring `src/pacebrain/` | ruff only |
| `reports/` | write-ups and plots | — |
| `app/` | Streamlit demo | ruff only |

`src/scratch/` is deliberately exempt from type-checking — it exists to be
quick and throwaway. If something there becomes load-bearing, move it into
the package and it inherits the package's standards.

Several package modules double as runnable scripts
(`python src/pacebrain/eval.py`), which is why they bootstrap `sys.path` and
carry ruff per-file ignores for it. Keep that working; the journal workflow
depends on it.

## Changes that touch a model or a metric

This repo has retired one headline claim already (see
`reports/day9_riegel_audit.md`), so:

- **Measure, don't assume.** If a change is supposed to improve something,
  show the before and after.
- **Use more than one seed.** A single seed is not a measurement. An apparent
  10% improvement from a learning-rate schedule turned out to be a coin flip
  across four seeds — the write-up is in `reports/lr_schedule_experiment.md`
  and the feature shipped disabled.
- **Read results against the noise floor**, not against zero. Both datasets
  have one (1.596 min for finish times, 0.08 min/km for pacing), and a number
  quoted without it overstates the result.
- **Negative results are worth committing.** A documented "this did not work,
  here is the measurement" saves the next person from trying it.

## Pull requests

- **One concern per PR**, and make it a real one. Splitting work to inflate a
  count is explicitly not wanted here.
- **Say why, not just what.** The diff shows what changed; the description is
  for the reasoning, the alternatives rejected, and anything surprising.
- **State what you verified and what you didn't.** "I couldn't build this"
  is a useful sentence. An unverified claim that turns out wrong costs more
  than the gap it papered over.
- CI must be green. Coverage must not drop below the floor.

## Commit messages

Imperative subject with a conventional prefix (`feat:`, `fix:`, `test:`,
`docs:`, `chore:`, `ci:`), then a body explaining the reasoning.

The bodies here run long on purpose — they carry the reasoning that would
otherwise be lost. Prefer explaining one decision well over listing every
file touched.

Messages end at the body: **no trailer block.**

## Updating the journal

`PROGRESS.md` gets an entry per day of substantive work, following the
existing shape: what was built, results, concept learned, and one thing to
review. Those "one thing to review" lines are the repo's backlog — several
have since become their own PRs.

Since every PR would otherwise append to the same spot, journal entries are
often batched into a follow-up rather than included in each feature PR.
