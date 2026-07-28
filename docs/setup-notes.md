# Environment and dependency notes

Decisions made before writing code, so the first day is not spent on tooling.

## Python version

Targeting 3.9 as the floor, because that is what ships with macOS and the
project should run on a stock interpreter without a version manager.

The cost is real: `X | None` union syntax is 3.10+, and `from __future__ import
annotations` is needed to use it on 3.9. Easy to forget until something fails at
import time rather than at the line that used it.

## Dependencies

| Package | Why |
|---|---|
| `torch` | The point of the project |
| `numpy` | Synthetic data generation, metrics |
| `pandas` | DataFrame as the boundary format between raw data and tensors |
| `scikit-learn` | `StandardScaler` only |
| `matplotlib` | Loss curves and scatter plots |
| `streamlit` | Demo UI later on |

Deliberately not included: a config framework (a dataclass is enough), an
experiment tracker (the runs are seconds long), a plotting wrapper.

**On scikit-learn** — pulling in a whole library for `StandardScaler` is
arguably overkill when it is two lines of numpy. Keeping it because the fitted
object has a defined interface for persisting and reapplying statistics, and
getting normalization subtly wrong between training and inference is exactly
the failure this project should avoid.

## Matplotlib backend

Set `matplotlib.use("Agg")` before importing `pyplot`, in every script that
plots. Without it, saving a figure on a headless machine can fail or try to
open a window. The ordering matters — the backend must be chosen before pyplot
is imported, which means the import cannot be sorted to the top of the file
with the others.

## Repo layout

```
src/pacebrain/   the package
src/scratch/     day-by-day learning scripts, kept deliberately
notebooks/       exploration
data/            gitignored, real exports land here
models/          gitignored, checkpoints are build artefacts
reports/         plots and writeups
tests/           unit tests
app/             demo
```

`src/scratch/` stays in the repo on purpose. The throwaway scripts are the
record of what was actually learned, and deleting them would leave only the
polished version, which is not an honest account.

`data/` and `models/` are gitignored. Checkpoints are regenerable in seconds and
binary diffs are useless; real exports are personal data and do not belong in a
public repo.

## Things to set up later, not now

- CI. Worth having, but not before there are tests to run.
- Packaging metadata. The `sys.path` bootstrap in each script is ugly but keeps
  every file runnable standalone, which matters more while learning.
- Linting. Adding a linter before there is code produces noise, not signal.
