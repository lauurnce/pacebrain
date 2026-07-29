# PaceBrain

ML models for predicting runner race finish times and pacing strategy.
Built with PyTorch over 11 days as a learning project tied to PacePack.

## What it does

1. **Finish-time predictor** — tabular MLP that takes recent training data and outputs predicted race finish time.
2. **Pacing model** — LSTM that predicts per-segment pace across a race and flags fade.

Models are small enough to train on a laptop CPU.

## Project structure

```
src/pacebrain/     main package (models, data pipeline, training, eval)
src/scratch/       day-by-day learning scripts
notebooks/         exploratory notebooks
data/              raw data (gitignored — drop your Strava export here)
models/            saved checkpoints (.pt files, gitignored)
reports/           plots and writeups
tests/             unit tests
app/               Streamlit demo app
```

## Setup

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To work on the project itself (tests and linting), install the dev extras instead:

```bash
pip install -e ".[dev]"
pytest          # full suite
ruff check .
mypy            # type-checks src/pacebrain
pytest --cov    # coverage report; CI enforces a 97% floor
```

Optionally install the git hooks, which run the same lint on commit and the
test suite on push — the same checks CI runs, several minutes earlier:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
pre-commit run --all-files    # check everything without committing
```

## Usage

### Train

Trains the finish-time predictor, saves the checkpoint to `models/finish_predictor.pt` and the loss curve to `reports/day4_loss_curve.png`.

```bash
python src/pacebrain/train_finish.py
```

### Evaluate

Compares the MLP against the Riegel baseline on the validation set and saves a scatter plot to `reports/day5_scatter.png`.

```bash
python src/pacebrain/eval.py
```

### Using your own data

`data/` is gitignored — drop a CSV there with the six feature columns plus
`finish_time_min`. Extra columns are ignored, so a raw export can be passed
through unchanged.

```python
from pacebrain.data import load_running_csv, make_datasets

df = load_running_csv("data/activities.csv")
train_ds, val_ds, scaler = make_datasets(df)
```

The loader validates the schema up front and names any missing column, rather
than failing later inside dataset construction. Rows with missing or
non-numeric values are dropped by default (`dropna=False` to keep them).

### Predict

Predicts a race finish time from recent training stats.

```bash
python src/pacebrain/predict.py --weekly-mileage 60 --avg-pace 5.5 --long-run 28 --race-distance 42.2
```

Required flags: `--weekly-mileage` (km per week), `--avg-pace` (easy-run pace in min/km), `--long-run` (longest recent run in km), `--race-distance` (km: 5, 10, 21.1, or 42.2).
Optional flags: `--days-since-long-run` (default 7), `--runs-per-week` (default 4).
Output: predicted finish time in minutes, in H:MM:SS, and the implied race pace.

### Demo app

The same prediction as `predict.py`, with sliders instead of flags.

```bash
streamlit run app/app.py
```

Run it from the repo root, and train a model first — the app needs `models/finish_predictor.pt`, which is gitignored and not shipped with the repo. Without it the app shows a message telling you to run `python src/pacebrain/train_finish.py` rather than failing.

Sliders go wider than the ranges the model was trained on; the app warns you when an input falls outside them, because predictions there are extrapolation.

### Demo app in Docker

```bash
docker build -t pacebrain .
docker run --rm -p 8501:8501 pacebrain
```

Then open <http://localhost:8501>.

The image trains a model during the build, so it works straight away rather
than starting up and telling you to train one. Training is seeded and takes
under a minute; it happens after the dependency layer, so editing source does
not re-download torch.

To use a checkpoint you trained yourself, mount over it:

```bash
docker run --rm -p 8501:8501 -v "$PWD/models:/app/models" pacebrain
```

The image installs the CPU-only torch wheel. The default PyPI build bundles
the CUDA runtime and is about 2 GB larger, which buys nothing here — the app
runs a 2,561-parameter MLP on one row at a time.

## Running the scratch scripts

```bash
# Tensor basics
python src/scratch/tensors.py

# Manual gradient descent (fits y = 2x + 1)
python src/scratch/manual_grad_descent.py
```

## Results

Validation set, 200 rows. Every baseline is fitted on the training split only.

| Predictor | Features seen | MAE (min) | x noise floor |
|---|---|---|---|
| Mean of train | none | 67.97 | 42.6 |
| Riegel (raw) | 2, mis-specified | 28.36 | 17.8 |
| Riegel x 0.8135 | 2, calibrated | 21.82 | 13.7 |
| Linear regression | all 6 | 16.89 | 10.6 |
| Finish-time MLP | all 6 | 4.12 | 2.6 |
| **Log-linear regression** | all 6, logged | **1.98** | **1.24** |
| *Noise floor* | — | *1.60* | *1.0* |

**The MLP is not the best model here.** A linear regression on log-transformed
features beats it by 51.9%, and lands within 24% of the noise floor against the
network's 2.6x.

That is not a tuning accident. `make_sample_data()` builds its target as a
product of five factors, so taking logs turns it into a sum — and the fitted
coefficients come back as the generator's own constants: 1.0556 against a true
1.06 on log-distance, -0.007863 against -0.008 on weekly mileage, -0.014568
against -0.015 on long run. `runs_per_week`, the one feature the generator never
uses, comes back at 0.0017. The fit did not approximate the target; it
recovered it.

So the honest reading of this table is that the MLP spent 189 epochs learning an
approximation to a relationship one `np.log` hands over exactly. Capacity
substitutes for knowledge about the problem, and it substitutes badly. The
network earns its place only where the right transform is *not* known in
advance — which is the real case, and precisely the case a closed-form synthetic
target cannot test.

Worth keeping in view: every revision of this table has made the baseline
stronger and the model's advantage smaller (85.5% at Day 5, 75.6% at Day 10,
now negative). Day 5's number measured `riegel_predict()`'s handicap — it is
algebraically the first two factors of the generator, then handed an input it
was never designed for.

Reproduce with `python src/scratch/day11_log_linear.py`. Full write-ups in
`reports/day11_log_linear.md`, `reports/day10_baselines.md` and
`reports/day9_riegel_audit.md`.

## Progress log

See `PROGRESS.md`.

## Model card

`MODEL_CARD.md` covers what these models are, how they were trained, and what
they must not be used for. The headline: **both are trained entirely on
synthetic data and have never seen a real run**, so the accuracy above
describes how well a network recovers a known equation, not how well it
predicts a person.

## Contributing

See `CONTRIBUTING.md` for setup, the four blocking checks, and the conventions
around measuring changes (more than one seed; read results against the noise
floor; negative results are worth committing).
