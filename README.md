# PaceBrain

ML models for predicting runner race finish times and pacing strategy.
Built with PyTorch over 10 days as a learning project tied to PacePack.

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
pytest          # 156 tests
ruff check .
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

## Running the scratch scripts

```bash
# Tensor basics
python src/scratch/tensors.py

# Manual gradient descent (fits y = 2x + 1)
python src/scratch/manual_grad_descent.py
```

## Results

| Model | MAE (min) | vs Riegel baseline |
|---|---|---|
| Finish-time MLP | 4.12 | 85.5% lower error |
| Riegel formula (baseline) | 28.36 | reference |

**Read that "85.5%" with suspicion.** The data is synthetic, and the baseline is
handicapped in a way that flatters the model. `riegel_predict()` turns out to be
algebraically identical to the first two factors of the data generator —
`10 * (D/10)**1.06` reduces to `D * (D/10)**0.06`, which is exactly
`base * distance_penalty` in `make_sample_data()`. So Riegel reproduces part of
the target exactly, and its entire error is the three fitness factors it cannot
see (`volume_bonus`, `long_run_bonus`, `freshness_penalty`). Their product
averages 0.834 rather than 1.0, so Riegel overpredicts by ~20% on every row —
systematic bias, not spread.

Two further caveats: the target is a closed-form function of exactly the six
input features, which is close to the friendliest possible learning problem;
and the noise floor is 1.60 min, so the MLP's 4.12 min is still 2.6x above the
best achievable error. Run `python src/scratch/riegel_audit.py` to reproduce
these numbers. See `reports/day9_riegel_audit.md` for the full write-up.

## Progress log

See `PROGRESS.md`.
