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

Validation set, 200 rows. Every baseline is fitted on the training split only.

| Predictor | Features seen | MAE (min) | x noise floor |
|---|---|---|---|
| Mean of train | none | 67.97 | 42.6 |
| Riegel (raw) | 2, mis-specified | 28.36 | 17.8 |
| Riegel x 0.8135 | 2, calibrated | 21.82 | 13.7 |
| Linear regression | all 6 | 16.89 | 10.6 |
| **Finish-time MLP** | all 6 | **4.12** | **2.6** |
| *Noise floor* | — | *1.60* | *1.0* |

The MLP beats the strongest honest baseline (linear regression) by 75.6%. That
is the number to quote — **not** the "85.5% vs Riegel" this table used to lead
with, which measured the baseline's handicap more than the model's skill.
`riegel_predict()` is algebraically the first two factors of the data generator
and was then handed an input it was never designed for (easy training pace, not
a previous race time). Rescaling it by a single fitted constant recovers 6.5 min
of that, confirming the error was systematic bias rather than spread.

Linear regression is the baseline that earns the MLP its place: on a target this
well behaved a linear fit is a genuine contender, and the MLP still cuts its
error by three quarters — the multiplicative structure of the generator is real
nonlinearity, not noise. Two caveats stand: the target is a closed-form function
of exactly the six input features, about the friendliest learning problem there
is; and at 4.12 min the MLP remains 2.6x above the 1.60 min noise floor, so
there is signal left on the table.

Reproduce with `python src/scratch/day10_baselines.py`. Full write-ups in
`reports/day10_baselines.md` and `reports/day9_riegel_audit.md`.

## Progress log

See `PROGRESS.md`.
