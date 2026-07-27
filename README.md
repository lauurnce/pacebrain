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

Note: the data is synthetic, and the Riegel baseline is weak here because it uses easy training pace as the reference time proxy, which overshoots race pace.

## Progress log

See `PROGRESS.md`.
