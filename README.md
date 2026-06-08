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
app/               Streamlit demo (Day 9)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the scratch scripts

```bash
# Tensor basics
python src/scratch/tensors.py

# Manual gradient descent (fits y = 2x + 1)
python src/scratch/manual_grad_descent.py
```

## Results

_Populated from Day 5 onward._

| Model | MAE (min) | vs Riegel baseline |
|---|---|---|
| Finish-time MLP | TBD | TBD |

## Progress log

See `PROGRESS.md`.
