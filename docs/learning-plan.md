# PaceBrain — learning plan

Goal: learn PyTorch properly by building something I would actually use, rather
than working through disconnected tutorials. Tying it to PacePack gives every
model a real consumer, so "does this work" has a concrete answer.

## Why a running model

I already have the domain knowledge. That matters more than it sounds: when a
model outputs a 2:05 marathon for a 60 km/week runner I will immediately know
it is wrong, without needing a metric to tell me. Debugging a model in a domain
you do not understand is guesswork.

## Scope

Two models, deliberately small enough to train on a laptop CPU:

1. **Finish-time predictor** — tabular MLP. Recent training stats in, predicted
   race time out. This is the "hello world" of supervised regression and the
   right place to learn the training loop.
2. **Pacing model** — sequence model over race segments. Predicts per-segment
   pace and flags fade. This is where an LSTM earns its place, because the
   whole point is that segment N depends on segments 1..N-1.

Deliberately out of scope: transformers, GPU training, distributed anything,
hyperparameter search frameworks. Those are optimisations on top of
fundamentals I do not have yet.

## Rough day plan

| Day | Focus |
|---|---|
| 1 | Tensors, autograd, manual gradient descent |
| 2 | First MLP on toy data |
| 3 | Real data pipeline — Dataset, normalization, splits |
| 4 | Training loop, validation, checkpointing |
| 5 | Evaluation and a non-ML baseline |
| 6 | Refactor into a package, inference CLI |
| 7 | Sequence model intro (LSTM) |
| 8-10 | Consolidation, demo app, writeup |

## Ground rules

- **Synthetic data first.** A real Strava export has missing fields, weird
  units and no labels for races I did not run. Starting synthetic means the
  signal is known, so when the model fails I know it is the model.
- **Always have a baseline.** If the network cannot beat a formula, it has
  learned nothing. Riegel is the obvious candidate.
- **Write down what I learned each day**, including what I did not understand.
  The "one thing to review" note is the most useful part — it is the honest
  record of where the gaps are.
- **Small commits.** Each one should be explainable in a sentence.
