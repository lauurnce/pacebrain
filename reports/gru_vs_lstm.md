# GRU vs LSTM for the pacing model

Day 7 closed with "review LSTM vs GRU trade-offs". The received wisdom is that
a GRU has fewer parameters and "often reaches similar accuracy" — a claim
`seq_models.py` had been repeating in a docstring without anyone checking it.
This checks it.

## Setup

`PacingConfig` defaults — 600 synthetic races, 10 segments each, hidden size
64, Adam at 1e-3, 200 epoch budget, early stopping at `patience=20`. Metric is
best validation RMSE in min/km per segment, which is what the training loop
already reports. Four seeds, because one seed is not a measurement (see
`reports/lr_schedule_experiment.md` for what happens when you forget that).

## Parameters

| | parameters | vs LSTM |
|---|---|---|
| `PacingLSTM` | 18,497 | — |
| `PacingGRU` | 13,889 | **−24.9%** |

An LSTM cell carries a hidden state *and* a cell state with four gates. A GRU
merges the two states and uses three gates, so at equal hidden size it needs
roughly three quarters of the weights.

## Results

| seed | LSTM RMSE | GRU RMSE | GRU better? |
|---|---|---|---|
| 42 | 0.0917 | **0.0898** | yes |
| 7 | 0.0944 | **0.0921** | yes |
| 123 | 0.0965 | **0.0889** | yes |
| 2024 | 0.0980 | **0.0962** | yes |
| **mean** | **0.0951** | **0.0917** | **4 / 4** |

The GRU wins on every seed, by 3.57% mean RMSE, with 25% fewer parameters and
about 10% less wall-clock time per run (3.7s vs 4.1s).

Unlike the LR-schedule experiment, this one replicates. Four out of four is
not proof, but it is a consistent direction rather than an average rescued by
one lucky run.

## The 3.57% understates it

Both numbers sit close to the noise floor, and that compresses the gap.
`make_sample_sequences()` adds `N(0, 0.08)` to every segment, so RMSE cannot
go below **0.08** however good the model is.

Errors add in quadrature, so the model's own contribution is
`sqrt(RMSE_total² − 0.08²)`:

| | total RMSE | model-only RMSE |
|---|---|---|
| LSTM | 0.0951 | 0.0514 |
| GRU | 0.0917 | **0.0448** |

On the part that is actually learnable, the GRU reduces error by **12.8%**,
not 3.6%. Quoting the raw 3.57% would make the difference look like rounding
when it is nearly four times that on the signal.

## Neither model stopped early

Both ran the full 200 epochs on all four seeds — early stopping never fired.
So neither had plateaued, and both would probably improve with a longer
budget. The comparison is fair (identical budget, identical data, identical
seeds) but it is a comparison at 200 epochs, not at convergence. Worth
re-running with a larger budget before treating the gap as settled.

## Decision

The default stays `cell="lstm"`. The GRU is better here, but the LSTM is what
every existing checkpoint, the Day 7 write-up and the README describe, and a
3.57% RMSE gain on synthetic data is not worth silently invalidating all of
that. The flag makes the swap one line:

```python
cfg = PacingConfig(cell="gru")
```

The honest summary is that the docstring's "often reaches similar accuracy"
was too modest for this problem: the GRU is not merely similar, it is better,
cheaper and faster. On a target this smooth the LSTM's extra cell state
appears to buy nothing and cost parameters to fit.
