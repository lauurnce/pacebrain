# Model card — PaceBrain

Two models ship in this repo. Both are trained **entirely on synthetic data**,
which is the single most important thing on this page and the reason it exists.

---

## The short version

> **Do not use these models to plan a real race.** They have never seen a real
> run. Every number below is measured against a formula this repo wrote, so
> the reported accuracy describes how well a network recovers a known equation
> — not how well it predicts a human being.

---

## 1. Finish-time predictor

**`FinishTimePredictor`** — `src/pacebrain/models.py`

| | |
|---|---|
| Architecture | MLP, 6 → 64 → 32 → 1, ReLU, dropout 0.1 |
| Parameters | 2,561 |
| Optimiser | Adam, lr 1e-3, MSE loss |
| Training | 1000 synthetic rows, 80/20 split, early stopping (patience 25) |
| Input | 6 features, standardised (scaler saved in the checkpoint) |
| Output | finish time in minutes |

**Inputs:** `weekly_mileage_km`, `avg_pace_min_per_km`, `long_run_km`,
`days_since_long_run`, `runs_per_week`, `race_distance_km`.

### Performance

Validation set, 200 rows. Baselines fitted on the training split only.

| Predictor | Features seen | MAE (min) | × noise floor |
|---|---|---|---|
| Mean of train | none | 67.97 | 42.6 |
| Riegel (raw) | 2, mis-specified | 28.36 | 17.8 |
| Riegel × 0.8135 | 2, calibrated | 21.82 | 13.7 |
| Linear regression | all 6 | 16.89 | 10.6 |
| **MLP** | all 6 | **4.12** | **2.6** |
| *Noise floor* | — | *1.60* | *1.0* |

MAE 4.12 / RMSE 5.52 min, ratio 1.34 — near-gaussian, so the errors are fairly
uniform rather than a few large misses.

**Quote the 75.6% over linear regression, not "85.5% over Riegel".** The
Riegel comparison measured the baseline's handicap, not the model's skill; see
`reports/day9_riegel_audit.md`.

---

## 2. Pacing model

**`PacingLSTM`** — `src/pacebrain/seq_models.py`

| | |
|---|---|
| Architecture | 1-layer LSTM, hidden 64, linear head → 1 per timestep |
| Parameters | 18,497 |
| Training | 600 synthetic races, 10 segments each, 80/20 split by race |
| Input | (batch, seq_len, 6) per-segment features |
| Output | (batch, seq_len, 1) pace in min/km per segment |

Validation RMSE ≈ **0.092 min/km** per segment, against a noise floor of
**0.08** — so it has recovered nearly all the available signal.

Variable-length races (per-km splits, 5–42 timesteps) are supported via
`pad_collate` + `pack_padded_sequence` + `masked_mse_loss`.

---

## Intended use

**In scope**

- Learning PyTorch: the repo is a day-by-day journal, and the models exist to
  be built and understood
- A worked example of honest evaluation — baseline ladders, noise floors,
  negative results
- A template to retrain on real data

**Out of scope**

- Planning, pacing or predicting any real race
- Anything where a wrong answer costs a person something
- Any claim about real runners. There is no evidence here about real runners.

---

## Training data

`make_sample_data()` in `src/pacebrain/data.py`. Synthetic, seeded,
physics-inspired:

```
finish_time = pace × distance
            × (distance/10)^0.06      # longer races are harder per km
            × exp(-0.008 × (mileage − 50))    # volume helps
            × exp(-0.015 × (long_run − 15))   # endurance helps
            × (1 + 0.005 × days_since_long)   # staleness hurts
            + N(0, 2)
```

Feature ranges: mileage 20–120 km/wk, easy pace 4.5–7.5 min/km, long run
10–35 km, 3–21 days since long run, 3–7 runs/wk, distance ∈ {5, 10, 21.1,
42.2}.

**No real running data was used at any point.** `load_running_csv()` exists so
real data *can* be dropped in, but no shipped result uses it.

---

## Limitations

**The target is a closed-form function of exactly the six inputs.** Nothing is
missing, mismeasured, or confounded. Real running data has none of those
properties — it has injuries, weather, terrain, sleep, pacing errors, GPS
drift, and a finish time that depends on things no training log records. This
is close to the friendliest learning problem that could be constructed, and
the accuracy above should be read as such.

**Everything is bounded by a noise floor.** `N(0, 2)` on the target means no
predictor can beat 1.596 min MAE. At 4.12 the MLP is 2.6× above it — there is
signal it is not extracting, and "75.6% better than linear" hides that.

**Extrapolation is unguarded.** The model happily accepts inputs far outside
its training ranges and returns a confident number. The Streamlit app warns
when a slider leaves the trained range; `predict.py` validates types and
bounds but cannot tell you the answer is meaningless. Nothing in the model
itself knows.

**Only four race distances.** Trained on 5, 10, 21.1 and 42.2 km. A 15 km
prediction is interpolation between clusters it has never seen.

**No uncertainty estimate.** A single point prediction with no interval. A
runner cannot tell a confident prediction from a guess.

**No fairness analysis, because the question is not yet meaningful.** The
population is drawn from uniform distributions the repo chose. There are no
real people in it and therefore no subgroups to be unfair to. Any move to real
data makes this a real question immediately: training logs skew toward runners
who own devices and log consistently, which is not the population of runners.

---

## Ethical considerations

Low stakes by construction — this predicts a hobby race time, and the models
are not deployed anywhere. The realistic harm is a person trusting a number
that was never about them, which is what the top of this page is for.

If this were ever trained on real data, the checkpoint would embed training
statistics from real people's logs, and `models/` is gitignored partly for
that reason.

---

## Reproducing

```bash
python src/pacebrain/train_finish.py      # finish-time predictor
python src/pacebrain/train_pacing.py      # pacing model
python src/scratch/day10_baselines.py     # the baseline table above
python src/pacebrain/eval.py              # MAE/RMSE against Riegel
```

Everything is seeded (`seed=42`), and there is a test asserting two runs with
the same seed produce identical weights.

---

## Further reading

- `reports/day10_baselines.md` — the honest baseline set
- `reports/day9_riegel_audit.md` — why the original claim was retired
- `reports/lr_schedule_experiment.md` — a negative result, kept
- `PROGRESS.md` — the day-by-day log
