# PaceBrain — Progress Log

---

## Day 1 — Foundations and repo setup (2026-06-08)

**What was built:**
Repo initialized with .gitignore, requirements.txt, and full folder structure.
Two scratch scripts: `tensors.py` (tensor creation, shapes, basic ops, requires_grad, .backward()) and `manual_grad_descent.py` (fitting y = 2x + 1 from noisy data using only autograd and manual weight updates).

**PyTorch concept learned:**
Autograd — when you set `requires_grad=True` on a tensor, PyTorch builds a computation graph as you run operations. Calling `loss.backward()` walks the graph in reverse and deposits d(loss)/d(param) into each parameter's `.grad`. You then step the parameter manually: `param -= lr * param.grad`. Always zero the grad after each step or it accumulates.

**One thing to review:**
Why `.grad` accumulates by default instead of resetting. Related: why `optimizer.zero_grad()` exists and when you would want accumulation (gradient accumulation for large batches).

---

## Day 2 — First neural net (MLP) on toy data (2026-06-08)

**What was built:**
`models.py` with a generic `MLP` class: an `nn.Module` subclass that stacks `nn.Linear` + `nn.ReLU` (plus optional `nn.Dropout`) layers into an `nn.Sequential`. `data.py` with `make_synthetic_data()` (a linear combination of features plus a nonlinear x0^2 term plus noise, so the net has something non-trivial to fit) and `train_val_split()`. `train.py` with a full training loop: manual mini-batching via `torch.randperm`, `nn.MSELoss`, `torch.optim.Adam`, best-checkpoint saving to `models/mlp_synthetic.pt`, and a train/val loss curve saved to `reports/day2_loss_curve.png`.

**PyTorch concept learned:**
`nn.Module` and the forward pass. A model is a class where layers are defined in `__init__` and `forward()` describes one pass of data through them; autograd derives the backward pass for free. Every training step is the same five-beat cycle: `optimizer.zero_grad()` to clear old gradients, forward pass with `model(x)`, compute the loss, `loss.backward()` to fill `.grad` on every parameter, `optimizer.step()` to update the weights. Every training loop I will ever write is a variation of this.

**One thing to review:**
Why you call `model(x)` instead of `model.forward(x)`. `nn.Module.__call__` runs registered hooks and other bookkeeping before delegating to `forward()`, so calling `forward()` directly silently skips that machinery.

---

## Day 3 — Real running data pipeline (2026-06-09)

**What was built:**
Extended `data.py` with three new pieces: `make_sample_data()` generates a synthetic running DataFrame (weekly mileage, avg pace, long run distance, days since long run, runs per week, race distance) with finish times derived from a physics-inspired formula. `RunningDataset` subclasses `torch.utils.data.Dataset` and wraps the DataFrame into indexed tensors. `make_datasets()` handles the train/val split and fits a `StandardScaler` on the training rows only. `tests/test_data.py` covers shapes, dtypes, reproducibility, the no-leakage invariant, and DataLoader batching.

**PyTorch concept learned:**
`Dataset` and `DataLoader`. A `Dataset` only needs two methods: `__len__` (total samples) and `__getitem__` (return one sample by index). `DataLoader` wraps a Dataset and handles shuffling, batching, and parallel loading automatically. You never write the batch loop by hand; you just iterate the loader.

**One thing to review:**
Data leakage via the scaler: if you fit `StandardScaler` on the full dataset before splitting, the val set's mean and std contaminate the scaler. The model then "sees" val statistics during training. Always split first, fit on train only, then transform both.

---

## Day 4 — Finish-time predictor (train) (2026-06-11)

**What was built:**
`FinishTimePredictor` model class in `models.py` (thin wrapper around MLP, locked to 6 running features). `config.py` with a `FinishPredictorConfig` dataclass holding all hyperparameters. `train_finish.py` with a full train/val loop using `DataLoader`, per-epoch validation, early stopping, and `state_dict` checkpointing to `models/finish_predictor.pt`. Training stopped at epoch 189 (patience=25), best val MSE 30.48 (RMSE ~5.5 min).

**PyTorch concept learned:**
`state_dict` — a plain ordered dict of every learnable tensor in a model keyed by layer name. Saving `model.state_dict()` (not the model object itself) is idiomatic because it is version-stable: you can load weights into a freshly constructed model even if the class definition changes, as long as the architecture matches. Load with `model.load_state_dict(torch.load(...))`.

**One thing to review:**
Overfitting signs: when train loss keeps falling but val loss flattens or rises, the model is memorizing training samples. Early stopping is one fix; dropout (already wired in) is another. Watch the gap between train and val loss curves, not just the absolute values.

---

## Day 5 — Evaluation and a real baseline (2026-06-11)

**What was built:**
`eval.py` with three components: `mae_minutes()` computes mean absolute error in interpretable units (minutes, not minutes-squared); `riegel_predict()` implements the Riegel race-time formula T2 = T1 * (D2/D1)^1.06 as a non-ML baseline; `run_evaluation()` loads the Day 4 checkpoint, runs both the MLP and Riegel on the validation set, prints a comparison, and saves a side-by-side scatter plot to `reports/day5_scatter.png`.

**Results:**
MLP MAE: 4.12 min. Riegel baseline MAE: 28.36 min. MLP beats Riegel by 85.5%. The Riegel baseline is weak here because we use easy training pace as the 10 km reference time proxy, which overshoots race pace significantly. In real use Riegel needs an actual prior race time, not a training pace.

**PyTorch concept learned:**
`model.eval()` and `torch.no_grad()` work together during inference. `model.eval()` disables Dropout and BatchNorm's training behaviour so predictions are deterministic. `torch.no_grad()` tells autograd to skip building the computation graph entirely, saving memory and speeding up forward passes. Always use both when you are not training.

**One thing to review:**
MAE vs RMSE vs MSE: all measure prediction error but penalise outliers differently. MSE squares errors (large mistakes dominate). RMSE is MSE square-rooted (same units as target, outlier-sensitive). MAE is the mean of absolute errors (robust to outliers, most interpretable for end users). For a running app where a 30-min prediction error is catastrophic, RMSE would punish it harder than MAE, which could be more appropriate.

---

## Day 6 — Refactor and inference CLI (2026-06-13)

**What was built:**
New `inference.py` module with three helpers: `load_finish_model()` (construct model, load state_dict, switch to eval mode), `rebuild_scaler()` (refit the training StandardScaler deterministically from the same seed, since the checkpoint stores only weights), and `predict_finish_time()` (scale one feature dict, forward pass under no_grad, return minutes). `eval.py` refactored to use the shared loader. `__init__.py` now exports the public package API with `__all__`. New `predict.py`: an argparse CLI that takes six training stats and prints predicted finish time in minutes, H:MM:SS, and implied race pace. Also backfilled the missing Day 2 log entry and filled the README results table and usage section.

**Results:**
CLI smoke test: 60 km/week at 5.5 min/km easy pace with a 28 km long run predicts a 42.2 km race at 199.0 min (3:19:00), within 0.1 min of the synthetic ground-truth formula. All 10 tests pass and eval.py output is unchanged after the refactor (MLP MAE still 4.12 min).

**PyTorch concept learned:**
A checkpoint made with `torch.save(model.state_dict())` holds weights only, not architecture and not preprocessing. Inference therefore needs three things reassembled: the model class built with the same hyperparameters, the weights loaded into it, and the exact same input scaling as training. Forgetting the scaler gives silently wrong predictions, not an error.

**One thing to review:**
Why the scaler can be rebuilt from a seed here (synthetic, deterministic data) but in a real project it must be persisted alongside the checkpoint, for example with joblib or by saving its mean and scale arrays in the checkpoint dict.

---

## Day 7 — Sequence model intro (LSTM) (2026-06-13)

**What was built:**
`seq_data.py`: each race becomes a sequence of 10 equal-distance segments. `make_sample_sequences()` generates per-segment pace with a fade model (pace rises in the back half, worse for long races and runners with an endurance deficit). `PacingSequenceDataset` plus `make_seq_datasets()` with a by-race train/val split and train-only normalization stats. `seq_models.py`: `PacingLSTM`, an `nn.LSTM` (batch_first) plus a per-timestep `nn.Linear` head mapping (batch, 10, 6) to (batch, 10, 1). `train_pacing.py` mirrors `train_finish.py` almost line for line. `tests/test_seq_data.py` covers shapes, determinism, the fade property, and no leakage.

**Results:**
Trained 200 epochs (no early stop). Best val MSE 0.0084, val RMSE 0.092 min/km per segment, which sits at the generator's noise floor (0.08), so the model learned essentially all the available signal. On a validation marathon with a 2.2 min/km blow-up the predicted back-half paces track the actual fade within about 0.3 min/km.

**PyTorch concept learned:**
An LSTM carries a hidden state and a cell state across timesteps, with gates deciding what to remember and forget; that running memory is what models fatigue accumulating over a race. `nn.LSTM` returns the hidden state at every timestep (used here for per-segment predictions) plus the final states only (what you would use for a single summary output). The training loop itself did not change at all from Day 4: only the model class and tensor shapes did.

**One thing to review:**
Variable-length sequences. Fixed 10 segments lets batching be a simple stack; real per-km splits give 5 to 42 timesteps per race and need padding plus `pack_padded_sequence` so the LSTM skips the padded steps. Also review LSTM vs GRU trade-offs.

---

## Day 8 — Persisting the scaler with the checkpoint (2026-06-19)

**What was built:**
Resolved the open question from Day 6. `train_finish.py` now saves a dict rather than a bare `state_dict`: the weights, the fitted scaler's `mean_` and `scale_` as float64 tensors, and the `FEATURE_COLS` order the model was trained on. New `load_scaler()` in `inference.py` reads those statistics back, falling back to the old `rebuild_scaler()` when the checkpoint predates the change. `load_finish_model()` accepts both formats by looking for a `"state_dict"` key, so Day 4-7 checkpoints still load. `predict.py` and the Streamlit app now call `load_scaler()`.

**Results:**
End-to-end unchanged: 60 km/week at 5.5 min/km with a 28 km long run still predicts 199.0 min (3:19:00) for a marathon, matching the Day 6 smoke test exactly. 166 tests pass. Six new tests cover the round trip, the fallback paths, and the case that actually matters — a checkpoint whose statistics differ from the synthetic generator now returns the saved values instead of silently substituting synthetic ones.

**PyTorch concept learned:**
`weights_only=True` restricts `torch.load` to tensors and simple containers rather than running the full pickle machinery, which can execute arbitrary code. That constrains what a checkpoint may contain: numpy arrays are not tensors, so the scaler statistics have to be converted with `torch.tensor(...)` before saving. Keeping the flag on was worth the conversion — loading an untrusted checkpoint should never be able to run code.

**One thing to review:**
Checkpoint versioning. The dict format is self-describing enough to sniff today, but a `"version"` key would beat key-presence checks once there are three formats. Also worth reading how `safetensors` handles this, since it stores metadata alongside tensors by design.

---

## Day 9 — Auditing the Riegel baseline (2026-06-26)

**What was built:**
`src/scratch/riegel_audit.py`, a reproducible check on the Day 5 baseline claim, plus `reports/day9_riegel_audit.md` writing up the result and a correction to the README results table.

**Results:**
The Day 5 explanation was wrong. `riegel_predict()` is algebraically identical to the first two factors of the data generator: `10 * (D/10)**1.06` reduces to `D * (D/10)**0.06`, which is exactly `base * distance_penalty` in `make_sample_data()`. Maximum absolute difference over 5000 rows is 6.10e-05, pure float32 rounding. So the baseline is not weak — it reproduces part of the target exactly, and its entire 27.56 min error is the three fitness factors it cannot see. Their product averages 0.8345 rather than 1.0, meaning Riegel overpredicts by about 20% on every row. Per-distance MAE (6.55, 13.72, 30.42, 61.82 min for 5/10/21.1/42.2 km) confirms multiplicative bias rather than a constant offset.

**Concept learned:**
A baseline is only informative if it had a fair shot at the problem. This one was built from a subset of the generating function and then handed an input it was never designed for — Riegel predicts a race time from another race time, not from easy training pace. The other half of the lesson is the noise floor: the target carries `N(0, 2)` noise, and `E|N(0,2)| = 2*sqrt(2/pi) = 1.596 min` is the best MAE anything can achieve here. The MLP's 4.12 min is 2.6x above that, so "85.5% better than baseline" was measuring the baseline's handicap more than the model's skill.

**One thing to review:**
Building an honest baseline set — predicting the training mean, a linear regression on the same six features, and a bias-corrected Riegel. If the MLP cannot clearly beat a linear model on data this well behaved, the extra capacity is not earning its place.

---

## Day 10 — An honest baseline set (2026-07-28)

**What was built:**
`src/pacebrain/baselines.py`, implementing the three baselines Day 9 asked for: the training mean, Riegel rescaled by a single constant fitted on train, and least squares on the same six features the MLP gets. `evaluate_baselines()` fits all of them on the training split and scores them on validation using the same seeded permutation the MLP is scored on. `src/scratch/day10_baselines.py` prints the comparison table, `reports/day10_baselines.md` writes it up, and the README results table is replaced with all five rows plus the noise floor.

**Results:**
MAE on the 200-row validation set: mean of train 67.97, raw Riegel 28.36, corrected Riegel 21.82, linear regression 16.89, MLP 4.12, against a noise floor of 1.60. So the honest headline is that the MLP beats the strongest baseline by **75.6%**, not the 85.5% the README used to claim against a mis-specified Riegel. The bias correction is the nicest confirmation: the fitted scale is 0.8135 and Day 9 predicted 0.8345 from first principles, and one constant recovers 6.5 of Riegel's 28.4 minutes without the formula learning anything. Day 9's open question is answered — linear regression is a real contender and the MLP still cuts its error by three quarters, because the generator is a product of five terms and no additive model can represent that. 12 new tests, 189 total.

**Concept learned:**
Baselines are a ladder, not a single rung. Each of these sees strictly more than the one above it, so the *gaps* localise where the model's advantage comes from: mean to Riegel is "features help", raw to corrected Riegel is "the error was bias", corrected Riegel to linear is "the other four features matter", linear to MLP is "the relationship is nonlinear". One baseline gives a number; a ladder gives an explanation. The second habit is reading every score against the noise floor rather than zero — a perfect model scores 1.60 min here, so the MLP's 4.12 is 2.6x off the achievable best, which "75.6% better than linear" completely hides.

**One thing to review:**
A log-space linear baseline. The generator is multiplicative, so logs turn it into a sum and a linear fit on log-transformed features should close most of the gap to the MLP. If it does, the honest claim weakens from "the MLP beats linear" to "the MLP rediscovered a log-linear relationship that one line of feature engineering would have handed it" — which is the more useful thing to know, and the natural Day 11.
