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
