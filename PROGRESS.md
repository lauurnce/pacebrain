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

## Day 3 — Real running data pipeline (2026-06-09)

**What was built:**
Extended `data.py` with three new pieces: `make_sample_data()` generates a synthetic running DataFrame (weekly mileage, avg pace, long run distance, days since long run, runs per week, race distance) with finish times derived from a physics-inspired formula. `RunningDataset` subclasses `torch.utils.data.Dataset` and wraps the DataFrame into indexed tensors. `make_datasets()` handles the train/val split and fits a `StandardScaler` on the training rows only. `tests/test_data.py` covers shapes, dtypes, reproducibility, the no-leakage invariant, and DataLoader batching.

**PyTorch concept learned:**
`Dataset` and `DataLoader`. A `Dataset` only needs two methods: `__len__` (total samples) and `__getitem__` (return one sample by index). `DataLoader` wraps a Dataset and handles shuffling, batching, and parallel loading automatically. You never write the batch loop by hand; you just iterate the loader.

**One thing to review:**
Data leakage via the scaler: if you fit `StandardScaler` on the full dataset before splitting, the val set's mean and std contaminate the scaler. The model then "sees" val statistics during training. Always split first, fit on train only, then transform both.
