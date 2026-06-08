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
