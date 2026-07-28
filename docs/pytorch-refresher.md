# PyTorch refresher notes

Written before Day 1 so the project does not stall on mechanics. These are the
concepts I expect to need, in the order they bite.

## Tensors

An n-dimensional array with two extras that matter: it can live on a device
(CPU/GPU), and it can record the operations performed on it.

The shape errors that will cost the most time:
- `(N,)` vs `(N, 1)` — a target of shape `(N,)` broadcast against predictions of
  shape `(N, 1)` silently produces an `(N, N)` loss instead of erroring. This is
  the single most common bug in regression code. Reshape targets to `(N, 1)`.
- float64 vs float32 — numpy defaults to float64, torch defaults to float32.
  Mixing them raises a dtype error at the layer boundary. Cast once, at the
  Dataset, not per batch.

## Autograd

`requires_grad=True` marks a tensor as something to differentiate with respect
to. Every operation on it records a node in a graph. `loss.backward()` walks
that graph backwards and accumulates `d(loss)/d(param)` into each `.grad`.

**Accumulates** is the key word — gradients add up across `backward()` calls
rather than replacing. That is why `optimizer.zero_grad()` comes first in the
loop. Forgetting it does not error; it just trains on the sum of all previous
gradients and diverges.

## nn.Module

Base class for anything with learnable parameters. Two rules:
- Call `super().__init__()` before assigning any layers, or parameter
  registration silently fails.
- Define one forward pass in `forward()`, then call the module (`model(x)`),
  never `model.forward(x)` — the former runs registered hooks, the latter
  skips them.

`nn.Sequential` is just a container that calls its children in order.

## The training loop

Always the same five lines, regardless of architecture:

```
optimizer.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
optimizer.step()
```

If the loop is right, swapping an MLP for an LSTM changes nothing but the model
class and the tensor shapes. Worth internalising early — it means the Day 7
sequence model should reuse the Day 4 loop almost verbatim.

## Train vs eval

`model.train()` and `model.eval()` toggle Dropout and BatchNorm behaviour.
`torch.no_grad()` separately stops graph construction. They are independent, and
inference wants both: `eval()` for correct behaviour, `no_grad()` for speed and
memory.

## What I expect to get wrong

- Normalising with statistics fitted on the full dataset instead of train only.
  This leaks validation information into training and inflates the score.
- Forgetting that a checkpoint saved via `state_dict()` contains weights only —
  not architecture, not preprocessing.
