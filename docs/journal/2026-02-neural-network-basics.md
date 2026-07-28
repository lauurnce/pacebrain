# Study notes — what a neural network actually adds

Following on from the foundations notes. The question I wanted answered: when
is a network worth it over a linear model?

## A single layer is just linear regression

`y = xW^T + b` is a linear model. Stacking two of them changes nothing —
`(xW1)W2 = x(W1 W2)`, which is another linear map. Depth without a nonlinearity
between layers is *arithmetically* pointless, not merely inefficient.

That is the entire justification for activation functions. Without them, a
hundred layers collapse to one.

## Activation functions

- **ReLU** — `max(0, x)`. Cheap, and its gradient is 1 for positive inputs, so
  it does not shrink signal on the way back. Default choice.
- **Sigmoid** — squashes to (0, 1). Its gradient is at most 0.25, so stacking
  sigmoids multiplies small numbers repeatedly and the gradient vanishes. This
  is the historical reason deep networks were hard to train.
- **Tanh** — zero-centred sigmoid, better than sigmoid, still saturates.

The "dying ReLU" problem: a unit whose input is always negative outputs zero
forever and receives zero gradient, so it never recovers. Leaky ReLU exists for
this. Probably not worth worrying about at small scale.

## What depth buys

A wide enough single hidden layer can approximate any continuous function in
principle. In practice depth is more parameter-efficient — each layer composes
features from the previous one, so representing an interaction between inputs
does not require enumerating every combination.

For tabular data with a handful of features, this advantage is modest. Which is
the honest answer to the original question: **on small tabular problems, a
network often does not beat a well-specified linear model.** It has to be
checked rather than assumed.

## Where networks clearly win

- Very high-dimensional inputs where hand-engineering features is impractical.
- Structure that a linear model cannot express and that would be tedious to
  encode manually — interactions, saturation, thresholds.
- Sequences and images, where the architecture itself encodes an assumption
  about the data.

## Capacity

More parameters means more ability to fit, including to fit noise. Capacity
should be matched to data size, not maximised. Starting small and growing only
when training error is stubbornly high is the sane order — the opposite
direction wastes time chasing overfitting you created yourself.

## Next

Optimisers — how the parameters actually get updated, and why plain SGD is
rarely what people use.
