# Math refresher

The calculus and linear algebra actually needed to follow what a training loop
is doing. Not the API — that is a separate note.

## Derivatives as slope

A derivative answers "if I nudge this input, how much does the output move, and
in which direction". Training is that question asked about every parameter at
once, with the answer used to nudge each one the opposite way.

## Partial derivatives and the gradient

With many inputs, the partial derivative treats all but one as constant. The
gradient is the vector of all of them.

Its key property: the gradient points in the direction of steepest **increase**.
Descent therefore steps against it. That is the entire reason for the minus
sign in `param -= lr * grad`, and it is worth being able to state, because a
sign error there produces a model that confidently gets worse every epoch.

## Chain rule

For `f(g(x))`, the derivative is `f'(g(x)) * g'(x)` — multiply the local
derivatives along the path.

A neural network is a deep composition of functions, so backpropagation is the
chain rule applied repeatedly from the loss backwards to each parameter. The
"vanishing gradient" problem is just this product becoming tiny when many
factors are below 1.

## Why MSE is convenient

`MSE = mean((y_pred - y_true)^2)`, so `d(MSE)/d(y_pred) = 2 * (y_pred - y_true)`.

Clean, and proportional to the error — bigger mistakes push harder. That is
also its weakness: the penalty grows quadratically, so a single outlier can
dominate. MAE penalises linearly but has an undefined derivative at zero and a
constant gradient elsewhere, which converges less smoothly.

Rule of thumb: **train on MSE, report MAE.** One is easier to optimise, the
other is easier for a human to interpret.

## Matrix shapes in a linear layer

`y = xW^T + b`. For a batch of `N` rows and `in`/`out` widths:

```
x: (N, in)    W: (out, in)    b: (out,)    y: (N, out)
```

Almost every shape error is a transpose or a missing batch dimension. Reading
the intended shape off this line is faster than guessing.

## Learning rate intuition

Too large overshoots the minimum and the loss oscillates or diverges. Too small
converges truly slowly and can stall in a flat region. `1e-3` is a reasonable
starting point for Adam and worth changing only with a reason.

## Standardisation

`z = (x - mean) / std` puts each feature on a comparable scale. Without it, a
feature measured in the tens (weekly mileage) dominates one measured in single
digits (pace) purely because of units, and the gradient for the smaller feature
gets swamped.

Fit those statistics on training data only. Using the full dataset lets
validation information leak into training.
