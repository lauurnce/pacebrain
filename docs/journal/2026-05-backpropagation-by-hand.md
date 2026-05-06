# Study notes — backprop on paper

Worked the chain rule through a two-layer network by hand before letting a
library do it, because the failure modes only make sense once you have seen the
shapes line up.

## The setup

    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    L  = (z2 - y)**2

## Backward

    dL/dz2 = 2 * (z2 - y)
    dL/dW2 = dL/dz2 @ a1.T
    dL/da1 = W2.T @ dL/dz2
    dL/dz1 = dL/da1 * (z1 > 0)
    dL/dW1 = dL/dz1 @ x.T

## What the exercise actually taught

Every weight gradient is an outer product of "error arriving here" with "input
that went in". That is why the shapes always work out, and why a transpose in
the wrong place produces a shape error rather than a subtly wrong answer —
which is a mercy.

The ReLU derivative is the interesting one. It is a gate: gradient passes
where the unit was active and is zeroed where it was not. A unit that is
never active receives no gradient ever and cannot recover. That is the dying
ReLU problem, and it is visible directly in the algebra rather than being a
fact to memorise.
