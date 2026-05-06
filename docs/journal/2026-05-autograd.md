# Study notes — automatic differentiation

Three ways to get a derivative, and only one of them is practical here.

**Symbolic** — manipulate the expression algebraically. Exact, but the
expression grows explosively for anything deep.

**Numerical** — `(f(x+h) - f(x)) / h`. Trivial to write, and wrong in an
interesting way: too large an `h` and the approximation is poor, too small and
floating point cancellation destroys it. Useful only as a check.

**Automatic** — apply the chain rule mechanically to the recorded operations.
Exact to floating point, and costs roughly one extra pass. This is what
frameworks do.

## Forward vs reverse mode

Reverse mode computes all derivatives of *one output* with respect to *many
inputs* in a single pass. Forward mode does the opposite. Training has one
scalar loss and millions of parameters, so reverse mode is the obvious fit —
and the reason it is the default everywhere.

## The gradient check

Numerical differentiation is bad for training and excellent for testing. If a
hand-written gradient disagrees with a finite difference beyond tolerance, the
hand-written one is wrong. Worth doing once on anything custom.
