# Study notes — two readings of probability

**Frequentist.** Probability is long-run frequency. Parameters are fixed and
unknown; the data is random. A confidence interval is a statement about the
procedure across repeated samples.

**Bayesian.** Probability is degree of belief. Parameters have distributions; a
credible interval says the parameter is in this range with stated probability —
which is what people wrongly assume a confidence interval means.

## Where it actually matters

Mostly it does not, at large sample sizes, where the two converge and the
argument is philosophical. It matters when data is scarce, when prior knowledge
is genuinely available, or when the question asked is about a parameter rather
than about a procedure.

## The practical borrowing

Regularisation is a Bayesian prior in frequentist clothing. L2 is a gaussian
prior on the weights, L1 a Laplace one. That equivalence is worth holding onto,
because it explains why regularisation strength behaves like a statement of
confidence about how large the weights should be.
