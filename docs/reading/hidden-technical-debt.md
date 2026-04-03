# Hidden Technical Debt in Machine Learning Systems — Sculley et al., 2015

Argues ML systems accumulate debt in ways ordinary software does not: the model
code is a small fraction of the system, and data dependencies are harder to
manage than code dependencies.

Names the specific patterns — entanglement, where changing anything changes
everything; correction cascades; undeclared consumers; configuration sprawl.

The most immediately applicable paper here for anything that will run more than
once.
