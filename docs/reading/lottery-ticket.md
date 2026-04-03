# The Lottery Ticket Hypothesis — Frankle & Carbin, 2018

Claims a trained network contains a sparse subnetwork that, retrained from the
*original* initialisation, matches the full model.

The initialisation dependence is the surprising part — the same sparse structure
retrained from scratch with fresh weights does not work.

Suggests much of the parameter count is scaffolding for finding a good
subnetwork rather than doing the eventual work, which reframes pruning as
discovery rather than compression.
