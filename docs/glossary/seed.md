# Seed

The value initialising a pseudo-random number generator, making a run reproducible. Fixing one is necessary for reproducibility but dangerous for evaluation: a result measured on a single seed may be reporting that seed rather than the method, which is why comparisons need several.

The failure mode to watch with Seed is silent optimism: results improve on the validation set for reasons that will not survive contact with next month's data.
