# Study notes — making a trained model fast

Training cost is paid once. Inference cost is paid per request, so the
optimisation targets are different.

**Quantisation.** Store and compute weights in 8-bit integers. Large speed and
memory wins, small accuracy cost, and post-training quantisation is often good
enough without retraining.

**Pruning.** Remove weights that contribute little. Unstructured pruning gives
sparse matrices that most hardware cannot exploit; structured pruning removes
whole channels and delivers real speedups.

**Distillation.** Train a small model to match a large one's outputs. The soft
predictions carry more information than the hard labels, which is why the
student can beat training on labels alone.

**Batching.** Grouping requests raises throughput at the cost of latency for
whichever request waits. The right point depends entirely on which one the
product cares about.

## The measurement point

Optimise after profiling. Preprocessing and data movement are frequently the
real cost, and shrinking a model that was never the bottleneck buys nothing.
