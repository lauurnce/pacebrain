# Study notes — distributions worth knowing

Not an exhaustive list, just the ones that keep appearing.

**Normal.** Sums of many independent effects. The default for aggregates and a
poor default for raw measurements, which are usually skewed and non-negative.

**Log-normal.** What you get when effects multiply rather than add. Durations,
incomes, file sizes. Taking logs turns it back into a normal, which is why a
log transform fixes so many skewed features.

**Poisson.** Counts of independent events in a fixed window. Mean equals
variance, and when real count data has variance well above its mean that is
overdispersion, not noise.

**Exponential.** Waiting time between Poisson events. Memoryless, which is a
strong assumption and usually the one that breaks first.

**Uniform.** Rarely a real generating process, common as a prior or a synthetic
data generator.

The habit worth forming is asking which of these the residuals resemble. A loss
function encodes a distributional assumption, so residuals that clearly are not
that shape mean the loss is answering the wrong question.
