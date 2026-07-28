# Study notes — statistics refresher

Starting from the bottom. Most of this was covered at some point and has since
gone fuzzy, and it underpins everything else.

## Distributions

A distribution describes how likely each value is. The ones worth recognising:

- **Normal.** Symmetric, defined by mean and standard deviation. Shows up
  whenever many small independent effects add together, which is often.
- **Uniform.** Every value in a range equally likely. Mostly appears when
  someone generates synthetic data rather than in nature.
- **Log-normal.** Right-skewed; the log of the variable is normal. Common for
  quantities that cannot go below zero and have a long upper tail — durations,
  incomes, file sizes.

Recognising a skewed distribution matters because the mean stops being a good
summary of it. For a right-skewed variable the mean sits above the median and
describes almost nobody.

## Mean, variance, standard deviation

Mean is the balance point. Variance is the average squared distance from it;
standard deviation is its square root, which puts it back into the units of the
original variable — which is why it is the one worth quoting.

Squaring in the variance is not arbitrary: it makes deviations positive and
penalises large ones more, and it makes the algebra work out. It also makes it
sensitive to outliers, which is a real cost.

## Standardisation

`z = (x - mean) / sd` expresses a value as "how many standard deviations from
average". It makes variables measured in different units comparable, which is
exactly the problem that comes up when features have wildly different scales.

## Correlation

Correlation measures linear association, between -1 and 1. Two traps:

- **Zero correlation does not mean independent.** A perfect U-shaped
  relationship has correlation near zero, because correlation only sees
  straight lines.
- **Correlation is not causation.** Genuinely worth repeating, because the
  failure mode in practice is subtler than the slogan suggests — usually a
  third variable driving both.

## Sampling and variation

Any statistic computed from a sample is itself uncertain. Two samples from the
same population give different means, and the spread of that difference shrinks
roughly with the square root of the sample size.

Practical consequence: small samples produce unstable estimates, and a
difference between two small groups is frequently just noise. Worth remembering
before concluding that a change improved anything.

## Why this matters for what comes next

Normalisation, error metrics and the idea that a measured improvement might be
noise are all statistics rather than machine learning. Getting them wrong
undermines everything built on top, and no amount of model sophistication
repairs it.
