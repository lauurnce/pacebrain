# Study notes — outliers

A value far from the rest. The hard part is deciding whether it is an error or
the most interesting row in the dataset.

## Two kinds

**Data errors.** A 2-minute-per-kilometre pace, a 500 km week, a negative
duration. Physically impossible, so they are bugs in collection — a paused
watch, a unit mix-up, a manual entry typo.

**Genuine extremes.** A real elite performance, a real 200 km week. Rare but
true, and deleting them teaches the model that the tail does not exist.

Distinguishing them requires domain knowledge, which is a concrete argument for
working in a field you understand. A statistic cannot tell you that 2:00/km is
impossible; you have to know it.

## Detection

- **Range checks from domain knowledge.** The most reliable, and the only one
  that catches an error sitting inside the statistical range.
- **z-score.** More than 3 standard deviations out. Assumes roughly normal
  data, and is itself distorted by the outliers it is looking for.
- **IQR.** Beyond 1.5x the interquartile range from the quartiles. Distribution
  free and much harder to fool.

## What to do

**Fix** if the true value is recoverable. **Remove** if it is clearly an error
and cannot be repaired. **Keep** if it is genuine — and use a loss that does
not let it dominate, which is the argument for MAE or Huber over MSE.

**Cap** as a middle path, though it fabricates a value that was never observed.

## The asymmetry worth remembering

Removing genuine extremes is more damaging than keeping a few errors. The model
learns the range it is shown, and a model that has never seen a high-volume
runner will extrapolate badly for one — confidently, and with no warning.

Which is the case for validating input ranges at inference rather than trusting
the model to behave sensibly outside what it was trained on.
