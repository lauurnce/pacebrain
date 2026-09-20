# Pace fade

The slowdown over the second half of a race: pace per kilometre gets slower as the race goes on, whether as a mild drift or a sharp blow-up near the end. It is the opposite of a negative split. Fade comes from a mismatch between the race distance and the endurance built in training, so it grows with race length and with an underdeveloped aerobic base, and it is invisible if you only look at the finish time.

## See also

- [Bonking](bonking.md)
- [Cardiac drift](cardiac-drift.md)
- [Negative split](negative-split.md)
- [Splits](splits.md)

In PaceBrain, fade is what the pacing model exists to predict. In the synthetic data (`make_sample_sequences` in `src/pacebrain/seq_data.py`) pace is flat through the first half of the race and then multiplied by `1 + fade_strength * (fraction - 0.5)^2`, so the slowdown is gentle at 60% of the race and steep by 90%. `fade_strength` grows with the race distance and with two endurance deficits: a long run that is short relative to the race, and low weekly mileage.
