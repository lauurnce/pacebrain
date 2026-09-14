# `seq_data.PacingSequenceDataset`

```python
class PacingSequenceDataset(Dataset):
    def __init__(self, X, y, mean=None, std=None)
```

Defined in `src/pacebrain/seq_data.py`. This walkthrough records what the class assumes about its inputs, what it guarantees about its outputs, and which tests pin that contract down.

## Assumes

- `X` has shape `(n_races, T, 6)` and `y` has shape `(n_races, T, 1)`, and every race has the same number of timesteps `T`. Races of different lengths use `VariableLengthPacingDataset` instead.
- `mean` and `std` have shape `(6,)` and were computed on the training races only, which is what `make_seq_datasets` does. Broadcasting then standardises every timestep of every race.
- Scaling happens only when both `mean` and `std` are given. Passing just one leaves `X` untouched, with no error.

## Guarantees

- `X` is standardised as `(X - mean) / std` into a new tensor, so the caller's tensor is not modified.
- `y` is never scaled, so losses and error metrics read directly in min/km.
- `len(ds)` is the number of races, and `ds[i]` returns `(X[i], y[i])`.

## Pinned by

- `tests/test_pacing_sequence_dataset.py`: passthrough without statistics, standardisation, unscaled targets, the caller's tensor left alone, and `len` with indexing.
- `tests/test_seq_data.py` exercises it through `make_seq_datasets`, including the training-split statistics.
