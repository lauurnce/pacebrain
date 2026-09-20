# `seq_data.VariableLengthPacingDataset`

```python
class VariableLengthPacingDataset(Dataset):
    def __init__(self, races, mean=None, std=None)
```

Defined in `src/pacebrain/seq_data.py`. This walkthrough records what the class assumes about its inputs, what it guarantees about its outputs, and which tests pin that contract down.

## Assumes

- `races` is a list of `(X_i, y_i)` pairs with `X_i` of shape `(T_i, 6)` and `y_i` of shape `(T_i, 1)`, where `T_i` differs by race: one timestep per kilometre.
- Batches are built with `pad_collate`, which pads to the longest race and returns the true lengths. The default collate calls `torch.stack` and fails as soon as a 5 km race and a marathon share a batch.
- As in `PacingSequenceDataset`, scaling needs both `mean` and `std`, computed on the training races.

## Guarantees

- Each race is standardised on its own, once, at construction, as `(X_i - mean) / std`. Its length is unchanged, and `__getitem__` is then a plain list index.
- Targets are never scaled.
- When scaling applies the dataset builds a new list and the caller's list is left alone. Without statistics it keeps the caller's list as is, with no copy.
- It is deliberately thinner than `PacingSequenceDataset`: a list rather than a stacked tensor, because the races cannot be stacked.

## Pinned by

- `tests/test_variable_length_dataset.py`: passthrough without statistics, per-race standardisation whatever the length, unscaled targets, the caller's list left alone, and generated races keeping their own lengths.
- `tests/test_variable_length.py` builds it through `make_variable_seq_datasets` and covers `pad_collate` and the padded-batch behaviour of the models.
