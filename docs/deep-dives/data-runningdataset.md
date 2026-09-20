# `data.RunningDataset`

```python
class RunningDataset(Dataset):
    def __init__(self, df, scaler=None, fit_scaler=False)
```

Defined in `src/pacebrain/data.py`. This walkthrough records what the class assumes about its inputs, what it guarantees about its outputs, and which tests pin that contract down.

## Assumes

- `df` holds every column in `FEATURE_COLS` plus `TARGET_COL`. Extra columns are ignored because only those columns are selected. A missing one raises a pandas `KeyError` here; `load_running_csv` names missing columns earlier, at load time.
- `fit_scaler=True` is used on the training split only. Fitting on validation rows would leak their mean and standard deviation into the scaler, and `make_datasets` is the caller that gets this right.

## Guarantees

- `X` is a float32 tensor of shape `(n, 6)` and `y` a float32 tensor of shape `(n, 1)`, built once at construction so `__getitem__` is a plain index.
- Exactly one of three scaling modes applies, in this order: `fit_scaler=True` fits a new `StandardScaler` (a `scaler` passed alongside it is ignored), otherwise a given `scaler` is applied with `transform`, otherwise features pass through raw and `.scaler` is `None`.
- The scaler that was used is available afterwards as `.scaler`, which is how `make_datasets` hands the training scaler to the validation set and on to inference.
- Targets are never scaled.

## Pinned by

- `tests/test_data.py`: `test_running_dataset_len`, `test_running_dataset_item_shapes`, `test_running_dataset_dtypes`, `test_no_scaler_passthrough` and `test_scaler_fit_on_train_only`.
- `tests/test_eval.py` builds datasets with `fit_scaler=True`.
