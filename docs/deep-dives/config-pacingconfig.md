# `config.PacingConfig`

```python
@dataclass
class PacingConfig
```

Defined in `src/pacebrain/config.py`. This walkthrough records what the class assumes about its inputs, what it guarantees about its outputs, and which tests pin that contract down.

## Assumes

- `input_size` equals `len(SEQ_FEATURES)` in `seq_data.py`. Nothing enforces it at construction; the default of 6 is only correct while the sequence features stay at six.
- `cell` is `"lstm"` or `"gru"`. The config does not validate it; `train_pacing.train` raises a `ValueError` naming the bad value when it builds the model.
- `dropout` only matters when `num_layers > 1`. It applies between stacked layers, and both models pass 0 to PyTorch for a single layer.

## Guarantees

- Every field has a default, so `PacingConfig()` is a complete, runnable configuration: an LSTM with hidden size 64 and one layer, 200 epochs, batch size 32, early stopping after 20 epochs, 600 synthetic races, and a checkpoint at `models/pacing_model.pt`.
- It has the same shape as `FinishPredictorConfig` on purpose, because `train_pacing.py` mirrors `train_finish.py`. One shared field differs in value: `lr_patience` is 6 here against 12 for the finish predictor.
- The learning-rate schedule is off by default, and `lr_patience` (6) is below `patience` (20).
- Switching architecture is a one-string change: `dataclasses.replace(PacingConfig(), cell="gru")`.

## Pinned by

- `tests/test_seq_models.py::test_config_input_size_matches_seq_features` for the `input_size` invariant.
- `tests/test_lr_schedule.py::test_schedule_is_off_by_default` and `test_lr_patience_is_below_early_stopping_patience` for the schedule defaults.
- `tests/test_gru.py::test_training_builds_the_requested_cell` and `test_unknown_cell_is_rejected_with_a_useful_message` for `cell`.
- `tests/test_api_surface_config.py` for the field names.
