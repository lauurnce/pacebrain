# `config.FinishPredictorConfig`

```python
@dataclass
class FinishPredictorConfig
```

Defined in `src/pacebrain/config.py`. This walkthrough records what the class assumes about its inputs, what it guarantees about its outputs, and which tests pin that contract down.

## Assumes

- `input_size` equals `len(FEATURE_COLS)` in `data.py`. Nothing enforces it at construction; the default of 6 is only correct while the feature list has six columns.
- The architecture fields (`input_size`, `hidden_sizes`, `output_size`, `dropout`) are the same ones that trained a checkpoint and later rebuild the model to load it. A checkpoint stores weights, not architecture, so a mismatch surfaces as a `load_state_dict` error rather than a wrong answer.
- If `lr_schedule` is turned on, `lr_patience` stays below `patience`. The two counters race, and at or above `patience` early stopping halts training before the schedule acts.

## Guarantees

- Every field has a default, so `FinishPredictorConfig()` is a complete, runnable configuration: 300 epochs, batch size 32, early stopping after 25 epochs without improvement, and a checkpoint at `models/finish_predictor.pt`.
- The learning-rate schedule is off by default (`lr_schedule=False`), because it was measured and did not earn its place (see `reports/lr_schedule_experiment.md`).
- `hidden_sizes` defaults through `field(default_factory=...)`, so each instance gets its own list and editing one config's widths does not change another's.
- To vary one field without mutating a shared object, use `dataclasses.replace(cfg, epochs=10)`.

## Pinned by

- `tests/test_models.py::test_config_input_size_matches_feature_cols` for the `input_size` invariant.
- `tests/test_lr_schedule.py::test_schedule_is_off_by_default` and `test_lr_patience_is_below_early_stopping_patience` for the schedule defaults.
- `tests/test_api_surface_config.py` for the field names and the unshared `hidden_sizes` default.
