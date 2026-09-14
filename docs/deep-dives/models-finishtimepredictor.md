# `models.FinishTimePredictor`

```python
class FinishTimePredictor(MLP):
    def __init__(self, input_size=None, hidden_sizes=None, dropout=0.1)
```

Defined in `src/pacebrain/models.py`. This walkthrough records what the class assumes about its inputs, what it guarantees about its outputs, and which tests pin that contract down.

## Assumes

- Inputs have `input_size` features, six by default (`N_FEATURES`), and that count stays in sync with `len(FEATURE_COLS)` in `data.py`. The class does not check the width itself; a wrong width fails in the first `nn.Linear` at forward time.
- `hidden_sizes` is a list of layer widths, and `None` means `[64, 32]`.

## Guarantees

- It is an `MLP` with the output size fixed at 1, so `model(x)` maps `(batch, features)` to `(batch, 1)`: a finish time in minutes.
- The defaults are `input_size=6`, `hidden_sizes=[64, 32]` and `dropout=0.1`. Dropout follows each hidden layer when it is above zero. `MLP`'s own default is 0.0, so this wrapper is where the 0.1 comes from.
- `forward` is inherited unchanged from `MLP`.
- Being its own class keeps checkpoints self-documenting and gives loading code one constructor to call.

## Pinned by

- `tests/test_models.py`: `test_finish_time_predictor_defaults`, `test_finish_time_predictor_accepts_custom_hidden_sizes`, `test_finish_time_predictor_accepts_six_features`, `test_finish_time_predictor_rejects_wrong_input_width` and `test_n_features_matches_feature_cols`.
- `tests/test_checkpoint.py`, `tests/test_inference.py` and `tests/test_training_loops.py` construct it as part of the checkpoint and training round trip.
