# `train_pacing.train`

```python
def train(cfg: PacingConfig) -> nn.Module
```

Defined in `src/pacebrain/train_pacing.py`. This walkthrough records what the function assumes about its inputs, what it guarantees about its outputs, and which tests pin that contract down.

## Assumes

- `cfg` is a `PacingConfig` whose `cell` is `"lstm"` or `"gru"`. Anything else raises a `ValueError` naming the bad value before any training happens.
- The data is fixed-length: it comes from `make_seq_datasets`, so every race has 10 segments and the plain `nn.MSELoss` is correct. The variable-length pieces (`pad_collate`, `masked_mse_loss`) are not wired into this trainer.
- `cfg.checkpoint_path` and `cfg.plot_path` sit in a directory whose own parent exists. The trainer creates one directory level (`mkdir(exist_ok=True)`, no `parents=True`).

## Guarantees

- It seeds torch with `cfg.seed`, so the split, the initial weights and the shuffling repeat for a given config.
- It trains with Adam on MSE, evaluates on the validation split every epoch, and writes the weights from the best validation epoch to `cfg.checkpoint_path`.
- It stops early once validation loss has not improved for `cfg.patience` epochs.
- With `cfg.lr_schedule` on, it steps a `ReduceLROnPlateau` on validation loss and prints a line each time the learning rate is cut. The schedule is off by default.
- It writes a train/validation loss plot to `cfg.plot_path`, prints the best validation MSE and RMSE (in min/km), and returns the model with the best weights reloaded from disk.

## Pinned by

- `tests/test_training_loops.py`: `test_pacing_train_end_to_end` and `test_pacing_early_stopping_halts_after_patience_epochs`.
- `tests/test_gru.py`: `test_training_builds_the_requested_cell` and `test_unknown_cell_is_rejected_with_a_useful_message`.
- `tests/test_lr_schedule.py`: the schedule runs end to end, and the learning-rate message is printed when the rate is cut and only then.
