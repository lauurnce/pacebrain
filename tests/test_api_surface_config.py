"""API surface checks for pacebrain.config."""

import dataclasses
import importlib

FINISH_FIELDS = {
    "input_size", "hidden_sizes", "output_size", "dropout",
    "lr", "epochs", "batch_size", "patience",
    "lr_schedule", "lr_factor", "lr_patience",
    "n_samples", "val_fraction", "seed",
    "checkpoint_path", "plot_path",
}

PACING_FIELDS = {
    "cell", "input_size", "hidden_size", "num_layers", "dropout",
    "lr", "epochs", "batch_size", "patience",
    "lr_schedule", "lr_factor", "lr_patience",
    "n_races", "val_fraction", "seed",
    "checkpoint_path", "plot_path",
}


def _config_module():
    return importlib.import_module('pacebrain.config')


def test_config_finish_predictor_config_is_a_dataclass():
    assert dataclasses.is_dataclass(getattr(_config_module(), 'FinishPredictorConfig'))


def test_config_pacing_config_is_a_dataclass():
    assert dataclasses.is_dataclass(getattr(_config_module(), 'PacingConfig'))


def test_config_finish_predictor_config_field_names():
    """The field names are the API: train, eval and inference read them by name."""
    cls = getattr(_config_module(), 'FinishPredictorConfig')
    assert {f.name for f in dataclasses.fields(cls)} == FINISH_FIELDS


def test_config_pacing_config_field_names():
    cls = getattr(_config_module(), 'PacingConfig')
    assert {f.name for f in dataclasses.fields(cls)} == PACING_FIELDS


def test_config_hidden_sizes_default_is_not_shared_between_instances():
    """A config is threaded through several modules; a shared list would leak edits."""
    cls = getattr(_config_module(), 'FinishPredictorConfig')
    first, second = cls(), cls()
    first.hidden_sizes.append(8)
    assert second.hidden_sizes == [64, 32]
