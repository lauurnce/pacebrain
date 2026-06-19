"""PaceBrain — ML models for runner finish-time prediction and pacing."""

from pacebrain.config import FinishPredictorConfig, PacingConfig
from pacebrain.data import (
    FEATURE_COLS,
    TARGET_COL,
    RunningDataset,
    make_datasets,
    make_sample_data,
)
from pacebrain.inference import (
    load_finish_model,
    load_scaler,
    predict_finish_time,
    rebuild_scaler,
)
from pacebrain.models import MLP, FinishTimePredictor
from pacebrain.seq_data import (
    N_SEGMENTS,
    SEQ_FEATURES,
    PacingSequenceDataset,
    make_sample_sequences,
    make_seq_datasets,
)
from pacebrain.seq_models import PacingLSTM

__all__ = [
    "MLP",
    "FinishTimePredictor",
    "RunningDataset",
    "make_sample_data",
    "make_datasets",
    "FEATURE_COLS",
    "TARGET_COL",
    "FinishPredictorConfig",
    "load_finish_model",
    "load_scaler",
    "rebuild_scaler",
    "predict_finish_time",
    "PacingLSTM",
    "PacingConfig",
    "make_sample_sequences",
    "PacingSequenceDataset",
    "make_seq_datasets",
    "N_SEGMENTS",
    "SEQ_FEATURES",
]
