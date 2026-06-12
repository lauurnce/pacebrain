"""PaceBrain — ML models for runner finish-time prediction and pacing."""

from pacebrain.config import FinishPredictorConfig
from pacebrain.data import (
    FEATURE_COLS,
    TARGET_COL,
    RunningDataset,
    make_datasets,
    make_sample_data,
)
from pacebrain.inference import load_finish_model, predict_finish_time, rebuild_scaler
from pacebrain.models import MLP, FinishTimePredictor

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
    "rebuild_scaler",
    "predict_finish_time",
]
