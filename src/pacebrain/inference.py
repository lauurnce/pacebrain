"""
Inference helpers — load the trained finish-time predictor and run it on
new inputs. Used by eval.py and the predict.py CLI.

New PyTorch concept:
    Loading a checkpoint is two steps: (1) build the model object with the
    SAME architecture hyperparameters used in training, (2) copy the saved
    weights into it with load_state_dict(). A state_dict is just weights,
    not architecture, so step 1 must match or loading fails with a shape error.

    weights_only=True tells torch.load to deserialize only tensors, which
    is safer than the default (full pickle can execute arbitrary code).
"""

import pathlib

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pacebrain.config import FinishPredictorConfig
from pacebrain.data import FEATURE_COLS, make_datasets, make_sample_data
from pacebrain.models import FinishTimePredictor


def load_finish_model(cfg: FinishPredictorConfig) -> FinishTimePredictor:
    """
    Build a FinishTimePredictor from cfg and load its trained weights.

    Returns the model already in eval() mode (Dropout disabled), ready
    for inference.
    """
    checkpoint_path = pathlib.Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run train_finish.py first."
        )
    model = FinishTimePredictor(hidden_sizes=cfg.hidden_sizes, dropout=cfg.dropout)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    return model


def rebuild_scaler(cfg: FinishPredictorConfig) -> StandardScaler:
    """
    Rebuild the StandardScaler that was fitted during training.

    The scaler is NOT stored in the checkpoint (state_dict holds only model
    weights). But because the training data is synthetic and seeded, calling
    make_sample_data + make_datasets with the same seed and val_fraction
    reproduces the exact same train split, so the refitted scaler has
    identical mean/std to the one used in training.
    """
    df = make_sample_data(n_samples=cfg.n_samples, seed=cfg.seed)
    _, _, scaler = make_datasets(df, val_fraction=cfg.val_fraction, seed=cfg.seed)
    return scaler


def predict_finish_time(
    model: FinishTimePredictor,
    scaler: StandardScaler,
    features: dict,
) -> float:
    """
    Predict a finish time (minutes) for one runner.

    Args:
        features: dict keyed by FEATURE_COLS names, e.g.
                  {"weekly_mileage_km": 60, "avg_pace_min_per_km": 5.5, ...}

    The model expects inputs scaled exactly like the training data, so new
    inputs must go through the SAME fitted scaler — never refit on one row.
    """
    # Build a (1, 6) array in FEATURE_COLS order — the order the model was trained on
    raw = np.array([[features[col] for col in FEATURE_COLS]], dtype=np.float32)
    X = scaler.transform(raw).astype(np.float32)

    model.eval()
    with torch.no_grad():  # inference only — skip building the autograd graph
        pred = model(torch.from_numpy(X))

    # .item() pulls a single value out of a tensor as a plain Python float
    return float(pred.item())
