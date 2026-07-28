"""
Model evaluation and Riegel baseline comparison — Day 5.

New PyTorch concept:
    model.eval() disables Dropout and BatchNorm training behaviour.
    torch.no_grad() tells autograd to skip building the computation graph,
    which saves memory and speeds up forward passes during inference.
    Always use both together when you are not training.

Riegel formula (non-ML baseline):
    T2 = T1 * (D2 / D1) ^ 1.06
    We estimate T1 (a 10 km proxy) from the runner's average training pace.
    If the MLP cannot beat this formula, it has not learned anything useful.

MAE vs MSE:
    MSE penalises large errors quadratically.  A MSE of 30 is hard to
    interpret.  MAE of 5.5 min means "wrong by 5.5 minutes on average",
    which a runner immediately understands.

Run:
    python src/pacebrain/eval.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pacebrain.config import FinishPredictorConfig
from pacebrain.data import make_sample_data, make_datasets, TARGET_COL
from pacebrain.inference import load_finish_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae_minutes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in minutes."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_minutes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error in minutes.

    Reported alongside MAE because the two answer different questions and
    this application cares about both. MAE is the average miss — the number
    to quote to a runner, because "wrong by 4 minutes on average" is
    immediately meaningful. RMSE squares before averaging, so a single
    30-minute blow-up moves it far more than thirty 1-minute misses do.

    For race pacing the tail is what hurts: a prediction that is quietly
    good most of the time but occasionally catastrophic will send someone
    out at a pace they cannot hold. MAE cannot see that; RMSE can.

    Reading them together is what makes them useful. RMSE >= MAE always,
    and the size of the gap is the story: close together means errors are
    uniform, far apart means a few large misses dominate.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Riegel baseline
# ---------------------------------------------------------------------------

def riegel_predict(
    avg_pace_min_per_km: np.ndarray,
    race_distance_km: np.ndarray,
    ref_distance_km: float = 10.0,
) -> np.ndarray:
    """
    Riegel race-time formula: T2 = T1 * (D2 / D1) ^ 1.06

    T1 is estimated as avg_pace * ref_distance (a 10 km proxy from training pace).
    The exponent 1.06 is Riegel's empirical coefficient: longer races are harder
    per kilometre than shorter ones.
    """
    t1 = avg_pace_min_per_km * ref_distance_km
    return t1 * (race_distance_km / ref_distance_km) ** 1.06


# ---------------------------------------------------------------------------
# Model predictions
# ---------------------------------------------------------------------------

def get_model_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a DataLoader and return (y_true, y_pred) as numpy arrays.

    model.eval() disables Dropout so predictions are deterministic.
    torch.no_grad() skips the computation graph — no gradients needed here.
    """
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            all_true.append(y_batch.numpy())
            all_pred.append(pred.numpy())
    y_true = np.concatenate(all_true).ravel()
    y_pred = np.concatenate(all_pred).ravel()
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    y_true_model: np.ndarray,
    y_pred_model: np.ndarray,
    y_true_riegel: np.ndarray,
    y_pred_riegel: np.ndarray,
    save_path: str,
) -> None:
    """
    Side-by-side scatter of predicted vs actual for MLP and Riegel.

    Points on the red dashed line are perfect predictions.
    Scatter above the line = over-estimated; below = under-estimated.
    """
    model_mae = mae_minutes(y_true_model, y_pred_model)
    riegel_mae = mae_minutes(y_true_riegel, y_pred_riegel)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (axes[0], y_true_model, y_pred_model, f"MLP  (MAE = {model_mae:.1f} min)"),
        (axes[1], y_true_riegel, y_pred_riegel, f"Riegel  (MAE = {riegel_mae:.1f} min)"),
    ]
    for ax, y_true, y_pred, title in panels:
        ax.scatter(y_true, y_pred, alpha=0.4, s=18, color="steelblue")
        lo = min(y_true.min(), y_pred.min()) - 5
        hi = max(y_true.max(), y_pred.max()) + 5
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect")
        ax.set_xlabel("Actual finish time (min)")
        ax.set_ylabel("Predicted finish time (min)")
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.legend()

    fig.suptitle("Finish-time predictor: MLP vs Riegel baseline (validation set)")
    plt.tight_layout()
    pathlib.Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Scatter plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(cfg: FinishPredictorConfig) -> None:
    """Load checkpoint, evaluate on val set, compare to Riegel, save plot."""
    torch.manual_seed(cfg.seed)

    df = make_sample_data(n_samples=cfg.n_samples, seed=cfg.seed)
    train_ds, val_ds, _ = make_datasets(df, val_fraction=cfg.val_fraction, seed=cfg.seed)

    # Reconstruct val DataFrame for Riegel (needs raw unscaled features).
    # Must use the same RNG seed and order as make_datasets() to get identical split.
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * cfg.val_fraction)
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # --- Load model checkpoint -----------------------------------------------
    model = load_finish_model(cfg)

    # --- Model predictions ---------------------------------------------------
    y_true_model, y_pred_model = get_model_predictions(model, val_loader)
    model_mae = mae_minutes(y_true_model, y_pred_model)

    # --- Riegel baseline -----------------------------------------------------
    riegel_pred = riegel_predict(
        val_df["avg_pace_min_per_km"].values,
        val_df["race_distance_km"].values,
    )
    y_true_riegel = val_df[TARGET_COL].values
    riegel_mae = mae_minutes(y_true_riegel, riegel_pred)

    # --- Print results -------------------------------------------------------
    model_rmse = rmse_minutes(y_true_model, y_pred_model)
    riegel_rmse = rmse_minutes(y_true_riegel, riegel_pred)
    improvement = (riegel_mae - model_mae) / riegel_mae * 100

    print("\n=== Evaluation (validation set) ===")
    print(f"  Samples : {len(val_df)}")
    print(f"  {'':8}{'MAE':>9}{'RMSE':>9}{'ratio':>9}")
    for name, mae, rmse in (
        ("MLP", model_mae, model_rmse),
        ("Riegel", riegel_mae, riegel_rmse),
    ):
        print(f"  {name:<8}{mae:>9.2f}{rmse:>9.2f}{rmse / mae:>9.2f}")

    # RMSE/MAE is the shape of the error distribution in one number. It is
    # sqrt(pi/2) = 1.2533 for pure gaussian noise, so near that means the
    # misses are uniform; well above it means a few large ones dominate and
    # the average is hiding a tail.
    print("\n  (ratio 1.25 = gaussian errors; higher = a few large misses dominate)")
    if improvement > 0:
        print(f"  MLP beats Riegel by {improvement:.1f}% on MAE")
    else:
        print(f"  Riegel beats MLP by {-improvement:.1f}% on MAE  (model needs improvement)")

    # --- Save scatter plot ---------------------------------------------------
    plot_comparison(
        y_true_model, y_pred_model,
        y_true_riegel, riegel_pred,
        save_path="reports/day5_scatter.png",
    )


if __name__ == "__main__":
    cfg = FinishPredictorConfig()
    run_evaluation(cfg)
