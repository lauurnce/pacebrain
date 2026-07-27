"""
Train the finish-time predictor on running data.

New PyTorch concepts introduced on Day 4
-----------------------------------------
DataLoader  — wraps a Dataset and yields shuffled mini-batches automatically.
              No more manual make_batches(); shuffle=True handles it.

state_dict  — an ordered dict of all learnable tensors (weights + biases).
              Saving state_dict (not the full model object) is the idiomatic
              PyTorch way to checkpoint because it's version-stable and
              architecture-agnostic.

Early stopping — if val loss doesn't improve for `patience` epochs we halt.
                 This is not a PyTorch feature; it's a plain Python counter.
                 It stops overfitting without manually tuning epoch count.

Run:
    python src/pacebrain/train_finish.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pacebrain.config import FinishPredictorConfig
from pacebrain.data import make_sample_data, make_datasets
from pacebrain.models import FinishTimePredictor


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Return mean train loss for one epoch."""
    model.train()
    total, n = 0.0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(X_batch)
        n += len(X_batch)
    return total / n


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> float:
    """Return mean val loss without touching the computation graph."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            loss = loss_fn(model(X_batch), y_batch)
            total += loss.item() * len(X_batch)
            n += len(X_batch)
    return total / n


def train(cfg: FinishPredictorConfig) -> FinishTimePredictor:
    torch.manual_seed(cfg.seed)

    # --- Data ------------------------------------------------------------------
    df = make_sample_data(n_samples=cfg.n_samples, seed=cfg.seed)
    train_ds, val_ds, _ = make_datasets(df, val_fraction=cfg.val_fraction, seed=cfg.seed)

    # DataLoader handles shuffling and batching; num_workers=0 is safe on Windows
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples   Val: {len(val_ds)} samples")

    # --- Model -----------------------------------------------------------------
    model = FinishTimePredictor(
        input_size=cfg.input_size,
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- Training loop with early stopping ------------------------------------
    checkpoint_path = pathlib.Path(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(exist_ok=True)

    best_val = float("inf")
    epochs_no_improve = 0       # early-stopping counter
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            # state_dict: save only weights/biases (not the whole model object)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            marker = "  *" if improved else ""
            print(
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
                f"  no-improve={epochs_no_improve}{marker}"
            )

        # Early stopping: halt if stuck for too long
        if epochs_no_improve >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
            break

    print(f"\nBest val MSE: {best_val:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

    # --- Loss curve ------------------------------------------------------------
    plot_path = pathlib.Path(cfg.plot_path)
    plot_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Finish-time predictor — loss over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Loss curve: {plot_path}")

    # Load best weights back before returning
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model


if __name__ == "__main__":
    cfg = FinishPredictorConfig()
    train(cfg)
