"""
Train the LSTM pacing model on per-segment race data.

New PyTorch concept introduced on Day 7
----------------------------------------
There is exactly one: the model is RECURRENT, but the training loop is
IDENTICAL to Day 4's. That is the point of PyTorch's abstraction — the
loop only sees "tensors in, loss out, backward, step". Swapping the MLP
for an LSTM changed the model class and the tensor shapes
((B, 6) -> (B, 10, 6)), not one line of the loop itself.

Diff this file against train_finish.py to see how little changed.

Run:
    python src/pacebrain/train_pacing.py
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

from pacebrain.config import PacingConfig
from pacebrain.seq_data import make_seq_datasets
from pacebrain.seq_models import PacingLSTM


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
        # MSELoss works unchanged on (B, T, 1) tensors: it just averages the
        # squared error over batch AND timesteps, so nothing special is
        # needed for sequences.
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


def train(cfg: PacingConfig) -> PacingLSTM:
    torch.manual_seed(cfg.seed)

    # --- Data ------------------------------------------------------------------
    train_ds, val_ds, _ = make_seq_datasets(
        n_races=cfg.n_races, val_fraction=cfg.val_fraction, seed=cfg.seed
    )

    # DataLoader handles shuffling and batching; num_workers=0 is safe on Windows
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} races   Val: {len(val_ds)} races")

    # --- Model -----------------------------------------------------------------
    model = PacingLSTM(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # See train_finish.py for why ReduceLROnPlateau rather than a fixed
    # schedule, why lr_patience must stay below the early-stopping one, and
    # why this is off by default.
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
        )
        if cfg.lr_schedule
        else None
    )

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

        # Stepped on val loss, not train — see train_finish.py.
        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                print(f"Epoch {epoch:3d}: LR {prev_lr:.2e} -> {new_lr:.2e}")

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
    # Targets are raw min/km, so RMSE reads as "average pace error per segment".
    print(f"Best val RMSE: {best_val ** 0.5:.4f} min/km")
    print(f"Checkpoint: {checkpoint_path}")

    # --- Loss curve ------------------------------------------------------------
    plot_path = pathlib.Path(cfg.plot_path)
    plot_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM pacing model — loss over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Loss curve: {plot_path}")

    # Load best weights back before returning
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model


if __name__ == "__main__":
    cfg = PacingConfig()
    train(cfg)
