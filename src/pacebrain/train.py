"""
Training loop for the MLP on synthetic data.

Run:  python -m pacebrain.train   (from repo root with src/ on PYTHONPATH)
  or: python src/pacebrain/train.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pacebrain.models import MLP
from pacebrain.data import make_synthetic_data, train_val_split


# ----- Config -------------------------------------------------------------
INPUT_SIZE = 4
HIDDEN_SIZES = [64, 32]
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 64
CHECKPOINT_PATH = pathlib.Path("models/mlp_synthetic.pt")
PLOT_PATH = pathlib.Path("reports/day2_loss_curve.png")


def make_batches(X, y, batch_size):
    """Yield (X_batch, y_batch) pairs, shuffled each call."""
    n = len(X)
    idx = torch.randperm(n)
    for start in range(0, n, batch_size):
        batch = idx[start : start + batch_size]
        yield X[batch], y[batch]


def train_one_epoch(model, X_train, y_train, loss_fn, optimizer):
    """
    model.train() switches on dropout / batch-norm training behaviour.
    For each mini-batch: zero grads -> forward -> loss -> backward -> step.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in make_batches(X_train, y_train, BATCH_SIZE):
        optimizer.zero_grad()           # clear gradients from last step
        preds = model(X_batch)          # forward pass — calls model.forward()
        loss = loss_fn(preds, y_batch)  # MSE: mean((pred - target)^2)
        loss.backward()                 # compute gradients
        optimizer.step()               # update weights: w -= lr * grad
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(X_train)


def evaluate(model, X_val, y_val, loss_fn):
    """
    model.eval() disables dropout so every run is deterministic.
    torch.no_grad() skips building the computation graph — faster, less memory.
    """
    model.eval()
    with torch.no_grad():
        preds = model(X_val)
        loss = loss_fn(preds, y_val)
    return loss.item()


def main():
    # Data
    X, y = make_synthetic_data(n_samples=500, n_features=INPUT_SIZE)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # Model, loss, optimizer
    model = MLP(input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, output_size=1)
    loss_fn = nn.MSELoss()
    # Adam adapts the learning rate per parameter — usually converges faster than plain SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {len(X_train)}  Val samples: {len(X_val)}\n")

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, X_train, y_train, loss_fn, optimizer)
        val_loss = evaluate(model, X_val, y_val, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}"
                  + ("  *" if val_loss == best_val else ""))

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    # Plot
    PLOT_PATH.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP on synthetic data — loss over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Loss curve saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()
