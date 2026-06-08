"""
Data utilities — synthetic generator (Day 2) and real RunningDataset (Day 3+).
"""

import torch
import numpy as np


def make_synthetic_data(
    n_samples: int = 500,
    n_features: int = 4,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic regression dataset.

    The true relationship is a weighted sum of the features plus a
    nonlinear interaction term, so the MLP has something non-trivial to fit.

    Returns:
        X: shape (n_samples, n_features)  — input features, standardized
        y: shape (n_samples, 1)           — regression target
    """
    rng = np.random.default_rng(seed)

    X_np = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # True target: linear combo + one nonlinear term + noise
    weights = np.array([2.0, -1.5, 0.8, 1.2], dtype=np.float32)[:n_features]
    y_np = X_np @ weights + 0.5 * X_np[:, 0] ** 2
    y_np += rng.standard_normal(n_samples).astype(np.float32) * noise
    y_np = y_np.reshape(-1, 1)

    return torch.from_numpy(X_np), torch.from_numpy(y_np)


def train_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split tensors into train and validation sets."""
    n = len(X)
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=rng)

    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
