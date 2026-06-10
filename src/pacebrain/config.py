"""
Hyperparameters for the finish-time predictor.

Keeping config separate from training code makes it easy to tune values
without touching model or loop logic.
"""

from dataclasses import dataclass, field


@dataclass
class FinishPredictorConfig:
    # Model architecture
    input_size: int = 6          # must match len(FEATURE_COLS) in data.py
    hidden_sizes: list = field(default_factory=lambda: [64, 32])
    output_size: int = 1
    dropout: float = 0.1

    # Optimiser
    lr: float = 1e-3

    # Training schedule
    epochs: int = 300
    batch_size: int = 32

    # Early stopping: halt if val loss doesn't improve for this many epochs.
    # Prevents overfitting without hand-tuning the epoch count.
    patience: int = 25

    # Data
    n_samples: int = 1000        # synthetic rows when no real CSV is provided
    val_fraction: float = 0.2
    seed: int = 42

    # Paths
    checkpoint_path: str = "models/finish_predictor.pt"
    plot_path: str = "reports/day4_loss_curve.png"
