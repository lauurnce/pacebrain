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

    # Learning-rate schedule (ReduceLROnPlateau): when val loss stalls for
    # lr_patience epochs, multiply the LR by lr_factor.
    #
    # OFF by default, because it was measured and does not earn its place —
    # see reports/lr_schedule_experiment.md. Aggressive decay actively hurts
    # (factor 0.5 / patience 8 gave 41.77 val MSE against 30.48 without, a
    # 37% regression: it fires on val-loss noise and shrinks the LR 32x
    # before early stopping halts it). The best setting found, 0.8 / 12, beat
    # the baseline on only 2 of 4 seeds for a mean 7.5% gain, which is a coin
    # flip rather than a result.
    #
    # Kept because it is one flag away and the experiment is worth repeating
    # on real data, where a noisier loss surface may reward it.
    #
    # If enabling: lr_patience must stay well below `patience` above. The two
    # counters race, and at 15 or 20 the single LR drop lands after the best
    # epoch and changes the outcome by exactly nothing.
    lr_schedule: bool = False
    lr_factor: float = 0.8
    lr_patience: int = 12

    # Data
    n_samples: int = 1000        # synthetic rows when no real CSV is provided
    val_fraction: float = 0.2
    seed: int = 42

    # Paths
    checkpoint_path: str = "models/finish_predictor.pt"
    plot_path: str = "reports/day4_loss_curve.png"


@dataclass
class PacingConfig:
    # Model architecture
    input_size: int = 6          # must match len(SEQ_FEATURES) in seq_data.py
    hidden_size: int = 64        # width of the LSTM hidden state
    num_layers: int = 1          # stacked LSTM layers
    dropout: float = 0.0         # only applies between stacked layers (num_layers > 1)

    # Optimiser
    lr: float = 1e-3

    # Training schedule
    epochs: int = 200
    batch_size: int = 32

    # Early stopping: halt if val loss doesn't improve for this many epochs.
    patience: int = 20

    # Learning-rate schedule — off by default for the same reason as the
    # finish predictor; see the note there and reports/lr_schedule_experiment.md.
    lr_schedule: bool = False
    lr_factor: float = 0.8
    lr_patience: int = 6

    # Data
    n_races: int = 600           # synthetic races from make_sample_sequences()
    val_fraction: float = 0.2
    seed: int = 42

    # Paths
    checkpoint_path: str = "models/pacing_model.pt"
    plot_path: str = "reports/day7_loss_curve.png"
