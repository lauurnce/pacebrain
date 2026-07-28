"""
Neural network model definitions.
"""

from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron for regression.

    nn.Module is the base class for every neural network in PyTorch.
    You define layers in __init__ and describe one forward pass in forward().
    PyTorch handles backward() automatically via autograd.

    Args:
        input_size:   number of input features
        hidden_sizes: list of hidden layer widths, e.g. [64, 32]
        output_size:  number of outputs (1 for single-value regression)
        dropout:      dropout probability applied after each hidden layer (0 = off)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()   # always call super().__init__() first

        # Annotated because the first append would otherwise fix the element
        # type as nn.Linear, making every ReLU and Dropout after it a type
        # error. The list genuinely holds mixed nn.Module subclasses.
        layers: list[nn.Module] = []
        prev = input_size
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))   # y = xW^T + b
            layers.append(nn.ReLU())                # zero out negatives
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, output_size))

        # nn.Sequential wraps a list of layers into one callable module
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One forward pass. Called by model(x), not model.forward(x)."""
        return self.net(x)


class FinishTimePredictor(MLP):
    """
    Thin wrapper around MLP wired to the 6 running features in data.py.

    Separating this from the generic MLP makes checkpoints self-documenting
    and simplifies loading in inference code later.

    Args:
        input_size:   number of input features; defaults to N_FEATURES.
                      Mirrors PacingLSTM, which takes the same argument so
                      train_pacing.py can drive it from PacingConfig.
        hidden_sizes: list of hidden layer widths (default [64, 32])
        dropout:      dropout probability after each hidden layer
    """

    N_FEATURES = 6  # must stay in sync with len(FEATURE_COLS) in data.py

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_sizes: Optional[list] = None,
        dropout: float = 0.1,
    ):
        if input_size is None:
            input_size = self.N_FEATURES
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            dropout=dropout,
        )
