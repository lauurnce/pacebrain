"""
Neural network model definitions.
"""

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

        layers = []
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
    """

    N_FEATURES = 6  # must stay in sync with len(FEATURE_COLS) in data.py

    def __init__(self, hidden_sizes: list[int] = None, dropout: float = 0.1):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        super().__init__(
            input_size=self.N_FEATURES,
            hidden_sizes=hidden_sizes,
            output_size=1,
            dropout=dropout,
        )
