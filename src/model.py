"""src/model.py
Model factory now provides:
  • MLP – for tabular / synthetic data
  • ResNet-18 – baseline CNN for CIFAR-10/100 experiments (downloaded from
    torchvision, last FC adapted to the requested number of classes)
Additional architectures can be added following the same pattern.
"""
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) or (B, C, H, W) flattened outside
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.net(x)


def _resnet18(num_classes: int):
    model = models.resnet18(weights=None)  # no ImageNet pre-training to stay fair
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ----------------------- model factory ---------------------------------------

def get_model(name: str, input_dim: int, num_classes: int, cfg: Dict[str, Any]):
    """Return an instantiated torch.nn.Module configured for this run."""
    name = name.lower()

    if name == "mlp":
        hidden = int(cfg.get("hidden_dim", 64))
        dropout = float(cfg.get("dropout", 0.0))
        return MLP(input_dim, num_classes, hidden_dim=hidden, dropout=dropout)

    if name == "resnet18":
        return _resnet18(num_classes)

    raise NotImplementedError(f"Model '{name}' is not implemented.  Available: mlp, resnet18.")
