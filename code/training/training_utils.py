"""
Simple training utilities for Robin Dale's hybrid model approach.
Just the essentials - loss functions and basic metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    """Root Mean Square Error loss following Robin Dale's approach"""
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(input, target)
        return torch.sqrt(mse)


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Square Error"""
    return torch.sqrt(F.mse_loss(pred, target)).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error"""
    return F.l1_loss(pred, target).item()