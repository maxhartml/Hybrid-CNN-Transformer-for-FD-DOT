"""
Simple training scripts for Robin Dale's two-stage hybrid CNN-Transformer approach.
Implements both stage 1 (CNN autoencoder) and stage 2 (transformer) training.
"""

from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer
from .training_utils import RMSELoss, compute_rmse, compute_mae

__all__ = [
    'Stage1Trainer',
    'Stage2Trainer', 
    'TrainingConfig',
    'TrainingUtils'
]
