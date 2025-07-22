"""
Training components for NIR-DOT reconstruction pipeline.

This package implements a comprehensive two-stage training pipeline for
hybrid CNN-Transformer models, designed for near-infrared diffuse optical
tomography (NIR-DOT) image reconstruction. The pipeline follows a progressive
approach with CNN autoencoder pre-training followed by transformer enhancement.

Training Stages:
- Stage 1: CNN autoencoder pre-training for feature extraction and basic reconstruction
- Stage 2: Transformer training with frozen CNN decoder for enhanced spatial modeling

Components:
- Stage1Trainer: CNN autoencoder pre-training with RMSE loss optimization
- Stage2Trainer: Transformer training with tissue context integration
- train_hybrid_model: Complete training pipeline orchestration and experiment management

Features:
- Progressive training strategy for stable convergence
- Comprehensive logging and experiment tracking
- Flexible configuration for different model variants
- Automated checkpoint management and model selection
"""

from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer

__all__ = ['Stage1Trainer', 'Stage2Trainer']
