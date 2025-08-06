#!/usr/bin/env python3
"""
Simple Training Configuration for NIR-DOT Reconstruction Pipeline.

This module centralizes all training hyperparameters in one place for easy tuning
and consistency across Stage 1 and Stage 2 training.

USAGE:
    To run Stage 1: Set CURRENT_TRAINING_STAGE = "stage1" and run train_hybrid_model.py
    To run Stage 2: Set CURRENT_TRAINING_STAGE = "stage2" and run train_hybrid_model.py
    
    All other hyperparameters can be modified directly in this file.

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# TRAINING CONTROL - SET WHICH STAGE TO RUN
# =============================================================================

# Training Control - Set which stage to run
CURRENT_TRAINING_STAGE = "stage1"       # Set to "stage1" or "stage2" to control which stage runs
STAGE1_CHECKPOINT_PATH = "checkpoints/stage1_best.pth"  # Path to Stage 1 checkpoint for Stage 2

# W&B Control
USE_WANDB_LOGGING = True                # Enable/disable Weights & Biases logging

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Learning Rates
LEARNING_RATE_STAGE1 = 5e-5             # CNN autoencoder learning rate
LEARNING_RATE_STAGE2 = 3e-5             # Transformer learning rate (lower for stability)

# Training Duration
EPOCHS_STAGE1 = 50                      # Increase for better convergence (was 10)
EPOCHS_STAGE2 = 100                     # Default epochs for Stage 2

# Batch Sizes
BATCH_SIZE_STAGE1 = 4                   # Stage 1: CNN autoencoder (increased for better generalization)
BATCH_SIZE_STAGE2 = 4                   # Stage 2: Transformer (increased for stable training)

# Regularization
WEIGHT_DECAY = 1e-4                     # L2 regularization strength
DROPOUT_RATE = 0.15                     # Dropout probability (for future use)
EARLY_STOPPING_PATIENCE = 8            # Early stopping patience (epochs)

# Learning Rate Scheduling
LR_SCHEDULER_PATIENCE = 5               # Learning rate scheduler patience  
LR_SCHEDULER_FACTOR = 0.5               # Learning rate reduction factor
LR_MIN = 1e-7                          # Minimum learning rate

# Progress Logging
PROGRESS_LOG_INTERVAL = 10              # Log progress every N epochs
BATCH_LOG_INTERVAL = 5                  # Detailed logging every N batches
FINAL_EPOCH_OFFSET = 1                  # Offset for final epoch logging

# Checkpoint Configuration
CHECKPOINT_BASE_DIR = "checkpoints"     # Base checkpoint directory
CHECKPOINT_STAGE1 = "stage1_best.pth"  # Stage 1 checkpoint filename
CHECKPOINT_STAGE2_BASELINE = "stage2_baseline_best.pth"  # Stage 2 baseline
CHECKPOINT_STAGE2_ENHANCED = "stage2_enhanced_best.pth"  # Stage 2 enhanced

# W&B Configuration
WANDB_PROJECT = "nir-dot-reconstruction"     # Unified project name
LOG_IMAGES_EVERY = 1                         # Log reconstruction images every N epochs
LOG_BOTH_CHANNELS = True                     # Log both absorption and scattering channels
WANDB_TAGS_STAGE1 = ["stage1", "cnn-autoencoder", "pretraining", "nir-dot"]
WANDB_TAGS_STAGE2_BASELINE = ["stage2", "transformer", "baseline", "nir-dot"]
WANDB_TAGS_STAGE2_ENHANCED = ["stage2", "transformer", "enhanced", "tissue-patches", "nir-dot"]

# Data Configuration
DATA_DIRECTORY = "data"                 # Default data directory path

# Device Configuration
CUDA_DEVICE = "cuda"                    # CUDA device identifier
CPU_DEVICE = "cpu"                      # CPU device identifier

# Stage Configuration
TRAINING_STAGE1 = "stage1"              # Stage 1 identifier
TRAINING_STAGE2 = "stage2"              # Stage 2 identifier
USE_TISSUE_PATCHES_STAGE1 = False       # Stage 1 doesn't use tissue patches
USE_TISSUE_PATCHES_STAGE2 = True        # Stage 2 default tissue patch usage (set to False for baseline mode)

# Mode Configuration
BASELINE_MODE = "Baseline"              # Baseline training mode name
ENHANCED_MODE = "Enhanced"              # Enhanced training mode name
