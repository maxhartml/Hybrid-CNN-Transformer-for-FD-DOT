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
# IMPORTS
# =============================================================================

import torch
import psutil
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# =============================================================================
# TRAINING CONTROL - SET WHICH STAGE TO RUN
# =============================================================================

# Training Control - Set which stage to run
CURRENT_TRAINING_STAGE = "stage2"       # Set to "stage1" or "stage2" to control which stage runs
STAGE1_CHECKPOINT_PATH = "checkpoints/stage1_best.pth"  # Path to Stage 1 checkpoint for Stage 2

# W&B Control
USE_WANDB_LOGGING = True                # Enable/disable Weights & Biases logging

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Learning Rates (optimized for two-stage training)
LEARNING_RATE_STAGE1 = 5e-5             # CNN autoencoder learning rate (higher for initial feature learning)
LEARNING_RATE_STAGE2 = 3e-5             # Transformer learning rate (lower for stable fine-tuning on frozen CNN)

# Training Duration
EPOCHS_STAGE1 = 50                      # Increased for better convergence 
EPOCHS_STAGE2 = 50                     # Default epochs for Stage 2

# Batch Sizes - CONSISTENT ACROSS BOTH STAGES & AUTO-DETECTED
def get_optimized_batch_size():
    """Auto-detect optimal batch size based on available hardware (same for both stages)."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory > 30:  # A100 40GB or similar
            return 64        # Increased for better stability and A100 utilization
        elif gpu_memory > 15:  # RTX 3080/4080 class
            return 32
        elif gpu_memory > 8:   # Smaller GPU
            return 16
        else:
            return 8         # Very small GPU
    else:
        return 4             # CPU fallback

# Use consistent batch size for both stages (better for comparison and simplicity)
BATCH_SIZE = get_optimized_batch_size()
BATCH_SIZE_STAGE1 = BATCH_SIZE           # Same batch size for both stages
BATCH_SIZE_STAGE2 = BATCH_SIZE           # Same batch size for both stages

# Data Loading Configuration - OPTIMIZED FOR SERVER
NUM_WORKERS = min(8, max(1, psutil.cpu_count(logical=False) - 2))  # Use most CPU cores
PIN_MEMORY = torch.cuda.is_available()  # Enable GPU memory pinning if CUDA available
PREFETCH_FACTOR = 4 if torch.cuda.is_available() else 2  # More prefetching on GPU systems

# Regularization and Optimization
WEIGHT_DECAY = 1e-4                     # L2 regularization strength (CNN standard)
WEIGHT_DECAY_TRANSFORMER = 0.01         # Higher weight decay for transformer (standard)
DROPOUT_RATE = 0.15                     # Dropout probability (for future use)
EARLY_STOPPING_PATIENCE = 8            # Early stopping patience (epochs)

# AdamW Optimizer Configuration
# Stage 1 (CNN Autoencoder): Stability-focused parameters
ADAMW_BETAS_STAGE1 = (0.9, 0.95)       # Slightly lower beta2 for CNN stability
ADAMW_EPS_STAGE1 = 1e-8                # Numerical stability epsilon

# Stage 2 (Transformer): Transformer-standard parameters
ADAMW_BETAS_STAGE2 = (0.9, 0.98)       # Transformer-standard betas (BERT/ViT)
ADAMW_EPS_STAGE2 = 1e-8                # Numerical stability epsilon

# Gradient Clipping Configuration
GRADIENT_CLIP_MAX_NORM = 1.0            # Maximum gradient norm for clipping (prevents explosion)
GRADIENT_MONITOR_THRESHOLD = 10.0       # Log warning if gradient norm exceeds this

# Stage 1 OneCycleLR Configuration (Research-Validated)
# Based on "Super-Convergence" (Smith, 2018) and medical imaging best practices
STAGE1_MAX_LR = 3e-3                    # Peak learning rate (found via LR range test)
STAGE1_BASE_LR = 1e-3                   # Base learning rate (max_lr / div_factor)
STAGE1_DIV_FACTOR = 25                  # Conservative div_factor for stability
STAGE1_FINAL_DIV_FACTOR = 1e4           # Strong final decay for polishing
STAGE1_PCT_START = 0.2                  # 20% warmup (conservative, proven)
STAGE1_CYCLE_MOMENTUM = True            # Enable momentum cycling for CNN

# Stage 2 Linear Warmup + Cosine Decay Configuration (BERT/ViT Standard)
# Based on "Attention Is All You Need", BERT, and ViT papers
STAGE2_BASE_LR = 2e-4                   # Base learning rate (conservative for fine-tuning)
STAGE2_WARMUP_PCT = 0.1                 # 10% warmup (transformer standard)
STAGE2_ETA_MIN_PCT = 0.03               # Final LR = 3% of peak (smooth convergence)

# Momentum Cycling Parameters (Stage 1 only)
BASE_MOMENTUM = 0.85                    # Base momentum value
MAX_MOMENTUM = 0.95                     # Maximum momentum value

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
USE_TISSUE_PATCHES_STAGE2 = False       # Stage 2 BASELINE mode (no tissue patches)

# Mode Configuration
BASELINE_MODE = "Baseline"              # Baseline training mode name
ENHANCED_MODE = "Enhanced"              # Enhanced training mode name

# =============================================================================
# GPU UTILITIES AND MONITORING
# =============================================================================

def log_gpu_stats():
    """Log GPU memory usage during training."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"🖥️  GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        logger.info(f"📊 GPU Utilization: {allocated/total*100:.1f}%")
        
        if allocated/total > 0.9:
            logger.warning("⚠️ GPU memory usage >90% - consider reducing batch size")
        elif reserved/total > 0.8:
            logger.warning("⚠️ GPU memory reservation >80% - monitor for potential issues")