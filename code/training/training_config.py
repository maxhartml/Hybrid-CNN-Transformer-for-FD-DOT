#!/usr/bin/env python3
"""
Training Configuration for NIR-DOT Reconstruction Pipeline

This module centralizes all training hyperparameters in one place for easy tuning
and consistency across Stage 1 (CNN autoencoder) and Stage 2 (transformer) training.

USAGE:
    To run Stage 1: Set CURRENT_TRAINING_STAGE = "stage1" and run train_hybrid_model.py
    To run Stage 2: Set CURRENT_TRAINING_STAGE = "stage2" and run train_hybrid_model.py

Author: Max Hart
Date: August 2025
"""

import torch
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# =============================================================================
# EXPERIMENT CONTROL
# =============================================================================

# Training Stage Control - Set which stage to run
CURRENT_TRAINING_STAGE = "stage1"  # "stage1" or "stage2"

# Weights & Biases Logging
USE_WANDB_LOGGING = True

# Debug Logging Control
DEBUG_VERBOSE = False                   # Set to True to enable verbose debug logging (attention, gradients, etc.)

# Latent-only training control (Stage 2)
TRAIN_STAGE2_LATENT_ONLY = True         # Train on latent RMSE only (no decoder during training)
LATENT_DIM = 256                        # Latent dimension for teacher-student matching

# Validation cadence for end-to-end metrics
VAL_E2E_EVERY_K_EPOCHS = 1              # Decode + raw metrics every K epochs

# Tissue patch gating for Stage 2
USE_TISSUE_PATCHES_STAGE2 = False       # Baseline: no patch extraction (set True for enhanced mode)

# Performance Optimizations
USE_MODEL_COMPILATION = True            # PyTorch 2.0 compilation for 2x speedup (fixed compilation issues)
COMPILATION_MODE = "default"            # Clean compilation without aggressive autotune noise
USE_CHANNELS_LAST_MEMORY_FORMAT = True  # Efficient memory layout for 3D convolutions

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training Duration
EPOCHS_STAGE1 = 150  # Stage 1 CNN training epochs - more (↑) = better feature learning, less (↓) = faster training
EPOCHS_STAGE2 = 200   # Stage 2 transformer epochs - more (↑) = better fine-tuning, less (↓) = faster completion

# Batch Sizes - Hard-coded for stability
BATCH_SIZE = 256  # Increased for better utilization - lower gradient noise and better GPU usage

# Early Stopping
EARLY_STOPPING_PATIENCE = 25  # Epochs to wait without improvement - higher (↑) = more exploration, lower (↓) = faster stopping

# =============================================================================
# Data Loading Configuration
# Optimized for large 3D medical imaging data (64x64x64 phantoms)

NUM_WORKERS = 16          # Increased for better throughput - more parallel loading to feed transformer
PIN_MEMORY = True        # Pin memory for faster GPU transfer - True = faster but uses more system memory
PREFETCH_FACTOR = 4      # Increased for better pipeline utilization - 16 workers × 4 = 64 batches prefetched
PERSISTENT_WORKERS = True # Keep workers alive between epochs - True = faster epoch transitions

# =============================================================================
# REGULARIZATION PARAMETERS
# =============================================================================

# Weight Decay (L2 regularization)
WEIGHT_DECAY = 7e-4             # CNN weight decay - higher (↑) = less overfitting but may underfit, lower (↓) = more capacity but overfitting risk
WEIGHT_DECAY_TRANSFORMER = 0.01  # Reduced transformer weight decay - less shrinkage for better capacity

# Dropout Rates (prevent overfitting)
DROPOUT_CNN = 0.12              # CNN dropout rate - higher (↑) = stronger regularization, lower (↓) = more model capacity
DROPOUT_TRANSFORMER = 0.05      # Reduced dropout - more capacity since not overfitting
DROPOUT_NIR_PROCESSOR = 0.08    # Reduced NIR dropout - less input masking for better signal learning

# Gradient Clipping (training stability)
GRADIENT_CLIP_MAX_NORM = 1.0      # Relaxed clipping - avoid over-clipping healthy updates
GRADIENT_MONITOR_THRESHOLD = 5.0  # Updated from 1.0 - less over-sensitive warnings

# AMP GradScaler Configuration (prevents scaling issues and crashes)
GRADSCALER_INIT_SCALE = 2**8            # Less conservative initial scale (256) for better gradient flow
GRADSCALER_GROWTH_FACTOR = 2.0          # Moderate growth rate
GRADSCALER_BACKOFF_FACTOR = 0.5         # Scale reduction when inf/nan detected
GRADSCALER_GROWTH_INTERVAL = 200        # Shorter interval for quicker scale growth

# =============================================================================
# STAGE 1: CNN AUTOENCODER TRAINING (OneCycleLR)
# =============================================================================
# Based on "Super-Convergence" paper (Smith, 2018) for optimal CNN training

# Learning Rate Schedule
STAGE1_MAX_LR = 3e-3          # Peak learning rate - higher values speed up training but risk instability
STAGE1_BASE_LR = 1.0e-4         # Starting/ending learning rate - lower values provide smoother convergence
STAGE1_DIV_FACTOR = 25          # Initial LR division factor - controls how low we start (max_lr/div_factor)
STAGE1_FINAL_DIV_FACTOR = 100   # Final LR reduction factor - higher values give gentler final decay
STAGE1_PCT_START = 0.30         # Warmup phase percentage - more warmup (↑) = more stable but slower start
STAGE1_CYCLE_MOMENTUM = True    # Enable momentum cycling for CNN - helps escape local minima

# Optimizer Parameters
ADAMW_BETAS_STAGE1 = (0.9, 0.95)  # Adam momentum parameters - beta1 (↑) = more momentum, beta2 (↑) = smoother updates
ADAMW_EPS_STAGE1 = 1e-8            # Numerical stability epsilon - prevents division by zero in optimizer

# Momentum Cycling (OneCycleLR feature)
BASE_MOMENTUM = 0.85  # Minimum momentum value - higher (↑) = more stability, lower (↓) = faster adaptation
MAX_MOMENTUM = 0.95   # Maximum momentum value - cycles between base and max during training

# =============================================================================
# STAGE 2: TRANSFORMER TRAINING (Linear Warmup + Cosine Decay)
# =============================================================================
# Based on "Attention Is All You Need", BERT, and ViT papers for transformer training

# Learning Rate Schedule
STAGE2_BASE_LR = 1.5e-4   # Higher peak LR to move transformer weights effectively  
STAGE2_WARMUP_PCT = 0.06  # Longer warmup for smoother ramp - reduces early volatility
STAGE2_ETA_MIN_PCT = 0.15 # Higher LR floor - keeps learning active late in training

# Optimizer Parameters
ADAMW_BETAS_STAGE2 = (0.9, 0.98)  # Transformer-optimized momentum - beta2=0.98 reduces noise in attention gradients
ADAMW_EPS_STAGE2 = 1e-8            # Numerical stability epsilon - prevents optimizer mathematical errors

# =============================================================================
# CHECKPOINT AND LOGGING CONFIGURATION
# =============================================================================

# Checkpoint Paths
CHECKPOINT_BASE_DIR = "checkpoints"
CHECKPOINT_STAGE1 = "stage1_best.pth"
CHECKPOINT_STAGE2_BASELINE = "stage2_baseline_best.pth"
CHECKPOINT_STAGE2_ENHANCED = "stage2_enhanced_best.pth"

# Full checkpoint paths (for easy import)
STAGE1_CHECKPOINT_PATH = f"{CHECKPOINT_BASE_DIR}/{CHECKPOINT_STAGE1}"
STAGE2_BASELINE_CHECKPOINT_PATH = f"{CHECKPOINT_BASE_DIR}/{CHECKPOINT_STAGE2_BASELINE}"
STAGE2_ENHANCED_CHECKPOINT_PATH = f"{CHECKPOINT_BASE_DIR}/{CHECKPOINT_STAGE2_ENHANCED}"

# Logging Configuration
LOG_LR_EVERY_N_BATCHES = 5      # Log learning rate every N batches
PROGRESS_LOG_INTERVAL = 5       # Log detailed epoch summary every N epochs
FINAL_EPOCH_OFFSET = 1          # Always log final epoch

# Weights & Biases Configuration
WANDB_PROJECT = "nir-dot-reconstruction"
WANDB_TAGS_STAGE1 = ["stage1", "cnn-autoencoder", "pretraining", "nir-dot"]
WANDB_TAGS_STAGE2_BASELINE = ["stage2", "transformer", "baseline", "nir-dot"]
WANDB_TAGS_STAGE2_ENHANCED = ["stage2", "transformer", "enhanced", "tissue-patches", "nir-dot"]

# =============================================================================
# STAGE-SPECIFIC CONFIGURATION
# =============================================================================

# Data and Device Paths
DATA_DIRECTORY = "data"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

# Stage 1 Configuration (CNN Autoencoder Pre-training)
TRAINING_STAGE1 = "stage1"
USE_TISSUE_PATCHES_STAGE1 = False   # No tissue patches in Stage 1

# Stage 2 Configuration (Transformer Enhancement)
TRAINING_STAGE2 = "stage2"
USE_TISSUE_PATCHES_STAGE2 = False    # Enable tissue patches in Stage 2

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_gpu_stats():
    """Log GPU memory usage and utilization for monitoring."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"🖥️  GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        logger.info(f"📊 GPU Utilization: {allocated/total*100:.1f}%")
        
        # Memory usage warnings
        if allocated/total > 0.9:
            logger.warning("⚠️ GPU memory usage >90% - consider reducing batch size")
        elif reserved/total > 0.8:
            logger.warning("⚠️ GPU memory reservation >80% - monitor for potential issues")