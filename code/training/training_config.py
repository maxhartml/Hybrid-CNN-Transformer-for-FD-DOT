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
# GLOBAL SETTINGS - SINGLE SOURCE OF TRUTH
# =============================================================================

# Global Random Seed - Controls all random number generation for reproducibility
GLOBAL_SEED = 1337                      # Master seed for reproducible experiments

# Global Architecture Settings
GLOBAL_POOLING_QUERIES = 4               # Number of pooling queries for transformer multi-query pooling

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

# Architecture dimensions (consistent across pipeline)
LATENT_DIM = 256                        # Latent dimension for teacher-student matching
EMBED_DIM = 256                         # Transformer embedding dimension  
ENCODED_SCAN_DIM = 256                  # Global pooling output dimension

# Volume and measurement dimensions
VOLUME_SHAPE = (64, 64, 64)             # Volume dimensions (D, H, W) in voxels
N_MEASUREMENTS = 256                    # Number of measurements per phantom (subsampled from 1000)

# Validation cadence for end-to-end metrics
VAL_E2E_EVERY_K_EPOCHS = 1              # Decode + raw metrics every K epochs

# Tissue patch gating for Stage 2
USE_TISSUE_PATCHES_STAGE2 = False       # Baseline: no patch extraction (set True for enhanced mode)

# Exponential Moving Average (EMA) Configuration - Progressive Decay Ramp
USE_EMA = True                          # Enable EMA of model weights for better generalization
EMA_DECAY = 0.997                       # Baseline EMA decay factor (maintained for compatibility)
EMA_DECAY_START = 0.995                 # Starting EMA decay value (more aggressive early learning)
EMA_DECAY_END = 0.9995                  # Final EMA decay value (more stable late training)

# Decoder Fine-tuning Control - Increased for Preset B
UNFREEZE_LAST_DECODER_BLOCK = True      # Allow fine-tuning of final decoder block in Stage 2
DECODER_FINETUNING_LR_SCALE = 0.3       # Increased LR scaling for unfrozen decoder block (relative to transformer LR)

# Attention Entropy Regularization
ATTENTION_ENTROPY_LAMBDA_BASE = 1e-4         # Regularization weight for attention entropy - encourages diverse attention patterns
ATTENTION_ENTROPY_LAMBDA = 0.0 if TRAIN_STAGE2_LATENT_ONLY else ATTENTION_ENTROPY_LAMBDA_BASE

# Performance Optimizations
USE_MODEL_COMPILATION = True            # PyTorch 2.0 compilation for 2x speedup (fixed compilation issues)
COMPILATION_MODE = "default"            # Clean compilation without aggressive autotune noise
USE_CHANNELS_LAST_MEMORY_FORMAT = True  # Efficient memory layout for 3D convolutions

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training Duration
EPOCHS_STAGE1 = 200  # Stage 1 CNN training epochs - more (↑) = better feature learning, less (↓) = faster training
EPOCHS_STAGE2 = 300   # Stage 2 transformer epochs - increased for better convergence, more (↑) = better fine-tuning, less (↓) = faster completion

# Batch Sizes - Hard-coded for stability
BATCH_SIZE = 128  # Reduced for better gradient diversity and memory efficiency

# Early Stopping
EARLY_STOPPING_PATIENCE = 50  # Epochs to wait without improvement - higher (↑) = more exploration, lower (↓) = faster stopping

# ===================================== Data Loading Configuration =========================================
# Optimized for large 3D medical imaging data (64x64x64 phantoms)

NUM_WORKERS = 16          # Increased for better throughput - more parallel loading to feed transformer
PIN_MEMORY = True        # Pin memory for faster GPU transfer - True = faster but uses more system memory
PREFETCH_FACTOR = 4      # Increased for better pipeline utilization - 16 workers × 4 = 64 batches prefetched
PERSISTENT_WORKERS = True # Keep workers alive between epochs - True = faster epoch transitions

# =============================================================================
# REGULARIZATION PARAMETERS
# =============================================================================

# Weight Decay (L2 regularization)
WEIGHT_DECAY = 1e-3             # CNN weight decay - higher (↑) = less overfitting but may underfit, lower (↓) = more capacity but overfitting risk
WEIGHT_DECAY_TRANSFORMER = 0.01  # Further reduced transformer weight decay for better capacity

# Dropout Rates (prevent overfitting)
DROPOUT_CNN = 0.05              # CNN dropout rate - higher (↑) = stronger regularization, lower (↓) = more model capacity
DROPOUT_TRANSFORMER = 0.05      # Further reduced dropout for more capacity since not overfitting
DROPOUT_NIR_PROCESSOR = 0.04    # Further reduced NIR dropout for better signal learning

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
STAGE1_MAX_LR = 2e-3          # Peak learning rate - higher values speed up training but risk instability
STAGE1_BASE_LR = 1.0e-4         # Starting/ending learning rate - lower values provide smoother convergence
STAGE1_DIV_FACTOR = 20          # Initial LR division factor - controls how low we start (max_lr/div_factor)
STAGE1_FINAL_DIV_FACTOR = 200   # Final LR reduction factor - higher values give gentler final decay
STAGE1_PCT_START = 0.40         # Warmup phase percentage - more warmup (↑) = more stable but slower start
STAGE1_CYCLE_MOMENTUM = True    # Enable momentum cycling for CNN - helps escape local minima

# Optimizer Parameters
ADAMW_BETAS_STAGE1 = (0.9, 0.98)  # Adam momentum parameters - beta1 (↑) = more momentum, beta2 (↑) = smoother updates
ADAMW_EPS_STAGE1 = 1e-8            # Numerical stability epsilon - prevents division by zero in optimizer

# Momentum Cycling (OneCycleLR feature)
BASE_MOMENTUM = 0.85  # Minimum momentum value - higher (↑) = more stability, lower (↓) = faster adaptation
MAX_MOMENTUM = 0.95   # Maximum momentum value - cycles between base and max during training

# =============================================================================
# STAGE 2: TRANSFORMER TRAINING (Linear Warmup -> Cosine Decay with Floor)
# =============================================================================

# Learning Rate Schedule
STAGE2_BASE_LR = 3.0e-4                 # Peak LR after warmup - optimized for transformer training
STAGE2_WARMUP_PCT = 0.10                # 10% of total training steps for warmup
STAGE2_MIN_LR = 1.0e-6                  # LR floor to prevent learning stagnation

# Scheduler Configuration
SCHEDULER_START_FACTOR = 0.01           # Start warmup at 1% of base to avoid zero-LR AdamW issues

# Optimizer Parameters
ADAMW_BETAS_STAGE2 = (0.9, 0.98)       # Transformer-optimized momentum - beta2=0.98 reduces noise in attention gradients
ADAMW_EPS_STAGE2 = 1e-8                 # Numerical stability epsilon - prevents optimizer mathematical errors

# =============================================================================
# CHECKPOINT AND LOGGING CONFIGURATION
# =============================================================================

# Checkpoint Management System
# The new checkpoint system saves timestamped files to prevent overwrites and
# automatically selects the best model based on validation loss.
#
# Filename Format: checkpoint_{stage}_{timestamp}_loss{val_loss:.4f}.pt
# Example: checkpoint_stage1_20250817-201530_loss0.3917.pt
#
# Benefits:
# - No accidental overwrites of good checkpoints
# - Automatic best model selection for Stage 2 initialization
# - Complete training history preservation
# - Robust filename parsing for loss extraction

CHECKPOINT_BASE_DIR = "checkpoints"

# New checkpoint system uses per-run filenames: checkpoint_{stage_id}_{run_id}.pt
# Automatic selection by find_best_checkpoint() reads metrics from inside files
# No more legacy path constants needed

# Logging Configuration
LOG_LR_EVERY_N_BATCHES = 5      # Log learning rate every N batches
PROGRESS_LOG_INTERVAL = 5       # Log detailed epoch summary every N epochs
FINAL_EPOCH_OFFSET = 1          # Always log final epoch

# Training loop constants
STANDARDIZER_BATCH_LOG_INTERVAL = 20    # Log every N batches during standardizer fitting
TRAINING_BATCH_LOG_INTERVAL = 10       # Log training stats every N batches  
GPU_MEMORY_LOG_INTERVAL = 5            # Log GPU memory every N epochs

# Scheduler constants
MIN_LR_FACTOR = 0.01                   # Cosine scheduler starts at 1% of base LR (not 0%)

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