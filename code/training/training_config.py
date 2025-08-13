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
CURRENT_TRAINING_STAGE = "stage1"       # Set to "stage1" or "stage2" to control which stage runs
STAGE1_CHECKPOINT_PATH = "checkpoints/stage1_best.pth"  # Path to Stage 1 checkpoint for Stage 2

# W&B Control
USE_WANDB_LOGGING = True                # Enable/disable Weights & Biases logging

# Progress Logging Control
PROGRESS_LOG_INTERVAL = 1               # Log progress every N epochs
FINAL_EPOCH_OFFSET = 1                  # Always log final epoch (epochs - 1)

# Performance Optimization Configuration
USE_MODEL_COMPILATION = True            # Enable PyTorch 2.0 compilation for 2x speedup
COMPILATION_MODE = "default"            # Options: "default", "reduce-overhead", "max-autotune"
USE_CHANNELS_LAST_MEMORY_FORMAT = True  # More efficient memory layout for 3D convolutions

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training Duration
EPOCHS_STAGE1 = 100                     # Extended to 100 - let early stopping decide when to stop
EPOCHS_STAGE2 = 20                      # Default epochs for Stage 2

# Batch Sizes - CONSISTENT ACROSS BOTH STAGES & AUTO-DETECTED
def get_optimized_batch_size():
    """Auto-detect optimal batch size based on available hardware (same for both stages)."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory > 30:  # A100 40GB or similar
            return 64        # Increased for better stability and A100 utilization
        elif gpu_memory > 15:  # RTX 3080/4080 class
            return 64
        elif gpu_memory > 8:   # Smaller GPU
            return 16
        else:
            return 8         # Very small GPU
    else:
        return 4             # CPU fallback

# Use consistent batch size for both stages (better for comparison and simplicity)
BATCH_SIZE = get_optimized_batch_size()

# Data Loading Configuration - OPTIMIZED FOR STABILITY + PERFORMANCE
NUM_WORKERS = min(8, max(4, psutil.cpu_count(logical=False) // 4))  # Reduced from 16 - more stable for 3D data
PIN_MEMORY = torch.cuda.is_available()  # Enable GPU memory pinning if CUDA available
PREFETCH_FACTOR = 4 if torch.cuda.is_available() else 2  # Reduced from 8 - balanced performance/memory
PERSISTENT_WORKERS = True               # Keep workers alive between epochs

# Regularization and Optimization - STRENGTHENED FOR 10K PHANTOMS
WEIGHT_DECAY = 5e-4                     # Increased from 1e-4 for better regularization with more data
WEIGHT_DECAY_TRANSFORMER = 0.01         # Higher weight decay for transformer (standard)
EARLY_STOPPING_PATIENCE = 15           # FIXED: Increased to 15 - more exploration time for transformer learning

# Dropout Configuration - Enhanced regularization for longer training with more data
DROPOUT_CNN = 0.15                      # Increased from 0.1 for stronger regularization with 10K phantoms
DROPOUT_TRANSFORMER = 0.1               # Dropout for transformer attention/MLP
DROPOUT_NIR_PROCESSOR = 0.15            # Dropout for NIR processor (more aggressive due to complexity)

# Gradient Clipping Configuration (CRITICAL for stable training)
GRADIENT_CLIP_MAX_NORM = 0.5            # More aggressive clipping for medical imaging stability
GRADIENT_MONITOR_THRESHOLD = 6.0        # Higher threshold for 3D medical imaging 
                                        # NOTE: High gradients (3-6) are NORMAL early in training
                                        # as model rapidly learns basic features from random initialization

# =============================================================================
# STAGE 1 ONECYCLELR SCHEDULER CONFIGURATION
# =============================================================================
# Based on "Super-Convergence" (Smith, 2018) and medical imaging best practices

# Stage 1 Learning Rate Schedule (OneCycleLR) - TRULY OPTIMAL FOR 10K PHANTOMS
STAGE1_MAX_LR = 3e-3                    # OPTIMAL: Higher peak for effective learning (matches successful runs)
STAGE1_BASE_LR = 1.2e-4                 # Base learning rate (max_lr / div_factor = 3e-3/25) 
STAGE1_DIV_FACTOR = 25                  # Standard div_factor for stability
STAGE1_FINAL_DIV_FACTOR = 100           # Aggressive final decay (final LR ≈ 3e-05)
STAGE1_PCT_START = 0.3                  # CRITICAL FIX: 30% warmup (30 epochs) - prevents plateau bug
STAGE1_CYCLE_MOMENTUM = True            # Enable momentum cycling for CNN training

# Stage 1 AdamW Optimizer Parameters
ADAMW_BETAS_STAGE1 = (0.9, 0.95)       # Slightly lower beta2 for CNN stability
ADAMW_EPS_STAGE1 = 1e-8                # Numerical stability epsilon

# =============================================================================
# STAGE 2 LINEAR WARMUP + COSINE DECAY SCHEDULER CONFIGURATION  
# =============================================================================
# Based on "Attention Is All You Need", BERT, and ViT papers
# FIXED: Addresses learning rate starvation causing "safe prediction syndrome"

# Stage 2 Learning Rate Schedule (Linear Warmup + Cosine Decay)
STAGE2_BASE_LR = 1e-4                   # FIXED: Reduced from 5e-3 to prevent gradient explosion in transformer fine-tuning
STAGE2_WARMUP_PCT = 0.1                 # FIXED: Increased to 10% warmup for more stable transformer training
STAGE2_ETA_MIN_PCT = 0.1                # FIXED: Increased to 10% final LR for gentler decay

# Stage 2 AdamW Optimizer Parameters  
ADAMW_BETAS_STAGE2 = (0.9, 0.98)       # Transformer-standard betas (BERT/ViT)
ADAMW_EPS_STAGE2 = 1e-8                # Numerical stability epsilon

# Momentum Cycling Parameters (Stage 1 only)
BASE_MOMENTUM = 0.85                    # Base momentum value
MAX_MOMENTUM = 0.95                     # Maximum momentum value

# =============================================================================
# LOGGING AND PROGRESS TRACKING
# =============================================================================

# Learning Rate Logging Configuration (only configurable parameter)
LOG_LR_EVERY_N_BATCHES = 5              # Log learning rate every 5 batches to avoid buffer warnings

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
USE_TISSUE_PATCHES_STAGE2 = False       # Stage 2 ENHANCED mode (set to False for baseline)

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