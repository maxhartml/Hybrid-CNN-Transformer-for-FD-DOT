#!/usr/bin/env python3
"""
Complete Training Pipeline for Hybrid CNN-Transformer NIR-DOT Reconstruction.

This module provides the main entry point for training the hybrid CNN-Transformer
model used for Near-Infrared Diffuse Optical Tomography (NIR-DOT) reconstruction.

Training Pipeline:
1. Stage 1: CNN autoencoder pre-training for robust feature extraction
2. Stage 2: Transformer enhancement with frozen CNN decoder

The pipeline supports multiple training configurations:
- Stage 1: CNN-only pre-training with RMSE optimization
- Stage 2 Baseline: Transformer training without tissue context
- Stage 2 Enhanced: Transformer training with tissue patch integration

Features:
- Comprehensive experiment tracking and logging
- Flexible command-line configuration
- Automatic device detection (CPU/GPU)
- Progressive training with checkpoint management
- Detailed progress monitoring and result reporting

Usage Examples:
    # Stage 1 training
    python train_hybrid_model.py --stage stage1 --epochs 50
    
    # Stage 2 baseline training
    python train_hybrid_model.py --stage stage2 --epochs 100 \
        --stage1_checkpoint checkpoints/stage1_best.pth
    
    # Stage 2 enhanced training with tissue patches
    python train_hybrid_model.py --stage stage2 --epochs 100 \
        --stage1_checkpoint checkpoints/stage1_best.pth --use_tissue_patches

Author: Max Hart
Date: July 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import argparse
import os
import sys
from pathlib import Path

# Third-party imports
import torch

# Project imports
from code.data_processing.data_loader import create_nir_dataloaders, create_phantom_dataloaders
from code.training.stage1_trainer import Stage1Trainer
from code.training.stage2_trainer import Stage2Trainer
from code.utils.logging_config import get_training_logger, NIRDOTLogger

# =============================================================================
# HYPERPARAMETERS AND CONSTANTS
# =============================================================================

# Training Configuration
BATCH_SIZE_STAGE1 = 4                   # Stage 1: CNN autoencoder (ground truth only, less memory)
BATCH_SIZE_STAGE2 = 4                   # Stage 2: Transformer (full phantoms + attention, more memory)
LEARNING_RATE = 1e-4                    # Default learning rate
EPOCHS_STAGE1 = 50                      # Default epochs for Stage 1
EPOCHS_STAGE2 = 100                     # Default epochs for Stage 2

# Device Configuration
CUDA_DEVICE = "cuda"                    # CUDA device identifier
CPU_DEVICE = "cpu"                      # CPU device identifier

# Data Configuration
DATA_DIRECTORY = "data"                 # Default data directory path
CHECKPOINT_DIRECTORY = "checkpoints"    # Default checkpoint directory

# Stage Identifiers
STAGE1_ID = "stage1"                    # Stage 1 training identifier
STAGE2_ID = "stage2"                    # Stage 2 training identifier

# Experiment Naming
EXPERIMENT_BASE = "NIR_DOT_Hybrid_Stage_"  # Base experiment name
ENHANCED_SUFFIX = "_Enhanced"           # Suffix for enhanced Stage 2
BASELINE_SUFFIX = "_Baseline"           # Suffix for baseline Stage 2

# Checkpoint Configuration
STAGE1_CHECKPOINT_FILE = "stage1_best.pth"  # Default Stage 1 checkpoint filename

# Logging Configuration
LOG_LEVEL = "INFO"                      # Default logging level

# =============================================================================
# EXPERIMENT TRACKING CONFIGURATION  
# =============================================================================

# Weights & Biases Configuration
WANDB_PROJECT = "nir-dot-reconstruction"     # Main project name
WANDB_ENTITY = None                          # Your W&B username (None = personal account)
USE_WANDB = True                             # Enable/disable W&B logging
LOG_IMAGES_EVERY = 10                        # Log reconstruction images every N epochs
LOG_GRADIENTS = False                        # Log gradient histograms (expensive)
WANDB_TAGS = ["nir-dot", "cnn-autoencoder", "transformer", "medical-imaging"]

# Initialize module logger
logger = get_training_logger(__name__)

# =============================================================================
# DEVICE DETECTION AND UTILITY FUNCTIONS
# =============================================================================

def get_best_device():
    """
    Detect the best available device for training.
    Priority: CUDA (NVIDIA GPU) > CPU
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return CUDA_DEVICE
    else:
        return CPU_DEVICE

DEVICE = get_best_device()

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """
    Main training pipeline orchestration function.
    
    This function coordinates the complete training process including:
    - Command-line argument parsing and validation
    - Experiment configuration and logging setup
    - Data loading with appropriate tissue patch configuration
    - Stage-specific trainer initialization and execution
    - Results collection and experiment completion logging
    
    The function handles both Stage 1 and Stage 2 training with
    comprehensive error handling and progress reporting.
    """
    # =========================================================================
    # SETUP AND CONFIGURATION
    # =========================================================================
    
    # Setup logging first
    NIRDOTLogger.setup_logging(log_level=LOG_LEVEL)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=[STAGE1_ID, STAGE2_ID], required=True,
                       help='Training stage to run')
    parser.add_argument('--epochs', type=int, default=EPOCHS_STAGE1,
                       help='Number of epochs to train')
    parser.add_argument('--use_tissue_patches', action='store_true',
                       help='Use tissue patches (only for stage2)')
    parser.add_argument('--stage1_checkpoint', type=str,
                       help='Path to stage1 checkpoint (required for stage2)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    args = parser.parse_args()

    # Adjust default epochs based on stage
    if args.epochs == EPOCHS_STAGE1 and args.stage == STAGE2_ID:
        args.epochs = EPOCHS_STAGE2

    # Prepare experiment configuration
    use_wandb = USE_WANDB and not args.no_wandb
    config = {
        'stage': args.stage,
        'epochs': args.epochs,
        'use_tissue_patches': args.use_tissue_patches if args.stage == STAGE2_ID else False,
        'batch_size': BATCH_SIZE_STAGE1 if args.stage == STAGE1_ID else BATCH_SIZE_STAGE2,
        'learning_rate': LEARNING_RATE,
        'device': DEVICE,
        'use_wandb': use_wandb
    }
    
    experiment_name = EXPERIMENT_BASE + args.stage
    if args.stage == STAGE2_ID:
        experiment_name += ENHANCED_SUFFIX if args.use_tissue_patches else BASELINE_SUFFIX
    
    # Start experiment logging
    NIRDOTLogger.log_experiment_start(experiment_name, config)

    # =========================================================================
    # EXPERIMENT INITIALIZATION AND LOGGING
    # =========================================================================
    
    logger.info(f"🔬 NIR-DOT Hybrid Training Pipeline")
    logger.info(f"📊 Stage: {args.stage}")
    logger.info(f"🖥️  Device: {DEVICE}")
    if DEVICE == CUDA_DEVICE:
        logger.info(f"🚀 Using CUDA GPU acceleration!")
    else:
        logger.info(f"⚠️  Using CPU - training will be slower")
    logger.info(f"📈 Epochs: {args.epochs}")
    logger.info(f"📦 Batch size: {BATCH_SIZE_STAGE1 if args.stage == STAGE1_ID else BATCH_SIZE_STAGE2}")
    
    if args.stage == STAGE2_ID:
        logger.info(f"🧬 Use tissue patches: {args.use_tissue_patches}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    # Load data - both stages use phantom-level batching but access different data keys
    logger.info("📊 Loading NIR-DOT phantom datasets...")
    logger.debug(f"🗂️  Data directory: {DATA_DIRECTORY}")
    logger.debug(f"📁 Current working directory: {os.getcwd()}")
    
    if args.stage == STAGE1_ID:
        logger.debug("🏗️  Creating Stage 1 data loaders (ground truth only)...")
        # Stage 1: Use phantom DataLoader for ground truth batching (CNN autoencoder training)
        data_loaders = create_phantom_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=BATCH_SIZE_STAGE1,  # Stage 1: ground truth volumes only
            use_tissue_patches=False  # Stage 1 doesn't use tissue patches
        )
        logger.info(f"✅ Stage 1 data loaders created successfully")
        logger.debug(f"📊 Train batches: {len(data_loaders['train'])}, Val batches: {len(data_loaders['val'])}")
    else:  # stage2
        logger.debug("🏗️  Creating Stage 2 data loaders (NIR measurements + ground truth)...")
        # Stage 2: Use phantom DataLoader for NIR measurement + ground truth batching
        data_loaders = create_phantom_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=BATCH_SIZE_STAGE2,  # Smaller batch size for complete phantoms (256 measurements each)
            use_tissue_patches=args.use_tissue_patches
        )
        logger.info(f"✅ Stage 2 data loaders created successfully")
        logger.debug(f"📊 Train batches: {len(data_loaders['train'])}, Val batches: {len(data_loaders['val'])}")
        logger.debug(f"🧬 Tissue patches enabled: {args.use_tissue_patches}")

    # =========================================================================
    # TRAINING EXECUTION
    # =========================================================================
    if args.stage == STAGE1_ID:
        logger.info("🏋️  Starting Stage 1: CNN Autoencoder Pre-training")
        logger.debug("🏗️  Initializing Stage 1 trainer...")
        trainer = Stage1Trainer(
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            use_wandb=use_wandb
        )
        logger.debug("✅ Stage 1 trainer initialized successfully")
        logger.info("🚀 Beginning Stage 1 training execution...")
        
        # Log experiment configuration for Stage 1
        if use_wandb:
            logger.info("🔬 W&B logging enabled for Stage 1")
        else:
            logger.info("📝 W&B logging disabled - using local logging only")
        results = trainer.train(data_loaders, epochs=args.epochs)
        logger.info("🎯 Stage 1 training execution completed!")
        
    elif args.stage == STAGE2_ID:
        if not args.stage1_checkpoint:
            raise ValueError("Stage 2 requires --stage1_checkpoint path")
            
        mode = "Enhanced" if args.use_tissue_patches else "Baseline"
        logger.info(f"🏋️  Starting Stage 2: Transformer Training ({mode})")
        logger.info(f"📂 Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        logger.debug(f"📁 Checkpoint file exists: {os.path.exists(args.stage1_checkpoint)}")
        
        logger.debug("🏗️  Initializing Stage 2 trainer...")
        trainer = Stage2Trainer(
            stage1_checkpoint_path=args.stage1_checkpoint,
            use_tissue_patches=args.use_tissue_patches,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            use_wandb=use_wandb
        )
        logger.debug("✅ Stage 2 trainer initialized successfully")
        logger.info("🚀 Beginning Stage 2 training execution...")
        
        # Log experiment configuration for Stage 2
        if use_wandb:
            mode = "Enhanced" if args.use_tissue_patches else "Baseline"
            logger.info(f"🔬 W&B logging enabled for Stage 2 {mode}")
        else:
            logger.info("📝 W&B logging disabled - using local logging only")
            
        results = trainer.train(data_loaders, epochs=args.epochs)
        logger.info("🎯 Stage 2 training execution completed!")

    # =========================================================================
    # EXPERIMENT COMPLETION AND RESULTS
    # =========================================================================
    
    # Log experiment completion
    logger.info("📝 Preparing experiment completion summary...")
    logger.debug(f"🔍 Results object: {results}")
    final_results = {
        'stage': args.stage,
        'final_val_loss': getattr(results, 'best_val_loss', results.get('best_val_loss', 'N/A')),
        'total_epochs': args.epochs,
        'device_used': DEVICE
    }
    logger.debug(f"📊 Final results summary: {final_results}")
    
    NIRDOTLogger.log_experiment_end(experiment_name, final_results)
    logger.info("✅ Training pipeline completed successfully!")
    logger.debug(f"🏁 Pipeline execution finished for {experiment_name}")


if __name__ == "__main__":
    main()
