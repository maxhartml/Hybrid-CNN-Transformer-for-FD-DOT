#!/usr/bin/env python3
"""
Complete Training Pipeline for Hybrid CNN-Transformer NIR-DOT Reconstruction.

This script orchestrates the complete two-stage training pipeline for near-in    elif args.stage == STAGE2_IDENTIFIER:
        if not args.stage1_checkpoint:
            raise ValueError("Stage 2 requires --stage1_checkpoint path")
            
        mode = "Enhanced" if args.use_tissue_patches else "Baseline"
        logger.info(f"🏋️  Starting Stage 2: Transformer Training ({mode})")
        logger.info(f"📂 Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        
        trainer = Stage2Trainer(
            stage1_checkpoint_path=args.stage1_checkpoint,
            use_tissue_patches=args.use_tissue_patches,
            learning_rate=DEFAULT_LEARNING_RATE,
            device=DEVICE
        )
        results = trainer.train(data_loaders, epochs=args.epochs)optical tomography (NIR-DOT) image reconstruction using a hybrid approach
that combines CNN autoencoders with transformer-based spatial modeling.

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

Usage:
    # Stage 1 training
    python train_hybrid_model.py --stage stage1 --epochs 50
    
    # Stage 2 baseline training
    python train_hybrid_model.py --stage stage2 --epochs 100 \
        --stage1_checkpoint checkpoints/stage1_best.pth
    
    # Stage 2 enhanced training with tissue patches
    python train_hybrid_model.py --stage stage2 --epochs 100 \
        --stage1_checkpoint checkpoints/stage1_best.pth --use_tissue_patches
"""

import torch
import argparse
import sys
import os
from pathlib import Path

# =============================================================================
# HYPERPARAMETERS AND CONSTANTS
# =============================================================================

# Training Configuration
DEFAULT_BATCH_SIZE = 8                  # Default batch size for training
DEFAULT_BATCH_SIZE_STAGE2 = 4           # Smaller batch size for Stage 2 (complete phantoms)
DEFAULT_LEARNING_RATE = 1e-4            # Default learning rate
DEFAULT_EPOCHS_STAGE1 = 50              # Default epochs for Stage 1
DEFAULT_EPOCHS_STAGE2 = 100             # Default epochs for Stage 2

# Device Configuration
CUDA_DEVICE_NAME = "cuda"               # CUDA device identifier
CPU_DEVICE_NAME = "cpu"                 # CPU device identifier

# Data Configuration
DATA_DIRECTORY = "data"                 # Default data directory path
CHECKPOINT_DIRECTORY = "checkpoints"    # Default checkpoint directory

# Stage Identifiers
STAGE1_IDENTIFIER = "stage1"            # Stage 1 training identifier
STAGE2_IDENTIFIER = "stage2"            # Stage 2 training identifier

# Experiment Naming
EXPERIMENT_BASE_NAME = "NIR_DOT_Hybrid_Stage_"  # Base experiment name
STAGE2_ENHANCED_SUFFIX = "_Enhanced"    # Suffix for enhanced Stage 2
STAGE2_BASELINE_SUFFIX = "_Baseline"    # Suffix for baseline Stage 2

# Checkpoint Filenames
STAGE1_BEST_CHECKPOINT = "stage1_best.pth"  # Default Stage 1 checkpoint filename

# Logging Configuration
DEFAULT_LOG_LEVEL = "INFO"              # Default logging level

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent  # Go up 3 levels: train_hybrid_model.py -> training -> code -> mah422
sys.path.insert(0, str(project_root))

# Now we can import our components
try:
    from code.data_processing.data_loader import create_nir_dataloaders, create_phantom_dataloaders
    from code.training.stage1_trainer import Stage1Trainer
    from code.training.stage2_trainer import Stage2Trainer
    from code.utils.logging_config import get_training_logger, NIRDOTLogger
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

# Initialize logger for this module
logger = get_training_logger(__name__)

# Device detection using constants
DEVICE = CUDA_DEVICE_NAME if torch.cuda.is_available() else CPU_DEVICE_NAME

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
    # Setup logging first
    NIRDOTLogger.setup_logging(log_level=DEFAULT_LOG_LEVEL)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=[STAGE1_IDENTIFIER, STAGE2_IDENTIFIER], required=True,
                       help='Training stage to run')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS_STAGE1,
                       help='Number of epochs to train')
    parser.add_argument('--use_tissue_patches', action='store_true',
                       help='Use tissue patches (only for stage2)')
    parser.add_argument('--stage1_checkpoint', type=str,
                       help='Path to stage1 checkpoint (required for stage2)')
    args = parser.parse_args()

    # Adjust default epochs based on stage
    if args.epochs == DEFAULT_EPOCHS_STAGE1 and args.stage == STAGE2_IDENTIFIER:
        args.epochs = DEFAULT_EPOCHS_STAGE2

    # Log experiment configuration
    config = {
        'stage': args.stage,
        'epochs': args.epochs,
        'use_tissue_patches': args.use_tissue_patches if args.stage == STAGE2_IDENTIFIER else False,
        'batch_size': DEFAULT_BATCH_SIZE if args.stage == STAGE1_IDENTIFIER else DEFAULT_BATCH_SIZE_STAGE2,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'device': DEVICE
    }
    
    experiment_name = EXPERIMENT_BASE_NAME + args.stage
    if args.stage == STAGE2_IDENTIFIER:
        experiment_name += STAGE2_ENHANCED_SUFFIX if args.use_tissue_patches else STAGE2_BASELINE_SUFFIX
    
    NIRDOTLogger.log_experiment_start(experiment_name, config)

    logger.info(f"🔬 NIR-DOT Hybrid Training Pipeline")
    logger.info(f"📊 Stage: {args.stage}")
    logger.info(f"🖥️  Device: {DEVICE}")
    logger.info(f"📈 Epochs: {args.epochs}")
    
    if args.stage == STAGE2_IDENTIFIER:
        logger.info(f"🧬 Use tissue patches: {args.use_tissue_patches}")

    # Load data - both stages use phantom-level batching but access different data keys
    logger.info("📊 Loading NIR-DOT phantom datasets...")
    
    if args.stage == STAGE1_IDENTIFIER:
        # Stage 1: Use phantom DataLoader for ground truth batching (CNN autoencoder training)
        data_loaders = create_phantom_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=DEFAULT_BATCH_SIZE,  # Can use larger batch since only ground truth volumes
            use_tissue_patches=False  # Stage 1 doesn't use tissue patches
        )
    else:  # stage2
        # Stage 2: Use phantom DataLoader for NIR measurement + ground truth batching
        data_loaders = create_phantom_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=DEFAULT_BATCH_SIZE_STAGE2,  # Smaller batch size for complete phantoms (1500 measurements each)
            use_tissue_patches=args.use_tissue_patches
        )

    # Train based on stage
    if args.stage == STAGE1_IDENTIFIER:
        logger.info("🏋️  Starting Stage 1: CNN Autoencoder Pre-training")
        trainer = Stage1Trainer(
            learning_rate=DEFAULT_LEARNING_RATE,
            device=DEVICE
        )
        results = trainer.train(data_loaders, epochs=args.epochs)
        
    elif args.stage == STAGE2_IDENTIFIER:
        if not args.stage1_checkpoint:
            raise ValueError("Stage 2 requires --stage1_checkpoint path")
            
        mode = "Enhanced" if args.use_tissue_patches else "Baseline"
        logger.info(f"🏋️  Starting Stage 2: Transformer Training ({mode})")
        logger.info(f"📂 Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        
        trainer = Stage2Trainer(
            stage1_checkpoint_path=args.stage1_checkpoint,
            use_tissue_patches=args.use_tissue_patches,
            learning_rate=DEFAULT_LEARNING_RATE,
            device=DEVICE
        )
        results = trainer.train(data_loaders, epochs=args.epochs)

    # Log experiment completion
    final_results = {
        'stage': args.stage,
        'final_val_loss': getattr(results, 'best_val_loss', 'N/A'),
        'total_epochs': args.epochs,
        'device_used': DEVICE
    }
    
    NIRDOTLogger.log_experiment_end(experiment_name, final_results)
    logger.info("✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
