#!/usr/bin/env python3
"""
Complete Training Pipeline for Hybrid CNN-Transformer NIR-DOT Reconstruction.

This module provides the main entry point for training the hybrid CNN-Transformer
model used for Near-Infrared Diffuse Optical Tomography (NIR-DOT) reconstruction.

Training Pipeline:
1. Stage 1: CNN autoencoder pre-training for robust feature extraction
2. Stage 2: Transformer enhancement with frozen CNN decoder

The pipeline supports multiple training configurations controlled entirely by
the training_config.py file:
- Stage 1: CNN-only pre-training with RMSE optimization
- Stage 2 Baseline: Transformer training without tissue context
- Stage 2 Enhanced: Transformer training with tissue patch integration

Features:
- Configuration-driven training (no command-line arguments needed)
- Comprehensive experiment tracking and logging
- Automatic device detection (CPU/GPU)
- Progressive training with checkpoint management
- Detailed progress monitoring and result reporting

Usage:
    # Simply run the script - all configuration is in training_config.py
    python train_hybrid_model.py

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os

# Third-party imports
import torch
import torch._dynamo

# Configure torch dynamo to suppress compilation errors and fall back to eager mode
torch._dynamo.config.suppress_errors = True

# Project imports
from code.data_processing.data_loader import create_phantom_dataloaders
from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer
from code.utils.logging_config import get_training_logger, NIRDOTLogger
from .training_config import *  # Import all training config

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

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

def create_dataloaders(data_dir, batch_size, extract_tissue_patches, stage_name):
    """
    Create DataLoaders using training_config.py settings as the single source of truth.
    Uses optimized Stage 1 loader (ground truth only) or full Stage 2 loader.
    
    Args:
        data_dir: Data directory path
        batch_size: Batch size for training
        extract_tissue_patches: Whether to extract tissue patches
        stage_name: Stage name for logging
        
    Returns:
        DataLoaders dictionary
    """
    logger.info(f"📊 Creating {stage_name} data loaders with training config settings...")
    
    data_loaders = create_phantom_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        extract_tissue_patches=extract_tissue_patches,
        stage="stage1" if stage_name == "Stage 1" else "stage2"  # NEW: Pass stage
    )
    logger.info(f"✅ {stage_name} data loaders created successfully with training config settings")
    return data_loaders

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """
    Main training pipeline orchestration function.
    
    This function coordinates the complete training process using configuration
    from training_config.py:
    - Automatic stage selection based on CURRENT_TRAINING_STAGE
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
    
    # Setup logging
    NIRDOTLogger.setup_logging(log_level="INFO")
    
    # Get training configuration from config file
    current_stage = CURRENT_TRAINING_STAGE
    epochs = EPOCHS_STAGE1 if current_stage == TRAINING_STAGE1 else EPOCHS_STAGE2
    use_tissue_patches = USE_TISSUE_PATCHES_STAGE1 if current_stage == TRAINING_STAGE1 else USE_TISSUE_PATCHES_STAGE2
    batch_size = BATCH_SIZE
    learning_rate = STAGE1_BASE_LR if current_stage == TRAINING_STAGE1 else STAGE2_BASE_LR
    
    # Prepare experiment configuration
    config = {
        'stage': current_stage,
        'epochs': epochs,
        'use_tissue_patches': use_tissue_patches,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': DEVICE,
        'use_wandb': USE_WANDB_LOGGING
    }
    
    experiment_name = f"NIR_DOT_Hybrid_Stage_{current_stage}"
    if current_stage == TRAINING_STAGE2:
        experiment_name += "_Enhanced" if use_tissue_patches else "_Baseline"
    
    # Start experiment logging
    NIRDOTLogger.log_experiment_start(experiment_name, config)

    # =========================================================================
    # EXPERIMENT INITIALIZATION AND LOGGING
    # =========================================================================
    
    logger.info(f"🔬 NIR-DOT Hybrid Training Pipeline")
    logger.info(f"📊 Stage: {current_stage}")
    logger.info(f"🖥️  Device: {DEVICE}")
    if DEVICE == CUDA_DEVICE:
        logger.info(f"🚀 Using CUDA GPU acceleration!")
    else:
        logger.info(f"⚠️  Using CPU - training will be slower")
    logger.info(f"📈 Epochs: {epochs}")
    logger.info(f"📦 Batch size: {batch_size}")
    
    if current_stage == TRAINING_STAGE2:
        logger.info(f"🧬 Use tissue patches: {use_tissue_patches}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    # Load data - both stages use phantom-level batching but access different data keys
    logger.info("📊 Loading NIR-DOT phantom datasets...")
    
    # Log DataLoader configuration for transparency
    logger.info("⚙️  DATALOADER CONFIGURATION (from training_config.py):")
    logger.info(f"   ├─ Workers: {NUM_WORKERS}")
    logger.info(f"   ├─ Pin Memory: {PIN_MEMORY}")
    logger.info(f"   ├─ Prefetch Factor: {PREFETCH_FACTOR}")
    logger.info(f"   └─ Persistent Workers: {PERSISTENT_WORKERS}")
    
    logger.debug(f"🗂️  Data directory: {DATA_DIRECTORY}")
    logger.debug(f"📁 Current working directory: {os.getcwd()}")
    
    if current_stage == TRAINING_STAGE1:
        logger.debug("🏗️  Creating Stage 1 data loaders (ground truth only)...")
        # Stage 1: Use phantom DataLoader for ground truth batching (CNN autoencoder training)
        data_loaders = create_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=batch_size,
            extract_tissue_patches=False,  # Skip tissue patches for Stage 1
            stage_name="Stage 1"
        )
        logger.debug(f"📊 Train batches: {len(data_loaders['train'])}, Val batches: {len(data_loaders['val'])}")
    else:  # stage2
        logger.debug("🏗️  Creating Stage 2 data loaders (NIR measurements + ground truth)...")
        # Stage 2: Use phantom DataLoader for NIR measurement + ground truth batching
        data_loaders = create_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=batch_size,
            extract_tissue_patches=use_tissue_patches,  # Extract tissue patches based on config
            stage_name="Stage 2"
        )
        logger.debug(f"📊 Train batches: {len(data_loaders['train'])}, Val batches: {len(data_loaders['val'])}")
        logger.debug(f"🧬 Tissue patches handled by model: {use_tissue_patches}")

    # =========================================================================
    # TRAINING EXECUTION
    # =========================================================================
    if current_stage == TRAINING_STAGE1:
        logger.info("🏋️  Starting Stage 1: CNN Autoencoder Pre-training")
        logger.debug("🏗️  Initializing Stage 1 trainer...")
        trainer = Stage1Trainer(
            learning_rate=learning_rate,
            device=DEVICE,
            use_wandb=USE_WANDB_LOGGING,
            weight_decay=WEIGHT_DECAY,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        logger.debug("✅ Stage 1 trainer initialized successfully")
        logger.info("🚀 Beginning Stage 1 training execution...")
        
        # Log experiment configuration for Stage 1
        if USE_WANDB_LOGGING:
            logger.info("🔬 W&B logging enabled for Stage 1")
        else:
            logger.info("📝 W&B logging disabled - using local logging only")
        results = trainer.train(data_loaders, epochs=epochs)
        logger.info("🎯 Stage 1 training execution completed!")
        
    elif current_stage == TRAINING_STAGE2:
        # Check if Stage 1 checkpoint exists
        if not os.path.exists(STAGE1_CHECKPOINT_PATH):
            raise ValueError(f"Stage 2 requires Stage 1 checkpoint at: {STAGE1_CHECKPOINT_PATH}")
            
        mode = "Enhanced" if use_tissue_patches else "Baseline"
        logger.info(f"🏋️  Starting Stage 2: Transformer Training ({mode})")
        logger.info(f"📂 Loading Stage 1 checkpoint: {STAGE1_CHECKPOINT_PATH}")
        logger.debug(f"📁 Checkpoint file exists: {os.path.exists(STAGE1_CHECKPOINT_PATH)}")
        
        logger.debug("🏗️  Initializing Stage 2 trainer...")
        trainer = Stage2Trainer(
            stage1_checkpoint_path=STAGE1_CHECKPOINT_PATH,
            use_tissue_patches=use_tissue_patches,
            learning_rate=learning_rate,
            device=DEVICE,
            use_wandb=USE_WANDB_LOGGING,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        logger.debug("✅ Stage 2 trainer initialized successfully")
        logger.info("🚀 Beginning Stage 2 training execution...")
        
        # Log experiment configuration for Stage 2
        if USE_WANDB_LOGGING:
            mode = "Enhanced" if use_tissue_patches else "Baseline"
            logger.info(f"🔬 W&B logging enabled for Stage 2 {mode}")
        else:
            logger.info("📝 W&B logging disabled - using local logging only")
            
        results = trainer.train(data_loaders, epochs=epochs)
        logger.info("🎯 Stage 2 training execution completed!")

    # =========================================================================
    # EXPERIMENT COMPLETION AND RESULTS
    # =========================================================================
    
    # Log experiment completion
    logger.info("📝 Preparing experiment completion summary...")
    logger.debug(f"🔍 Results object: {results}")
    final_results = {
        'stage': current_stage,
        'final_val_loss': getattr(results, 'best_val_loss', results.get('best_val_loss', 'N/A')),
        'total_epochs': epochs,
        'device_used': DEVICE
    }
    logger.debug(f"📊 Final results summary: {final_results}")
    
    NIRDOTLogger.log_experiment_end(experiment_name, final_results)
    logger.info("✅ Training pipeline completed successfully!")
    logger.debug(f"🏁 Pipeline execution finished for {experiment_name}")


if __name__ == "__main__":
    main()
