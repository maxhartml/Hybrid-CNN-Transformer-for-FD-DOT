#!/usr/bin/env python3
"""
Stage 1 Training: CNN Autoencoder Pre-training for NIR-DOT Reconstruction.

This module implements the first stage of the two-stage training pipeline,
focusing on pre-training a CNN autoencoder for feature extraction and basic
volumetric reconstruction from NIR-DOT measurements. The pre-trained features
serve as a foundation for subsequent transformer enhancement.

The training process uses RMSE loss optimization to learn robust feature
representations and basic reconstruction capabilities before introducing
transformer-based spatial modeling.

Classes:
    RMSELoss: Root Mean Square Error loss function for reconstruction optimization
    Stage1Trainer: Complete training pipeline for CNN autoencoder pre-training

Features:
    - Progressive learning rate optimization
    - Comprehensive training and validation loops
    - Automated checkpoint management
    - Detailed logging and progress tracking
    - Device-agnostic training (CPU/GPU)

Author: Max Hart
Date: July 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import pickle
from typing import Dict, Optional
from datetime import datetime

# Third-party imports
import torch
import numpy as np
import wandb
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training for A100 optimization

# Project imports
from code.models.hybrid_model import HybridCNNTransformer
from code.utils.logging_config import get_training_logger
from code.utils.metrics import NIRDOTMetrics, create_metrics_for_stage, calculate_batch_metrics, RMSELoss
from code.training.training_config import *  # Import all training config

# =============================================================================
# STAGE-SPECIFIC CONFIGURATION
# =============================================================================

# Initialize module logger
logger = get_training_logger(__name__)

# =============================================================================
# TRAINING CLASSES
# =============================================================================


class Stage1Trainer:
    """
    CNN Autoencoder Pre-training Pipeline for NIR-DOT Reconstruction.
    
    This trainer implements the first stage of the two-stage training approach,
    focusing on pre-training a CNN autoencoder to learn robust feature representations
    and basic volumetric reconstruction capabilities from NIR-DOT measurements.
    
    The training pipeline includes:
    - CNN autoencoder optimization with RMSE loss
    - Progressive training with validation monitoring
    - Automated checkpoint management for best models
    - Comprehensive logging and progress tracking
    
    Attributes:
        device (torch.device): Training device (CPU/GPU)
        learning_rate (float): Optimizer learning rate
        model (HybridCNNTransformer): CNN autoencoder model
        criterion (RMSELoss): Loss function for reconstruction
        optimizer (torch.optim.Adam): Model parameter optimizer
    
    Example:
        >>> trainer = Stage1Trainer(learning_rate=STAGE1_BASE_LR, device="cuda")
        >>> results = trainer.train(data_loaders, epochs=EPOCHS_STAGE1)
        >>> print(f"Best validation loss: {results['best_val_loss']:.6f}")
    """
    
    def __init__(self, learning_rate=STAGE1_BASE_LR, device=CPU_DEVICE, use_wandb=True, 
                 weight_decay=WEIGHT_DECAY, early_stopping_patience=EARLY_STOPPING_PATIENCE):
        """
        Initialize the Stage 1 trainer with model and optimization components.
        
        Args:
            learning_rate (float): Learning rate for Adam optimizer. Default from constants
            device (str): Training device ('cpu' or 'cuda'). Default from constants
            use_wandb (bool): Whether to use Weights & Biases logging. Default: True
            weight_decay (float): L2 regularization strength. Default: 1e-4
            early_stopping_patience (int): Early stopping patience in epochs. Default: 8
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Phase 2: Latent vector caching will be done AFTER training completes
        # No need to cache during training - we'll generate the full cache post-training
        
        # Initialize enhanced metrics for Stage 1
        self.metrics = create_metrics_for_stage("stage1")
        
        # Initialize model (stage 1: CNN autoencoder only, no tissue patches)
        # NOTE: Full hybrid model is initialized (including transformer) because:
        # 1. Stage 1 checkpoint needs complete model state for Stage 2 loading
        # 2. Only CNN parameters receive gradients in Stage 1 (transformer stays frozen)
        # 3. Memory allocation is done once for consistency across stages
        self.model = HybridCNNTransformer(
            use_tissue_patches=USE_TISSUE_PATCHES_STAGE1,
            training_stage=TRAINING_STAGE1,
            dropout=DROPOUT_TRANSFORMER,
            cnn_dropout=DROPOUT_CNN,
            nir_dropout=DROPOUT_NIR_PROCESSOR
        )
        self.model.to(self.device)
        
        # Performance optimizations
        if USE_CHANNELS_LAST_MEMORY_FORMAT and self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last_3d)
            logger.info("🔧 Enabled channels_last_3d memory format for better performance")
        
        # PyTorch 2.0 compilation for 2x speedup (if available)
        if USE_MODEL_COMPILATION and hasattr(torch, 'compile'):
            logger.info(f"⚡ Compiling model with mode '{COMPILATION_MODE}' for 2x speedup...")
            self.model = torch.compile(self.model, mode=COMPILATION_MODE)
            logger.info("✅ Model compilation complete")
        elif USE_MODEL_COMPILATION:
            logger.warning("⚠️ PyTorch 2.0+ required for model compilation - skipping")
        
        # Loss function - Use standard RMSE loss (Phase 2: simplified loss configuration)
        logger.info("📏 Using RMSE loss for Stage 1 training")
        self.criterion = RMSELoss()
        
        # NOTE: Optimizer and scheduler are created in this stage using research-validated
        # AdamW + OneCycleLR for optimal CNN autoencoder training from scratch.
        self.optimizer = None
        self.scheduler = None
        
        # Mixed precision training for A100 optimization (2x speedup + memory savings)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"🚀 STAGE 1 TRAINING INITIALIZATION")
        logger.info(f"{'='*80}")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"📈 Base Learning Rate: {learning_rate}")
        logger.info(f"🔒 L2 Regularization: {weight_decay}")
        logger.info(f"⏰ Early Stopping Patience: {early_stopping_patience}")
        if self.scaler:
            logger.info(f"🚀 Mixed Precision: Enabled (A100 Optimized)")
        logger.info(f"📅 Scheduler: Will be created during training initialization")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 Model Parameters: {total_params:,}")
        
        # Log GPU info if available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"🔧 Current Batch Size: {BATCH_SIZE}")
        else:
            logger.info(f"💻 CPU Mode: Enabled")
        logger.info(f"{'='*80}")
        
        # Initialize Weights & Biases
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases experiment tracking."""
        experiment_name = f"CNN_Spatial_Features_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=WANDB_PROJECT,
            name=experiment_name,
            tags=WANDB_TAGS_STAGE1,
            config={
                # Model architecture
                "stage": "CNN_Autoencoder_Pretraining",
                "model_type": "3D_CNN_Autoencoder", 
                "training_stage": "Stage_1_Spatial_Feature_Learning",
                
                # Training hyperparameters
                "learning_rate": self.learning_rate,
                "device": str(self.device),
                "optimizer": "AdamW",
                "optimizer_betas": ADAMW_BETAS_STAGE1,
                "scheduler": "OneCycleLR",
                "max_lr": STAGE1_MAX_LR,
                "pct_start": STAGE1_PCT_START,
                "cycle_momentum": STAGE1_CYCLE_MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "loss_function": "RMSE",
                
                # Model specifications (Stage 1: Autoencoder training)
                "input_data": "ground_truth_volumes",
                "input_shape": "64x64x64x2_channels",
                "target_data": "same_ground_truth_volumes", 
                "reconstruction_task": "autoencoder_identity_mapping",
                "use_tissue_patches": USE_TISSUE_PATCHES_STAGE1,
                
                # Architecture details
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        )
        
        # Define custom metrics with proper x-axes for clean dissertation graphs
        wandb.define_metric("LR_Scheduler/*", step_metric="training_step")
        wandb.define_metric("Metrics/*", step_metric="epoch")
        wandb.define_metric("RMSE_Details/*", step_metric="epoch") 
        wandb.define_metric("System/*", step_metric="epoch")
        wandb.define_metric("Analysis/*", step_metric="epoch")
        wandb.define_metric("Reconstructions/*", step_metric="epoch")
        
        logger.info(f"🔬 W&B experiment initialized: {experiment_name}")
        logger.info(f"📊 Custom metrics defined for clean dissertation graphs")
    
    def _create_optimizer_and_scheduler(self, epochs: int, steps_per_epoch: int):
        """
        Create AdamW optimizer and OneCycleLR scheduler for Stage 1 CNN training.
        
        This method implements research-validated optimization for CNN autoencoder
        training from scratch, based on "Super-Convergence" (Smith, 2018) and
        medical imaging best practices.
        
        AdamW Configuration:
        - Decoupled weight decay for better regularization
        - CNN-optimized betas for stability
        - Research-validated learning rate
        
        OneCycleLR Configuration:
        - Aggressive exploration phase (20% warmup)
        - Strong final decay for polishing
        - Momentum cycling for better feature learning
        
        Args:
            epochs (int): Total training epochs
            steps_per_epoch (int): Batches per epoch
            
        Returns:
            tuple: (optimizer, scheduler) configured for Stage 1
        """
        # Create AdamW optimizer with CNN-optimized parameters
        # Based on "Fixing Weight Decay Regularization in Adam" (Loshchilov & Hutter, 2019)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=STAGE1_BASE_LR,                    # Base LR (overridden by OneCycleLR)
            weight_decay=WEIGHT_DECAY,            # L2 regularization (medical imaging standard)
            betas=ADAMW_BETAS_STAGE1,            # CNN-optimized betas (0.9, 0.95)
            eps=ADAMW_EPS_STAGE1                 # Numerical stability
        )
        
        # Create OneCycleLR scheduler with research-validated parameters
        # Based on "Super-Convergence" (Smith, 2018)
        total_steps = epochs * steps_per_epoch
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=STAGE1_MAX_LR,                # Peak LR (2e-3, found via LR range test)
            total_steps=total_steps,
            pct_start=STAGE1_PCT_START,          # 15% warmup (conservative)
            div_factor=STAGE1_DIV_FACTOR,        # Conservative div_factor (25)
            final_div_factor=STAGE1_FINAL_DIV_FACTOR,  # Strong final decay (1e4)
            anneal_strategy='cos',               # Smooth cosine annealing
            cycle_momentum=STAGE1_CYCLE_MOMENTUM,      # Enable momentum cycling
            base_momentum=BASE_MOMENTUM,         # Base momentum (0.85)
            max_momentum=MAX_MOMENTUM            # Max momentum (0.95)
        )
        
        logger.info(f"🚀 STAGE 1 ADAMW OPTIMIZER:")
        logger.info(f"   ├─ Base LR: {STAGE1_BASE_LR:.0e}")
        logger.info(f"   ├─ Weight Decay: {WEIGHT_DECAY:.0e}")
        logger.info(f"   └─ Betas: {ADAMW_BETAS_STAGE1}")
        
        logger.info(f"🚀 STAGE 1 ONECYCLELR SCHEDULER:")
        logger.info(f"   ├─ Max LR: {STAGE1_MAX_LR:.0e}")
        logger.info(f"   ├─ Total Steps: {total_steps:,}")
        logger.info(f"   ├─ Warmup: {STAGE1_PCT_START*100:.0f}%")
        logger.info(f"   ├─ Div Factor: {STAGE1_DIV_FACTOR}")
        logger.info(f"   └─ Momentum Cycling: {STAGE1_CYCLE_MOMENTUM}")
        
        return self.optimizer, self.scheduler
    
    def _log_learning_rate_to_wandb(self, epoch: int, batch_idx: int, total_batches: int):
        """
        Log current learning rate to W&B for OneCycleLR visualization.
        
        OneCycleLR updates per-batch, so we log with high frequency for
        complete visualization of the learning rate schedule progression.
        
        Args:
            epoch (int): Current epoch number
            batch_idx (int): Current batch index within epoch
            total_batches (int): Total batches per epoch
        """
        if not self.use_wandb:
            return
            
        try:
            if not wandb.run:
                return
                
            current_lr = self.optimizer.param_groups[0]['lr']
            current_momentum = self.optimizer.param_groups[0].get('betas', [0, 0])[0]
            
            # Calculate global step for OneCycleLR tracking (dissertation graphs need this!)
            global_step = epoch * total_batches + batch_idx + 1  # Start from 1, not 0
            
            # Log detailed learning rate and momentum for smooth OneCycleLR visualization
            # Essential for dissertation graphs showing scheduler behavior
            wandb.log({
                "training_step": global_step,  # Custom x-axis for LR metrics
                "LR_Scheduler/Learning_Rate": current_lr,
                "LR_Scheduler/Momentum": current_momentum,
                "LR_Scheduler/Training_Progress": epoch + (batch_idx / total_batches)
            })
                
        except Exception as e:
            logger.debug(f"W&B LR logging failed: {e}")
    
    def train_epoch(self, data_loader, epoch=0):
        """
        Execute one complete training epoch over the dataset with enhanced metrics.
        
        This method performs forward propagation, loss computation, and
        backpropagation for all batches in the training dataset. The model
        parameters are updated using the Adam optimizer.
        
        Args:
            data_loader: DataLoader containing training batches with
                        'ground_truth' key from phantom DataLoader
            epoch (int): Current epoch number for W&B logging
        
        Returns:
            tuple: (average_loss, epoch_metrics) for training monitoring
        """
        logger.debug("🔄 Starting training epoch...")
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Initialize metrics tracking
        epoch_metrics = {
            'dice': 0.0, 'contrast_ratio': 0.0, 'rmse_overall': 0.0,
            'rmse_mu_a': 0.0, 'rmse_mu_s': 0.0
        }
        
        logger.debug(f"📊 Processing {len(data_loader)} batches in training epoch")
        
        for batch_idx, batch in enumerate(data_loader):
            logger.debug(f"🔍 Processing batch {batch_idx + 1}/{len(data_loader)}")
            
            # Stage 1: Only use ground truth volumes (no NIR measurements)
            ground_truth = batch['ground_truth'].to(self.device)  # Shape: (batch_size, 2, 64, 64, 64)
            logger.debug(f"📦 Ground truth batch shape: {ground_truth.shape}")
            logger.debug(f"🖥️  Ground truth moved to device: {ground_truth.device}")
            
            # Forward pass with mixed precision for A100 optimization
            logger.debug("⚡ Starting forward pass...")
            self.optimizer.zero_grad()
            
            if self.scaler:  # Mixed precision training
                with autocast():
                    outputs = self.model(ground_truth, tissue_patches=None)
                    logger.debug(f"📤 Model output shape: {outputs['reconstructed'].shape}")
                    
                    # Compute loss - reconstruction vs original
                    logger.debug("📏 Computing RMSE loss...")
                    loss = self.criterion(outputs['reconstructed'], ground_truth)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Apply gradient clipping before optimizer step
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                
                # Monitor gradient norm for training health
                if grad_norm > GRADIENT_MONITOR_THRESHOLD:
                    logger.warning(f"⚠️ High gradient norm detected: {grad_norm:.4f} > {GRADIENT_MONITOR_THRESHOLD}")
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # Standard precision training (CPU fallback)
                outputs = self.model(ground_truth, tissue_patches=None)
                logger.debug(f"📤 Model output shape: {outputs['reconstructed'].shape}")
                
                # Compute loss - reconstruction vs original
                logger.debug("📏 Computing RMSE loss...")
                loss = self.criterion(outputs['reconstructed'], ground_truth)
                
                # Standard backward pass
                loss.backward()
                
                # Apply gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                
                # Monitor gradient norm for training health
                if grad_norm > GRADIENT_MONITOR_THRESHOLD:
                    logger.warning(f"⚠️ High gradient norm detected: {grad_norm:.4f} > {GRADIENT_MONITOR_THRESHOLD}")
                
                self.optimizer.step()
                
            # OneCycleLR updates per-batch (essential for proper LR scheduling)
            self.scheduler.step()
            
            # Log learning rate every 5 batches to avoid W&B buffer warnings
            if batch_idx % LOG_LR_EVERY_N_BATCHES == 0:
                self._log_learning_rate_to_wandb(epoch, batch_idx, len(data_loader))
                
            logger.debug(f"💰 Batch loss: {loss.item():.6f}")
            logger.debug("✅ Optimizer step completed")
            
            # Calculate enhanced metrics for this batch
            with torch.no_grad():
                batch_metrics = calculate_batch_metrics(
                    self.metrics, outputs, ground_truth, "stage1"
                )
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
            
            total_loss += loss.item()
            num_batches += 1
            
            # Show batch progress with enhanced metrics format (including A-RMSE and S-RMSE)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"🏋️  TRAIN | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                       f"RMSE: {loss.item():.4f} | Dice: {batch_metrics.get('dice', 0):.4f} | "
                       f"CR: {batch_metrics.get('contrast_ratio', 0):.4f} | "
                       f"A-RMSE: {batch_metrics.get('rmse_mu_a', 0):.4f} | "
                       f"S-RMSE: {batch_metrics.get('rmse_mu_s', 0):.4f} | LR: {current_lr:.2e}")
            
            # Log gradient norm at debug level for monitoring training health
            logger.debug(f"🔧 Batch {batch_idx + 1} | Gradient Norm: {grad_norm:.3f}")
            
            # Additional detailed logging at DEBUG level
            if batch_idx % 10 == 0:  # Log every 10 batches for stability monitoring
                logger.debug(f"🔍 Detailed: Batch {batch_idx}: Loss = {loss.item():.6f}, "
                           f"Running Avg = {total_loss/num_batches:.6f}")
        
        avg_loss = total_loss / num_batches
        
        # Average metrics across epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        logger.debug(f"✅ Training epoch completed. Average loss: {avg_loss:.6f}")
        logger.info(f"📊 TRAIN SUMMARY | RMSE: {avg_loss:.4f} | Dice: {epoch_metrics['dice']:.4f} | "
                   f"CR: {epoch_metrics['contrast_ratio']:.4f} | μₐ: {epoch_metrics['rmse_mu_a']:.4f} | "
                   f"μₛ: {epoch_metrics['rmse_mu_s']:.4f}")
        
        return avg_loss, epoch_metrics
    
    def _log_reconstruction_images(self, predictions, targets, epoch, phantom_ids=None, step=None):
        """Log 3D reconstruction slices to W&B for visualization."""
        if not self.use_wandb:
            return
            
        # Use shared visualization function to avoid code duplication
        from code.utils.visualization import log_reconstruction_images_to_wandb
        log_reconstruction_images_to_wandb(predictions, targets, epoch, "Reconstructions", step, phantom_ids)
    
    def validate(self, data_loader):
        """
        Evaluate the model on the validation dataset with enhanced metrics.
        
        This method performs forward propagation without gradient computation
        to assess model performance on unseen data. Used for monitoring
        training progress and implementing early stopping criteria.
        
        Args:
            data_loader: DataLoader containing validation batches with
                        'ground_truth' key from phantom DataLoader
        
        Returns:
            float: Average validation loss across all validation batches
        """
        logger.debug("🔍 Starting validation epoch...")
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Initialize metrics tracking
        epoch_metrics = {
            'dice': 0.0, 'contrast_ratio': 0.0, 'rmse_overall': 0.0,
            'rmse_mu_a': 0.0, 'rmse_mu_s': 0.0
        }
        
        logger.debug(f"📊 Processing {len(data_loader)} validation batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                logger.debug(f"🔍 Validating batch {batch_idx + 1}/{len(data_loader)}")
                
                # Stage 1: Only use ground truth volumes (no NIR measurements)
                ground_truth = batch['ground_truth'].to(self.device)  # Shape: (batch_size, 2, 64, 64, 64)
                logger.debug(f"📦 Validation batch shape: {ground_truth.shape}")
                
                logger.debug("⚡ Forward pass (no gradients)...")
                if self.scaler:  # Mixed precision validation
                    with autocast():
                        outputs = self.model(ground_truth, tissue_patches=None)
                        loss = self.criterion(outputs['reconstructed'], ground_truth)
                else:  # Standard precision validation
                    outputs = self.model(ground_truth, tissue_patches=None)
                    loss = self.criterion(outputs['reconstructed'], ground_truth)
                    
                logger.debug(f"💰 Validation batch loss: {loss.item():.6f}")
                
                # Calculate enhanced metrics for this batch
                batch_metrics = calculate_batch_metrics(
                    self.metrics, outputs, ground_truth, "stage1"
                )
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                
                total_loss += loss.item()
                num_batches += 1
                
                # Show validation batch progress with enhanced metrics format
                logger.info(f"🔍 VALID | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                           f"RMSE: {loss.item():.4f} | Dice: {batch_metrics.get('dice', 0):.4f} | "
                           f"CR: {batch_metrics.get('contrast_ratio', 0):.4f} | "
                           f"A-RMSE: {batch_metrics.get('rmse_mu_a', 0):.4f} | "
                           f"S-RMSE: {batch_metrics.get('rmse_mu_s', 0):.4f}")
        
        avg_loss = total_loss / num_batches
        
        # Average metrics across epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        logger.debug(f"✅ Validation completed. Average loss: {avg_loss:.6f}")
        logger.info(f"📊 VALID SUMMARY | RMSE: {avg_loss:.4f} | Dice: {epoch_metrics['dice']:.4f} | "
                   f"CR: {epoch_metrics['contrast_ratio']:.4f} | μₐ: {epoch_metrics['rmse_mu_a']:.4f} | "
                   f"μₛ: {epoch_metrics['rmse_mu_s']:.4f}")
        
        return avg_loss, epoch_metrics
    
    def train(self, data_loaders, epochs=EPOCHS_STAGE1):
        """
        Execute the complete Stage 1 training pipeline.
        
        This method orchestrates the full training process including epoch-wise
        training and validation, progress monitoring, and automatic checkpoint
        saving for the best performing model based on validation loss.
        
        Args:
            data_loaders (dict): Dictionary containing 'train' and 'val' DataLoaders
            epochs (int): Number of training epochs to execute. Default from constants
        
        Returns:
            dict: Training results containing 'best_val_loss' for analysis
        
        Example:
            >>> data_loaders = {'train': train_loader, 'val': val_loader}
            >>> results = trainer.train(data_loaders, epochs=EPOCHS_STAGE1)
            >>> print(f"Training completed with best loss: {results['best_val_loss']}")
        """
        # Initialize AdamW optimizer and OneCycleLR scheduler
        if self.optimizer is None:
            steps_per_epoch = len(data_loaders['train'])
            self._create_optimizer_and_scheduler(epochs, steps_per_epoch)
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"🚀 STARTING STAGE 1 TRAINING | {epochs} Epochs")
        logger.info(f"{'='*80}")
        logger.debug(f"📊 Training configuration: device={self.device}, lr={self.learning_rate}, epochs={epochs}")
        logger.debug(f"📈 Data loaders: train_batches={len(data_loaders['train'])}, val_batches={len(data_loaders['val'])}")
        logger.info(f"📅 AdamW + OneCycleLR | Steps per epoch: {len(data_loaders['train'])}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"")
            logger.info(f"📅 EPOCH {epoch + 1}/{epochs}")
            logger.info(f"{'-'*40}")
            
            # Train: Update model parameters using training data
            logger.debug(f"🏋️  Beginning training phase for epoch {epoch+1}")
            train_loss, train_metrics = self.train_epoch(data_loaders['train'], epoch)
            logger.info(f"🏋️  TRAIN COMPLETE | Avg RMSE: {train_loss:.4f}")
            
            # Validate: Evaluate on unseen data (no parameter updates) 
            # We validate every epoch to: 1) Monitor overfitting, 2) Save best models, 3) Track progress
            logger.debug(f"🔍 Beginning validation phase for epoch {epoch+1}")
            val_loss, val_metrics = self.validate(data_loaders['val'])
            logger.info(f"🔍 VALID COMPLETE | Avg RMSE: {val_loss:.4f}")
            
            # Log enhanced metrics to W&B
            if self.use_wandb:
                # Log comprehensive metrics in organized format with clean epoch x-axis
                wandb.log({
                    "epoch": epoch + 1,  # Custom x-axis for epoch metrics
                    
                    # === PRIMARY METRICS (most important) ===
                    "Metrics/RMSE_Overall_Train": train_loss,
                    "Metrics/RMSE_Overall_Valid": val_loss,
                    "Metrics/Dice_Train": train_metrics['dice'],
                    "Metrics/Dice_Valid": val_metrics['dice'],
                    "Metrics/ContrastRatio_Train": train_metrics['contrast_ratio'],
                    "Metrics/ContrastRatio_Valid": val_metrics['contrast_ratio'],
                    
                    # === DETAILED RMSE BREAKDOWN ===
                    "RMSE_Details/Absorption_Train": train_metrics['rmse_mu_a'],
                    "RMSE_Details/Absorption_Valid": val_metrics['rmse_mu_a'],
                    "RMSE_Details/Scattering_Train": train_metrics['rmse_mu_s'],
                    "RMSE_Details/Scattering_Valid": val_metrics['rmse_mu_s'],
                    
                    # === TRAINING SYSTEM ===
                    "System/Epoch": epoch + 1,
                    
                    # === ANALYSIS METRICS ===
                    "Analysis/Train_Valid_RMSE_Ratio": train_loss / val_loss if val_loss > 0 else 0,
                    "Analysis/Dice_Improvement": val_metrics['dice'] - train_metrics['dice'],
                    "Analysis/ContrastRatio_Improvement": val_metrics['contrast_ratio'] - train_metrics['contrast_ratio'],
                })
                
                # Log reconstruction images periodically (and always on first/last epoch)
                should_log_images = (epoch % LOG_IMAGES_EVERY == 0) or (epoch == 0) or (epoch == epochs - 1)
                if should_log_images:
                    # Get a batch for visualization
                    try:
                        sample_batch = next(iter(data_loaders['val']))
                        ground_truth = sample_batch['ground_truth'].to(self.device)
                        phantom_ids = sample_batch['phantom_id'].cpu().numpy()  # Get phantom IDs
                        with torch.no_grad():
                            outputs = self.model(ground_truth, tissue_patches=None)
                        self._log_reconstruction_images(outputs['reconstructed'], ground_truth, epoch, phantom_ids)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to log images at epoch {epoch + 1}: {e}")
            
            # Print epoch summary with clear visual formatting
            if epoch % PROGRESS_LOG_INTERVAL == 0 or epoch == epochs - FINAL_EPOCH_OFFSET:
                logger.info(f"")
                logger.info(f"{'='*80}")
                logger.info(f"� EPOCH {epoch+1:3d}/{epochs} SUMMARY")
                logger.info(f"{'='*80}")
                logger.info(f"📈 Train RMSE: {train_loss:.4f} | Valid RMSE: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"📊 Train Dice: {train_metrics['dice']:.4f} | Valid Dice: {val_metrics['dice']:.4f}")
                logger.info(f"📊 Train CR: {train_metrics['contrast_ratio']:.4f} | Valid CR: {val_metrics['contrast_ratio']:.4f}")
                logger.info(f"📊 Train μₐ RMSE: {train_metrics['rmse_mu_a']:.4f} | Valid μₐ RMSE: {val_metrics['rmse_mu_a']:.4f}")
                logger.info(f"📊 Train μₛ RMSE: {train_metrics['rmse_mu_s']:.4f} | Valid μₛ RMSE: {val_metrics['rmse_mu_s']:.4f}")
            logger.info(f"{'='*80}")
            
            # Log GPU stats every 5 epochs
            if epoch % 5 == 0 and torch.cuda.is_available():
                log_gpu_stats()            # Learning rate scheduling (OneCycleLR updates per-batch, no epoch-level action needed)
            # OneCycleLR is updated per-batch in train_epoch, so no action needed here
            
            # Early stopping and best model tracking
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0  # Reset patience counter
                checkpoint_path = f"{CHECKPOINT_BASE_DIR}/{CHECKPOINT_STAGE1}"
                logger.info(f"🎉 NEW BEST MODEL | Improvement: {improvement:.4f} | Best RMSE: {self.best_val_loss:.4f}")
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
            else:
                self.patience_counter += 1
                logger.debug(f"📊 No improvement. Current: {val_loss:.6f}, Best: {self.best_val_loss:.6f}, Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                # Check for early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"")
                    logger.info(f"🛑 EARLY STOPPING TRIGGERED")
                    logger.info(f"🔄 No improvement for {self.early_stopping_patience} epochs")
                    logger.info(f"🏆 Best RMSE achieved: {self.best_val_loss:.4f}")
                    self.early_stopped = True
                    break
        
        # Training completion message
        logger.info(f"")
        logger.info(f"{'='*80}")
        if self.early_stopped:
            logger.info(f"✅ STAGE 1 TRAINING COMPLETED (Early Stopped)")
        else:
            logger.info(f"✅ STAGE 1 TRAINING COMPLETED (Full {epochs} Epochs)")
        logger.info(f"🏆 Best RMSE Loss: {self.best_val_loss:.4f}")
        logger.info(f"📊 Final Epoch: {epoch+1}")
        logger.info(f"{'='*80}")
        
        logger.debug(f"🏁 Training summary: Completed epochs: {epoch+1}, Final best loss: {self.best_val_loss:.6f}")
        
        # Finish W&B run
        if self.use_wandb:
            wandb.log({"System/final_best_val_loss": self.best_val_loss, "System/early_stopped": self.early_stopped}, commit=False)
            wandb.finish()
            logger.info("🔬 W&B experiment finished")
        
        return {'best_val_loss': self.best_val_loss, 'early_stopped': self.early_stopped}
    
    def save_checkpoint(self, path, epoch, val_loss):
        """
        Save model checkpoint with training state information.
        
        This method creates a comprehensive checkpoint containing the model state,
        optimizer state, and training metadata for resuming training or
        transferring to Stage 2 training.
        
        Args:
            path (str): File path for saving the checkpoint
            epoch (int): Current training epoch number
            val_loss (float): Current validation loss value
        
        The checkpoint includes:
        - Model state dictionary (learned parameters)
        - Optimizer state dictionary (for training resumption)
        - Training metadata (epoch, validation loss)
        """
        logger.debug(f"💾 Saving Stage 1 checkpoint: epoch={epoch}, val_loss={val_loss:.6f}")
        
        # Only create directory if path has a directory component
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create directory if it's not empty (i.e., not current directory)
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"📁 Created checkpoint directory: {dir_path}")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        torch.save(checkpoint_data, path)
        logger.info(f"💾 ✅ CHECKPOINT SAVED | Path: {path} | Epoch: {epoch+1} | Val Loss: {val_loss:.6f}")
        logger.debug(f"📊 Checkpoint data keys: {list(checkpoint_data.keys())}")
    
    def generate_latent_cache_post_training(self, data_loaders, checkpoint_path=None):
        """
        Generate latent vector cache AFTER Stage 1 training completes.
        
        This method loads the best trained CNN encoder and passes all phantoms
        through it to generate latent vectors for Stage 2 training targets.
        
        Args:
            data_loaders (dict): Data loaders with 'train', 'val', 'test' keys
            checkpoint_path (str): Path to best checkpoint (optional)
        
        Returns:
            dict: Latent cache with phantom_id -> latent_vector mapping
        """
        logger.info("🎯 Generating latent vectors cache for Stage 2 training...")
        
        # Load best checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"📂 Loading best checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")
        
        self.model.eval()
        latent_cache = {}
        total_processed = 0
        
        # Process all splits (train, val, test) to create complete latent cache
        splits_to_process = ['train', 'val', 'test'] if 'test' in data_loaders else ['train', 'val']
        
        with torch.no_grad():
            for split_name in splits_to_process:
                logger.info(f"🔄 Processing {split_name} split for latent generation...")
                data_loader = data_loaders[split_name]
                
                for batch_idx, batch in enumerate(data_loader):
                    phantom_ids = batch['phantom_id'].cpu().numpy()
                    ground_truth = batch['ground_truth'].to(self.device)
                    
                    # Forward pass through trained encoder
                    if self.scaler:
                        with autocast():
                            outputs = self.model(ground_truth, tissue_patches=None)
                    else:
                        outputs = self.model(ground_truth, tissue_patches=None)
                    
                    # Extract and store latent vectors
                    if 'latent_vectors' in outputs:
                        latent_vectors = outputs['latent_vectors'].detach().cpu().numpy()
                        
                        for i, phantom_id in enumerate(phantom_ids):
                            # Use consistent phantom ID format
                            phantom_key = f"phantom_{int(phantom_id):05d}"
                            latent_cache[phantom_key] = latent_vectors[i]
                            total_processed += 1
                    else:
                        logger.error("❌ Model output missing 'latent_vectors' - check hybrid model implementation")
                        return {}
                    
                    if (batch_idx + 1) % 10 == 0:
                        logger.debug(f"📊 Processed {batch_idx + 1}/{len(data_loader)} batches in {split_name} split")
        
        # Save latent cache to disk
        cache_path = "checkpoints/stage1_latent_cache.pkl"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(latent_cache, f)
            
            logger.info(f"💾 ✅ LATENT CACHE GENERATED | Path: {cache_path} | Count: {len(latent_cache)} vectors")
            logger.info(f"📊 Total phantoms processed: {total_processed}")
            
            # Log cache statistics
            if latent_cache:
                sample_vector = next(iter(latent_cache.values()))
                logger.info(f"📏 Latent vector shape: {sample_vector.shape}")
                logger.info(f"📈 Cache memory usage: ~{len(latent_cache) * sample_vector.nbytes / 1024 / 1024:.2f} MB")
                logger.debug(f"🔑 Sample phantom IDs: {list(latent_cache.keys())[:5]}...")
        
        except Exception as e:
            logger.error(f"❌ Failed to save latent cache: {e}")
            raise
        
        return latent_cache
