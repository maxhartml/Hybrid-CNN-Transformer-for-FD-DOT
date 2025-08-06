#!/usr/bin/env python3
"""
Stage 1 Training: CNN Autoencoder Pre-training for NIR-DOT Reconstruction.

This module implements the first stage of the two-stage training pipeline,
focusing on pre-training a CNN autoencoder for feature extraction and basic
volumetric reconstruction from NIR-DOT measurements. The pre-trained features
serve as a foundation for subsequent transformer enhancement.

The training process uses RMSE loss optimization to learn robust feature
representations and basic reconstruction capabilities before introducing
the complexity of transformer-based spatial modeling.

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
import sys
from pathlib import Path
from typing import Dict

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np
from datetime import datetime

# Project imports
from code.models.hybrid_model import HybridCNNTransformer
from code.utils.logging_config import get_training_logger
from code.training.training_config import *  # Import all training config

# =============================================================================
# STAGE-SPECIFIC CONFIGURATION
# =============================================================================

# Initialize module logger
logger = get_training_logger(__name__)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class RMSELoss(nn.Module):
    """
    Root Mean Square Error loss function for volumetric reconstruction.
    
    This loss function computes the RMSE between predicted and target volumes,
    providing a measure of reconstruction accuracy that is sensitive to both
    small and large errors. RMSE is particularly suitable for volumetric
    reconstruction tasks where spatial accuracy is critical.
    
    The loss is computed as: sqrt(mean((pred - target)^2))
    
    Returns:
        torch.Tensor: Scalar RMSE loss value for optimization
    """
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE loss between input and target tensors.
        
        Args:
            input (torch.Tensor): Predicted volume reconstruction
            target (torch.Tensor): Ground truth volume data
            
        Returns:
            torch.Tensor: RMSE loss value
        """
        mse = F.mse_loss(input, target)
        return torch.sqrt(mse)


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
        >>> trainer = Stage1Trainer(learning_rate=1e-4, device="cuda")
        >>> results = trainer.train(data_loaders, epochs=50)
        >>> print(f"Best validation loss: {results['best_val_loss']:.6f}")
    """
    
    def __init__(self, learning_rate=LEARNING_RATE_STAGE1, device=CPU_DEVICE, use_wandb=True, 
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
        
        # Initialize model (stage 1: CNN autoencoder only, no tissue patches)
        # NOTE: Full hybrid model is initialized (including transformer) because:
        # 1. Stage 1 checkpoint needs complete model state for Stage 2 loading
        # 2. Only CNN parameters receive gradients in Stage 1 (transformer stays frozen)
        # 3. Memory allocation is done once for consistency across stages
        self.model = HybridCNNTransformer(
            use_tissue_patches=USE_TISSUE_PATCHES_STAGE1,
            training_stage=TRAINING_STAGE1  # Explicit stage 1 setting
        )
        self.model.to(self.device)
        
        # Loss and optimizer with L2 regularization
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=LR_SCHEDULER_FACTOR,           # Reduce LR by half
            patience=LR_SCHEDULER_PATIENCE,      # Wait 5 epochs before reducing
            min_lr=LR_MIN          # Minimum learning rate
        )
        
        logger.info(f"🏋️  Stage 1 Trainer initialized on {self.device}")
        logger.info(f"📈 Learning rate: {learning_rate}")
        logger.info(f"🔒 L2 regularization (weight decay): {weight_decay}")
        logger.info(f"⏰ Early stopping patience: {early_stopping_patience}")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,}")
        
        # Initialize Weights & Biases
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases experiment tracking."""
        experiment_name = f"stage1_cnn_autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=WANDB_PROJECT,
            name=experiment_name,
            tags=WANDB_TAGS_STAGE1,
            config={
                # Model architecture
                "stage": "CNN_Autoencoder_Pretraining",
                "model_type": "3D_CNN_Autoencoder",
                "training_stage": TRAINING_STAGE1,
                
                # Training hyperparameters
                "learning_rate": self.learning_rate,
                "device": str(self.device),
                "optimizer": "Adam",
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
        logger.info(f"🔬 W&B experiment initialized: {experiment_name}")
    
    def train_epoch(self, data_loader):
        """
        Execute one complete training epoch over the dataset.
        
        This method performs forward propagation, loss computation, and
        backpropagation for all batches in the training dataset. The model
        parameters are updated using the Adam optimizer.
        
        Args:
            data_loader: DataLoader containing training batches with
                        'ground_truth' key from phantom DataLoader
        
        Returns:
            float: Average training loss across all batches in the epoch
        """
        logger.debug("🔄 Starting training epoch...")
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        logger.debug(f"📊 Processing {len(data_loader)} batches in training epoch")
        
        for batch_idx, batch in enumerate(data_loader):
            logger.debug(f"🔍 Processing batch {batch_idx + 1}/{len(data_loader)}")
            
            # Stage 1: Only use ground truth volumes (no NIR measurements)
            ground_truth = batch['ground_truth'].to(self.device)  # Shape: (batch_size, 2, 64, 64, 64)
            logger.debug(f"📦 Ground truth batch shape: {ground_truth.shape}")
            logger.debug(f"🖥️  Ground truth moved to device: {ground_truth.device}")
            
            # Forward pass - ground truth as both input and target for autoencoder training
            logger.debug("⚡ Starting forward pass...")
            self.optimizer.zero_grad()
            outputs = self.model(ground_truth, tissue_patches=None)
            logger.debug(f"📤 Model output shape: {outputs['reconstructed'].shape}")
            
            # Compute loss - reconstruction vs original
            logger.debug("📏 Computing RMSE loss...")
            loss = self.criterion(outputs['reconstructed'], ground_truth)
            logger.debug(f"💰 Batch loss: {loss.item():.6f}")
            
            # Backward pass
            logger.debug("🔙 Starting backward pass...")
            loss.backward()
            self.optimizer.step()
            logger.debug("✅ Optimizer step completed")
            
            total_loss += loss.item()
            num_batches += 1
            
            # Show batch progress at INFO level (every batch)
            logger.info(f"📈 Stage 1 Batch {batch_idx + 1}/{len(data_loader)}: Loss = {loss.item():.6f}, Avg = {total_loss/num_batches:.6f}")
            
            # Additional detailed logging at DEBUG level
            if batch_idx % BATCH_LOG_INTERVAL == 0:  # Log every 5 batches during DEBUG
                logger.debug(f"🔍 Detailed: Batch {batch_idx}: Loss = {loss.item():.6f}, Running Avg = {total_loss/num_batches:.6f}")
        
        avg_loss = total_loss / num_batches
        logger.debug(f"✅ Training epoch completed. Average loss: {avg_loss:.6f}")
        return avg_loss
    
    def _log_reconstruction_images(self, predictions, targets, epoch):
        """Log 3D reconstruction slices to W&B for visualization."""
        if not self.use_wandb:
            return
            
        try:
            # Take middle slices of first batch item for visualization
            # Shape: (batch_size, channels, D, H, W) -> take middle slice in each dimension
            pred_batch = predictions[0].cpu().numpy()  # First item in batch
            target_batch = targets[0].cpu().numpy()
            
            logger.debug(f"Logging images - Pred shape: {pred_batch.shape}, Target shape: {target_batch.shape}")
            
            # Function to normalize data to 0-255 range for W&B visualization
            def normalize_for_display(data):
                """Normalize data to 0-255 range for W&B visualization."""
                data_min = data.min()
                data_max = data.max()
                if data_max > data_min:
                    normalized = ((data - data_min) / (data_max - data_min)) * 255.0
                else:
                    normalized = np.zeros_like(data)
                return normalized.astype(np.uint8)
            
            # Log slices from different dimensions (absorption coefficient channel)
            absorption_channel = 0
            
            # XY plane (Z=32) - middle slice in Z dimension
            pred_xy = pred_batch[absorption_channel, :, :, pred_batch.shape[-1]//2]
            target_xy = target_batch[absorption_channel, :, :, target_batch.shape[-1]//2]
            
            # XZ plane (Y=32) - middle slice in Y dimension
            pred_xz = pred_batch[absorption_channel, :, pred_batch.shape[-2]//2, :]
            target_xz = target_batch[absorption_channel, :, target_batch.shape[-2]//2, :]
            
            # YZ plane (X=32) - middle slice in X dimension
            pred_yz = pred_batch[absorption_channel, pred_batch.shape[-3]//2, :, :]
            target_yz = target_batch[absorption_channel, target_batch.shape[-3]//2, :, :]
            
            # Normalize all images for proper W&B display
            pred_xy_norm = normalize_for_display(pred_xy)
            target_xy_norm = normalize_for_display(target_xy)
            pred_xz_norm = normalize_for_display(pred_xz)
            target_xz_norm = normalize_for_display(target_xz)
            pred_yz_norm = normalize_for_display(pred_yz)
            target_yz_norm = normalize_for_display(target_yz)
            
            wandb.log({
                f"Reconstructions/predicted_xy_slice": wandb.Image(pred_xy_norm, caption=f"Epoch {epoch} - Predicted XY slice (phantom_idx=0, z=32)"),
                f"Reconstructions/target_xy_slice": wandb.Image(target_xy_norm, caption=f"Epoch {epoch} - Ground Truth XY slice (phantom_idx=0, z=32)"),
                f"Reconstructions/predicted_xz_slice": wandb.Image(pred_xz_norm, caption=f"Epoch {epoch} - Predicted XZ slice (phantom_idx=0, y=32)"),
                f"Reconstructions/target_xz_slice": wandb.Image(target_xz_norm, caption=f"Epoch {epoch} - Ground Truth XZ slice (phantom_idx=0, y=32)"),
                f"Reconstructions/predicted_yz_slice": wandb.Image(pred_yz_norm, caption=f"Epoch {epoch} - Predicted YZ slice (phantom_idx=0, x=32)"),
                f"Reconstructions/target_yz_slice": wandb.Image(target_yz_norm, caption=f"Epoch {epoch} - Ground Truth YZ slice (phantom_idx=0, x=32)"),
            }, step=epoch)
            
            logger.debug(f"✅ Successfully logged reconstruction images for epoch {epoch}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log reconstruction images: {e}")
            logger.debug(f"Error details: {str(e)}")
    
    def validate(self, data_loader):
        """
        Evaluate the model on the validation dataset.
        
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
        
        logger.debug(f"📊 Processing {len(data_loader)} validation batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                logger.debug(f"🔍 Validating batch {batch_idx + 1}/{len(data_loader)}")
                
                # Stage 1: Only use ground truth volumes (no NIR measurements)
                ground_truth = batch['ground_truth'].to(self.device)  # Shape: (batch_size, 2, 64, 64, 64)
                logger.debug(f"📦 Validation batch shape: {ground_truth.shape}")
                
                logger.debug("⚡ Forward pass (no gradients)...")
                outputs = self.model(ground_truth, tissue_patches=None)
                loss = self.criterion(outputs['reconstructed'], ground_truth)
                logger.debug(f"💰 Validation batch loss: {loss.item():.6f}")
                
                total_loss += loss.item()
                num_batches += 1
                
                # Show validation batch progress at INFO level (every batch)
                logger.info(f"🔍 Stage 1 Val Batch {batch_idx + 1}/{len(data_loader)}: Loss = {loss.item():.6f}, Avg = {total_loss/num_batches:.6f}")
        
        avg_loss = total_loss / num_batches
        logger.debug(f"✅ Validation completed. Average loss: {avg_loss:.6f}")
        return avg_loss
    
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
            >>> results = trainer.train(data_loaders, epochs=100)
            >>> print(f"Training completed with best loss: {results['best_val_loss']}")
        """
        logger.info(f"🚀 Starting Stage 1 training for {epochs} epochs")
        logger.debug(f"📊 Training configuration: device={self.device}, lr={self.learning_rate}, epochs={epochs}")
        logger.debug(f"📈 Data loaders: train_batches={len(data_loaders['train'])}, val_batches={len(data_loaders['val'])}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"📅 Starting Epoch {epoch + 1}/{epochs}")
            
            # Train: Update model parameters using training data
            logger.debug(f"🏋️  Beginning training phase for epoch {epoch+1}")
            train_loss = self.train_epoch(data_loaders['train'])
            logger.info(f"🏋️  Training completed - Average Loss: {train_loss:.6f}")
            
            # Validate: Evaluate on unseen data (no parameter updates) 
            # We validate every epoch to: 1) Monitor overfitting, 2) Save best models, 3) Track progress
            logger.debug(f"🔍 Beginning validation phase for epoch {epoch+1}")
            val_loss = self.validate(data_loaders['val'])
            logger.info(f"🔍 Validation completed - Average Loss: {val_loss:.6f}")
            
            # Log to W&B with organized structure
            if self.use_wandb:
                wandb.log({
                    "Charts/train_loss": train_loss,
                    "Charts/val_loss": val_loss,
                    "Charts/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "Charts/train_val_loss_ratio": train_loss / val_loss if val_loss > 0 else 0,
                }, step=epoch + 1)  # Use step parameter instead of logging epoch as data
                
                # Log reconstruction images periodically (and always on first/last epoch)
                should_log_images = (epoch % LOG_IMAGES_EVERY == 0) or (epoch == 0) or (epoch == epochs - 1)
                if should_log_images:
                    # Get a batch for visualization
                    try:
                        sample_batch = next(iter(data_loaders['val']))
                        ground_truth = sample_batch['ground_truth'].to(self.device)
                        with torch.no_grad():
                            outputs = self.model(ground_truth, tissue_patches=None)
                        self._log_reconstruction_images(outputs['reconstructed'], ground_truth, epoch + 1)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to log images at epoch {epoch + 1}: {e}")
            
            # Print progress
            if epoch % PROGRESS_LOG_INTERVAL == 0 or epoch == epochs - FINAL_EPOCH_OFFSET:
                logger.info(f"📈 Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
            
            # Early stopping and best model tracking
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0  # Reset patience counter
                checkpoint_path = f"{CHECKPOINT_BASE_DIR}/{CHECKPOINT_STAGE1}"
                logger.info(f"🎉 New best model! Improvement: {improvement:.6f} → Best validation loss: {self.best_val_loss:.6f}")
                logger.debug(f"💾 Checkpoint path: {checkpoint_path}")
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                logger.debug(f"💾 New best model saved at epoch {epoch+1}")
            else:
                self.patience_counter += 1
                logger.debug(f"📊 No improvement. Current: {val_loss:.6f}, Best: {self.best_val_loss:.6f}, Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                # Check for early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    logger.info(f"🔄 No improvement for {self.early_stopping_patience} epochs")
                    logger.info(f"🏆 Best validation loss: {self.best_val_loss:.6f}")
                    self.early_stopped = True
                    break
        
        # Training completion message
        if self.early_stopped:
            logger.info(f"✅ Stage 1 training stopped early! Best val loss: {self.best_val_loss:.6f}")
        else:
            logger.info(f"✅ Stage 1 training complete! Best val loss: {self.best_val_loss:.6f}")
        
        logger.debug(f"🏁 Training summary: Completed epochs: {epoch+1}, Final best loss: {self.best_val_loss:.6f}")
        
        # Finish W&B run
        if self.use_wandb:
            wandb.log({"System/final_best_val_loss": self.best_val_loss, "System/early_stopped": self.early_stopped})
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
        logger.debug(f"💾 Checkpoint saved: {path}")
        logger.debug(f"📊 Checkpoint data keys: {list(checkpoint_data.keys())}")
