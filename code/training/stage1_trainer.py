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

# Project imports - Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.models.hybrid_model import HybridCNNTransformer
    from code.utils.logging_config import get_training_logger
except ImportError:
    # Fallback to relative imports
    sys.path.insert(0, str(project_root / "code"))
    from models.hybrid_model import HybridCNNTransformer
    from utils.logging_config import get_training_logger

# =============================================================================
# HYPERPARAMETERS AND CONSTANTS
# =============================================================================

# Training Configuration
LEARNING_RATE = 1e-4                    # Adam optimizer learning rate
EPOCHS = 50                             # Default number of training epochs
DEVICE = "cpu"                          # Default training device

# Training Progress and Logging
PROGRESS_LOG_INTERVAL = 10              # Log progress every N epochs
FINAL_EPOCH_OFFSET = 1                  # Offset for final epoch logging
BATCH_LOG_INTERVAL = 5                  # Detailed logging every N batches

# Checkpoint Configuration
CHECKPOINT_FILENAME = "stage1_best.pth" # Default checkpoint filename
CHECKPOINT_BASE_DIR = "checkpoints"     # Base checkpoint directory

# Model Configuration
USE_TISSUE_PATCHES = False              # Stage 1 doesn't use tissue patches
TRAINING_STAGE = "stage1"               # Training stage identifier

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
    
    def __init__(self, learning_rate=LEARNING_RATE, device=DEVICE):
        """
        Initialize the Stage 1 trainer with model and optimization components.
        
        Args:
            learning_rate (float): Learning rate for Adam optimizer. Default: 1e-4
            device (str): Training device ('cpu' or 'cuda'). Default: 'cpu'
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        
        # Initialize model (stage 1: CNN autoencoder only, no tissue patches)
        # NOTE: Full hybrid model is initialized (including transformer) because:
        # 1. Stage 1 checkpoint needs complete model state for Stage 2 loading
        # 2. Only CNN parameters receive gradients in Stage 1 (transformer stays frozen)
        # 3. Memory allocation is done once for consistency across stages
        self.model = HybridCNNTransformer(
            use_tissue_patches=USE_TISSUE_PATCHES,
            training_stage=TRAINING_STAGE  # Explicit stage 1 setting
        )
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"🏋️  Stage 1 Trainer initialized on {self.device}")
        logger.info(f"📈 Learning rate: {learning_rate}")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,}")
    
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
    
    def train(self, data_loaders, epochs=EPOCHS):
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
            
            # Print progress
            if epoch % PROGRESS_LOG_INTERVAL == 0 or epoch == epochs - FINAL_EPOCH_OFFSET:
                logger.info(f"📈 Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                checkpoint_path = f"{CHECKPOINT_BASE_DIR}/{CHECKPOINT_FILENAME}"
                logger.info(f"🎉 New best model! Improvement: {improvement:.6f} -> Best validation loss: {best_val_loss:.6f}")
                logger.debug(f"💾 Checkpoint path: {checkpoint_path}")
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                logger.debug(f"💾 New best model saved at epoch {epoch+1}")
            else:
                logger.debug(f"📊 No improvement. Current: {val_loss:.6f}, Best: {best_val_loss:.6f}")
        
        logger.info(f"✅ Stage 1 training complete! Best val loss: {best_val_loss:.6f}")
        logger.debug(f"🏁 Training summary: Total epochs: {epochs}, Final best loss: {best_val_loss:.6f}")
        return {'best_val_loss': best_val_loss}
    
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
