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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict

from ..models.hybrid_model import HybridCNNTransformer
from ..utils.logging_config import get_training_logger

# Initialize logger for this module
logger = get_training_logger(__name__)


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
    
    def __init__(self, learning_rate=1e-4, device="cpu"):
        """
        Initialize the Stage 1 trainer with model and optimization components.
        
        Args:
            learning_rate (float): Learning rate for Adam optimizer. Default: 1e-4
            device (str): Training device ('cpu' or 'cuda'). Default: 'cpu'
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        
        # Initialize model (no tissue patches for stage 1)
        self.model = HybridCNNTransformer(use_tissue_patches=False)
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
                        'measurements' and 'volumes' keys
        
        Returns:
            float: Average training loss across all batches in the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            measurements = batch['measurements'].to(self.device)
            targets = batch['volumes'].to(self.device).permute(0, 4, 1, 2, 3)  # Convert (B, H, W, D, C) to (B, C, H, W, D)
            
            # Forward pass (no tissue patches for stage 1)
            self.optimizer.zero_grad()
            outputs = self.model(measurements, tissue_patches=None)
            
            # Compute loss
            loss = self.criterion(outputs['reconstructed'], targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, data_loader):
        """
        Evaluate the model on the validation dataset.
        
        This method performs forward propagation without gradient computation
        to assess model performance on unseen data. Used for monitoring
        training progress and implementing early stopping criteria.
        
        Args:
            data_loader: DataLoader containing validation batches with
                        'measurements' and 'volumes' keys
        
        Returns:
            float: Average validation loss across all validation batches
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                measurements = batch['measurements'].to(self.device)
                targets = batch['volumes'].to(self.device).permute(0, 4, 1, 2, 3)  # Convert (B, H, W, D, C) to (B, C, H, W, D)
                
                outputs = self.model(measurements, tissue_patches=None)
                loss = self.criterion(outputs['reconstructed'], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, data_loaders, epochs=50):
        """
        Execute the complete Stage 1 training pipeline.
        
        This method orchestrates the full training process including epoch-wise
        training and validation, progress monitoring, and automatic checkpoint
        saving for the best performing model based on validation loss.
        
        Args:
            data_loaders (dict): Dictionary containing 'train' and 'val' DataLoaders
            epochs (int): Number of training epochs to execute. Default: 50
        
        Returns:
            dict: Training results containing 'best_val_loss' for analysis
        
        Example:
            >>> data_loaders = {'train': train_loader, 'val': val_loader}
            >>> results = trainer.train(data_loaders, epochs=100)
            >>> print(f"Training completed with best loss: {results['best_val_loss']}")
        """
        logger.info(f"🚀 Starting Stage 1 training for {epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(data_loaders['train'])
            
            # Validate  
            val_loss = self.validate(data_loaders['val'])
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"📈 Epoch {epoch:3d}/{epochs}: Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"checkpoints/stage1_best.pth", epoch, val_loss)
                logger.debug(f"💾 New best model saved at epoch {epoch}")
        
        logger.info(f"✅ Stage 1 training complete! Best val loss: {best_val_loss:.6f}")
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
        import os
        # Only create directory if path has a directory component
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create directory if it's not empty (i.e., not current directory)
            os.makedirs(dir_path, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
        logger.debug(f"💾 Checkpoint saved: {path}")
