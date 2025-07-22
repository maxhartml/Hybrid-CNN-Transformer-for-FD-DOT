"""
Stage 2 Training: Transformer Enhancement with Frozen CNN Decoder.

This module implements the second stage of the two-stage training pipeline,
focusing on training transformer components while keeping the pre-trained
CNN decoder frozen. This approach leverages the robust feature representations
learned in Stage 1 while adding sophisticated spatial modeling capabilities.

The training process supports both baseline and enhanced modes:
- Baseline: Standard transformer training without tissue context
- Enhanced: Transformer training with tissue patch integration for improved
  spatial awareness and context-sensitive reconstruction

Classes:
    RMSELoss: Root Mean Square Error loss function for reconstruction optimization
    Stage2Trainer: Complete training pipeline for transformer enhancement

Features:
    - Frozen CNN decoder to preserve Stage 1 learned features
    - Optional tissue patch integration for enhanced spatial modeling
    - Progressive learning with reduced learning rates
    - Comprehensive checkpoint management and experiment tracking
    - Support for both baseline and enhanced training modes
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


class Stage2Trainer:
    """
    Transformer Enhancement Training Pipeline with Frozen CNN Decoder.
    
    This trainer implements the second stage of the two-stage training approach,
    focusing on training transformer components while preserving the robust
    feature representations learned during Stage 1 CNN pre-training.
    
    The training pipeline supports two modes:
    - Baseline: Standard transformer training without tissue context
    - Enhanced: Transformer training with tissue patch integration
    
    Key features:
    - Frozen CNN decoder to preserve Stage 1 learned features
    - Selective parameter optimization (only unfrozen components)
    - Optional tissue patch integration for enhanced spatial modeling
    - Reduced learning rates for stable transformer training
    - Comprehensive logging and checkpoint management
    
    Attributes:
        device (torch.device): Training device (CPU/GPU)
        learning_rate (float): Optimizer learning rate (typically lower than Stage 1)
        use_tissue_patches (bool): Whether to use tissue context enhancement
        model (HybridCNNTransformer): Complete hybrid model with frozen CNN components
        criterion (RMSELoss): Loss function for reconstruction optimization
        optimizer (torch.optim.Adam): Optimizer for unfrozen parameters only
    
    Example:
        >>> trainer = Stage2Trainer(
        ...     stage1_checkpoint_path="checkpoints/stage1_best.pth",
        ...     use_tissue_patches=True,
        ...     learning_rate=5e-5,
        ...     device="cuda"
        ... )
        >>> trainer.train(data_loaders, epochs=100)
    """
    
    def __init__(self, stage1_checkpoint_path, use_tissue_patches=True, learning_rate=5e-5, device="cpu"):
        """
        Initialize the Stage 2 trainer with pre-trained CNN components.
        
        Args:
            stage1_checkpoint_path (str): Path to Stage 1 checkpoint file containing
                                        pre-trained CNN autoencoder weights
            use_tissue_patches (bool): Whether to enable tissue patch integration
                                     for enhanced spatial modeling. Default: True
            learning_rate (float): Learning rate for transformer optimization.
                                 Typically lower than Stage 1. Default: 5e-5
            device (str): Training device ('cpu' or 'cuda'). Default: 'cpu'
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.use_tissue_patches = use_tissue_patches
        
        # Initialize model
        self.model = HybridCNNTransformer(use_tissue_patches=use_tissue_patches)
        self.model.to(self.device)
        
        # Load Stage 1 checkpoint
        self.load_stage1_checkpoint(stage1_checkpoint_path)
        
        # Freeze CNN decoder (Robin Dale's approach)
        self.freeze_cnn_decoder()
        
        # Loss and optimizer (only for unfrozen parameters)
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        
        mode = "Enhanced" if use_tissue_patches else "Baseline"
        logger.info(f"🏋️  Stage 2 Trainer initialized on {self.device} ({mode})")
        logger.info(f"📈 Learning rate: {learning_rate}")
        logger.info(f"🧬 Use tissue patches: {use_tissue_patches}")
    
    def load_stage1_checkpoint(self, checkpoint_path):
        """
        Load pre-trained Stage 1 checkpoint into the model.
        
        This method loads the CNN autoencoder weights from Stage 1 training,
        providing the foundation feature representations for Stage 2 transformer
        enhancement. The checkpoint includes model state and training metadata.
        
        Args:
            checkpoint_path (str): Path to the Stage 1 checkpoint file
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint format is invalid
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dict with compatibility for missing tissue encoder parameters
        model_state = checkpoint['model_state_dict']
        current_state = self.model.state_dict()
        
        # Only load compatible parameters (skip tissue encoder if not present in checkpoint)
        compatible_state = {}
        missing_keys = []
        
        for key, value in current_state.items():
            if key in model_state:
                # Check shape compatibility
                if model_state[key].shape == value.shape:
                    compatible_state[key] = model_state[key]
                else:
                    logger.warning(f"Shape mismatch for {key}: checkpoint {model_state[key].shape} vs model {value.shape}")
                    missing_keys.append(key)
            else:
                missing_keys.append(key)
        
        # Load compatible parameters
        self.model.load_state_dict(compatible_state, strict=False)
        
        if missing_keys:
            logger.info(f"Initialized {len(missing_keys)} new parameters (tissue encoder, etc.)")
        
        logger.info(f"📂 Loaded Stage 1 checkpoint: {checkpoint_path}")
        logger.info(f"📊 Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}, "
                   f"val_loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    def freeze_cnn_decoder(self):
        """
        Freeze CNN decoder parameters to preserve Stage 1 learned features.
        
        This method implements the core strategy of the two-stage approach by
        freezing all CNN autoencoder parameters, ensuring that the robust
        feature representations learned in Stage 1 are preserved while only
        the transformer components are optimized.
        
        The freezing strategy:
        - Disables gradient computation for all CNN autoencoder parameters
        - Reduces the parameter space for efficient transformer optimization
        - Preserves stable feature extraction capabilities
        """
        # Freeze the entire CNN autoencoder
        for param in self.model.cnn_autoencoder.parameters():
            param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"🔒 CNN decoder frozen. Trainable params: {trainable_params:,}/{total_params:,} "
                   f"({100 * trainable_params / total_params:.1f}%)")
    
    def train_epoch(self, data_loader):
        """
        Execute one complete training epoch for transformer components.
        
        This method performs forward propagation through the hybrid model,
        with tissue patch integration when enabled. Only unfrozen transformer
        parameters are updated during backpropagation.
        
        Args:
            data_loader: DataLoader containing training batches with
                        'measurements', 'volumes', and optionally 'tissue_patches'
        
        Returns:
            float: Average training loss across all batches in the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            measurements = batch['measurements'].to(self.device)
            targets = batch['volumes'].to(self.device)
            
            # Get tissue patches if using them
            tissue_patches = None
            if self.use_tissue_patches:
                tissue_patches = batch['tissue_patches'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(measurements, tissue_patches)
            
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
        Evaluate the hybrid model on the validation dataset.
        
        This method performs forward propagation without gradient computation
        to assess the performance of the enhanced model on unseen data.
        Includes tissue patch processing when enabled.
        
        Args:
            data_loader: DataLoader containing validation batches with
                        'measurements', 'volumes', and optionally 'tissue_patches'
        
        Returns:
            float: Average validation loss across all validation batches
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                measurements = batch['measurements'].to(self.device)
                targets = batch['volumes'].to(self.device)
                
                tissue_patches = None
                if self.use_tissue_patches:
                    tissue_patches = batch['tissue_patches'].to(self.device)
                
                outputs = self.model(measurements, tissue_patches)
                loss = self.criterion(outputs['reconstructed'], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, data_loaders, epochs=100):
        """
        Execute the complete Stage 2 training pipeline.
        
        This method orchestrates the full transformer enhancement training process,
        including epoch-wise training and validation, progress monitoring, and
        automatic checkpoint saving. Supports both baseline and enhanced modes.
        
        Args:
            data_loaders (dict): Dictionary containing 'train' and 'val' DataLoaders
            epochs (int): Number of training epochs to execute. Default: 100
        
        The training process:
        - Trains only unfrozen transformer parameters
        - Monitors validation loss for model selection
        - Saves mode-specific checkpoints (baseline/enhanced)
        - Provides comprehensive progress logging
        
        Example:
            >>> # Enhanced mode with tissue patches
            >>> trainer = Stage2Trainer(checkpoint_path, use_tissue_patches=True)
            >>> trainer.train(data_loaders, epochs=150)
        """
        mode = "Enhanced" if self.use_tissue_patches else "Baseline"
        logger.info(f"🏋️ Starting Stage 2 training ({mode}) for {epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(data_loaders['train'])
            
            # Validate
            val_loss = self.validate(data_loaders['val'])
            
            # Print progress
            if epoch % 10 == 0:
                logger.info(f"📈 Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mode_suffix = "enhanced" if self.use_tissue_patches else "baseline"
                self.save_checkpoint(f"checkpoints/stage2_{mode_suffix}_best.pth", epoch, val_loss)
                logger.debug(f"💾 New best model saved at epoch {epoch}")
        
        logger.info(f"✅ Stage 2 training complete! Best val loss: {best_val_loss:.6f}")
        return {'best_val_loss': best_val_loss}
    
    def save_checkpoint(self, path, epoch, val_loss):
        """
        Save model checkpoint with complete training state information.
        
        This method creates a comprehensive checkpoint containing the full model state,
        optimizer state, and Stage 2 specific metadata including tissue patch
        configuration for proper model restoration.
        
        Args:
            path (str): File path for saving the checkpoint
            epoch (int): Current training epoch number
            val_loss (float): Current validation loss value
        
        The checkpoint includes:
        - Complete model state dictionary (CNN + Transformer)
        - Optimizer state dictionary for training resumption
        - Training metadata and configuration
        - Tissue patch usage flag for proper restoration
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
            'val_loss': val_loss,
            'use_tissue_patches': self.use_tissue_patches
        }, path)
        logger.debug(f"💾 Saved checkpoint: {path}")
