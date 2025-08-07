#!/usr/bin/env python3
"""
Stage 2 Training: Transformer Enhancement with Frozen CNN Decoder.

This module implements the second stage of the two-stage training pipeline,
focusing on training transformer components while keeping the pre-trained
CNN decoder frozen. This approach leverages the robust feature representations
learned in Stage 1 while adding sophisticated spatial modeling capabilities.

The training process supports both baseline and enhanced modes:
- Baseline: Transformer training without tissue context
- Enhanced: Transformer training with tissue patch integration for improved
  spatial awareness and context-sensitive reconstruction

Classes:
    Stage2Trainer: Complete training pipeline for transformer enhancement

Features:
    - Frozen CNN decoder to preserve Stage 1 learned features
    - Optional tissue patch integration for enhanced spatial modeling
    - Progressive learning with reduced learning rates
    - Comprehensive checkpoint management and experiment tracking
    - Support for both baseline and enhanced training modes

Author: Max Hart
Date: August 2025
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
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training for A100 optimization
import wandb
import numpy as np
from datetime import datetime

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
    
    def __init__(self, stage1_checkpoint_path, use_tissue_patches=USE_TISSUE_PATCHES_STAGE2, 
                 learning_rate=LEARNING_RATE_STAGE2, device=CPU_DEVICE, use_wandb=True,
                 weight_decay=WEIGHT_DECAY, early_stopping_patience=EARLY_STOPPING_PATIENCE):
        """
        Initialize the Stage 2 trainer with pre-trained CNN components.
        
        Args:
            stage1_checkpoint_path (str): Path to Stage 1 checkpoint file containing
                                        pre-trained CNN autoencoder weights
            use_tissue_patches (bool): Whether to enable tissue patch integration
                                     for enhanced spatial modeling. Default from constants
            learning_rate (float): Learning rate for transformer optimization.
                                 Typically lower than Stage 1. Default from constants
            device (str): Training device ('cpu' or 'cuda'). Default from constants
            use_wandb (bool): Whether to use Weights & Biases logging. Default: True
            weight_decay (float): L2 regularization strength. Default: 1e-4
            early_stopping_patience (int): Early stopping patience in epochs. Default: 8
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.use_tissue_patches = use_tissue_patches
        self.use_wandb = use_wandb
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Initialize enhanced metrics for Stage 2 (includes feature analysis)
        self.metrics = create_metrics_for_stage("stage2")
        
        # Initialize model
        self.model = HybridCNNTransformer(
            use_tissue_patches=use_tissue_patches,
            training_stage=TRAINING_STAGE2  # IMPORTANT: Set to stage 2 for correct forward pass
        )
        self.model.to(self.device)
        
        # Load Stage 1 checkpoint
        self.load_stage1_checkpoint(stage1_checkpoint_path)
        
        # Freeze CNN decoder (Robin Dale's approach)
        self.freeze_cnn_decoder()
        
        # Loss and optimizer (only for unfrozen parameters) with L2 regularization
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], 
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
        
        # Mixed precision training for A100 optimization (2x speedup + memory savings)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        mode = ENHANCED_MODE if use_tissue_patches else BASELINE_MODE
        logger.info(f"🏋️  Stage 2 Trainer initialized on {self.device} ({mode})")
        logger.info(f"📈 Learning rate: {learning_rate}")
        logger.info(f"🔒 L2 regularization (weight decay): {weight_decay}")
        logger.info(f"⏰ Early stopping patience: {early_stopping_patience}")
        logger.info(f"🧬 Use tissue patches: {use_tissue_patches}")
        if self.scaler:
            logger.info(f"🚀 Mixed precision training enabled for A100 optimization!")
        
        # Log GPU info if available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"🔧 Current batch size: {BATCH_SIZE_STAGE2}")
        else:
            logger.info(f"💻 Running on CPU")
        
        # Initialize Weights & Biases
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases experiment tracking for Stage 2."""
        mode_suffix = "enhanced" if self.use_tissue_patches else "baseline"
        experiment_name = f"stage2_transformer_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Choose tags based on mode
        tags = WANDB_TAGS_STAGE2_ENHANCED if self.use_tissue_patches else WANDB_TAGS_STAGE2_BASELINE
        
        wandb.init(
            project=WANDB_PROJECT,
            name=experiment_name,
            tags=tags,
            config={
                # Model architecture
                "stage": "Transformer_Enhancement",
                "model_type": "Hybrid_CNN_Transformer",
                "training_stage": TRAINING_STAGE2,
                "mode": ENHANCED_MODE if self.use_tissue_patches else BASELINE_MODE,
                
                # Training hyperparameters
                "learning_rate": self.learning_rate,
                "device": str(self.device),
                "optimizer": "Adam",
                "loss_function": "RMSE",
                
                # Model specifications (Stage 2: NIR measurements → transformer → decoder)
                "input_data": "nir_measurements",
                "input_measurements": "256_subsampled_from_1000",
                "input_shape": "256_measurements_per_phantom_subsampled",
                "generation_strategy": "1000_measurements_generated_50_sources_x_20_detectors",
                "data_augmentation": "random_subsampling_from_1000_to_256",
                "target_data": "ground_truth_volumes",
                "output_voxels": "64x64x64x2_channels",
                "use_tissue_patches": self.use_tissue_patches,
                "frozen_cnn_decoder": True,
                
                # Architecture details
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
            }
        )
        logger.info(f"🔬 W&B experiment initialized: {experiment_name}")
    
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
        logger.debug("🔒 Starting CNN parameter freezing process...")
        
        # Freeze the entire CNN autoencoder
        frozen_params = 0
        for name, param in self.model.cnn_autoencoder.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            logger.debug(f"🔒 Frozen: {name} ({param.numel():,} params)")
        
        # Count trainable parameters after freezing
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"🔒 CNN decoder frozen. Frozen: {frozen_params:,}, Trainable: {trainable_params:,}/{total_params:,} "
                   f"({100 * trainable_params / total_params:.1f}%)")
        
        # Verify we have trainable parameters
        if trainable_params == 0:
            logger.error("🚨 ERROR: No trainable parameters found! All parameters are frozen!")
            raise RuntimeError("No trainable parameters - all model parameters are frozen")
        
        logger.info(f"✅ Parameter freezing completed successfully")
    
    def train_epoch(self, data_loader):
        """
        Execute one complete training epoch for transformer components with enhanced metrics.
        
        This method performs forward propagation through the hybrid model,
        with tissue patch integration when enabled. Only unfrozen transformer
        parameters are updated during backpropagation.
        
        Args:
            data_loader: DataLoader containing training batches with
                        'measurements', 'volumes', and optionally 'tissue_patches'
        
        Returns:
            tuple: (Average training loss, metrics dictionary) across all batches
        """
        logger.debug("🔄 Starting Stage 2 training epoch...")
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Initialize metrics tracking (includes feature analysis for Stage 2)
        epoch_metrics = {
            'ssim': 0.0, 'psnr': 0.0, 'rmse_overall': 0.0,
            'rmse_absorption': 0.0, 'rmse_scattering': 0.0,
            'feature_enhancement_ratio': 0.0, 'attention_entropy': 0.0
        }
        
        logger.debug(f"📊 Processing {len(data_loader)} batches in Stage 2 training epoch")
        
        for batch_idx, batch in enumerate(data_loader):
            logger.debug(f"🔍 Processing Stage 2 batch {batch_idx + 1}/{len(data_loader)}")
            
            # In Stage 2: Complete phantom NIR measurements are input, ground truth volumes are target
            nir_measurements = batch['nir_measurements'].to(self.device)  # Shape: (batch_size, 256_subsampled, 8)
            targets = batch['ground_truth'].to(self.device)               # Shape: (batch_size, 2, 64, 64, 64)
            
            logger.debug(f"📦 NIR measurements shape: {nir_measurements.shape}")
            logger.debug(f"📦 Ground truth targets shape: {targets.shape}")
            logger.debug(f"🖥️  Data moved to device: {nir_measurements.device}")
            
            # Get tissue patches if using them (not yet implemented for complete phantom format)
            tissue_patches = None
            if self.use_tissue_patches and 'tissue_patches' in batch:
                tissue_patches = batch['tissue_patches'].to(self.device)
                logger.debug(f"🧬 Using tissue patches: {tissue_patches.shape}")
            else:
                logger.debug("🧬 No tissue patches used")
            
            # Forward pass through hybrid model with mixed precision
            logger.debug("⚡ Starting Stage 2 forward pass (NIR → features → reconstruction)...")
            self.optimizer.zero_grad()
            
            with autocast():
                # The hybrid model handles: NIR measurements (batch, 256_subsampled, 8) → transformer → CNN decoder → reconstruction
                # Note: 256 measurements are subsampled from 1000 generated measurements for data augmentation
                outputs = self.model(nir_measurements, tissue_patches)
                logger.debug(f"📤 Stage 2 model output shape: {outputs['reconstruction'].shape}")
                
                # Compute loss
                logger.debug("📏 Computing Stage 2 RMSE loss...")
                loss = self.criterion(outputs['reconstruction'], targets)
                logger.debug(f"💰 Stage 2 batch loss: {loss.item():.6f}")
            
            # Backward pass with mixed precision scaling
            logger.debug("🔙 Starting Stage 2 backward pass (only transformer gradients)...")
            try:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                logger.debug("✅ Stage 2 mixed precision optimizer step completed")
            except RuntimeError as e:
                logger.error(f"🚨 Gradient error: {e}")
                raise e
            
            total_loss += loss.item()
            num_batches += 1
            
            # Show batch progress at INFO level (every batch)
            mode = "Enhanced" if self.use_tissue_patches else "Baseline"
            logger.info(f"📈 Stage 2 {mode} Batch {batch_idx + 1}/{len(data_loader)}: Loss = {loss.item():.6f}, Avg = {total_loss/num_batches:.6f}")
            
            # Additional detailed logging at DEBUG level
            if batch_idx % 5 == 0:  # Log every 5 batches during DEBUG
                logger.debug(f"� Detailed: Stage 2 Batch {batch_idx}: Loss = {loss.item():.6f}, Running Avg = {total_loss/num_batches:.6f}")
        
        avg_loss = total_loss / num_batches
        logger.debug(f"✅ Stage 2 training epoch completed. Average loss: {avg_loss:.6f}")
        return avg_loss
    
    def _log_reconstruction_images(self, predictions, targets, nir_measurements, epoch):
        """Log 3D reconstruction slices to W&B for Stage 2 visualization."""
        if not self.use_wandb:
            return
            
        try:
            # Take middle slices of first batch item for visualization
            pred_batch = predictions[0].cpu().numpy()  # First item in batch
            target_batch = targets[0].cpu().numpy()
            
            logger.debug(f"Logging Stage 2 images - Pred shape: {pred_batch.shape}, Target shape: {target_batch.shape}")
            
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
            
            # Calculate reconstruction error map for analysis
            error_xy = np.abs(pred_xy - target_xy)
            error_xz = np.abs(pred_xz - target_xz) 
            error_yz = np.abs(pred_yz - target_yz)
            
            # Normalize all images for proper W&B display
            pred_xy_norm = normalize_for_display(pred_xy)
            target_xy_norm = normalize_for_display(target_xy)
            error_xy_norm = normalize_for_display(error_xy)
            pred_xz_norm = normalize_for_display(pred_xz)
            target_xz_norm = normalize_for_display(target_xz)
            error_xz_norm = normalize_for_display(error_xz)
            pred_yz_norm = normalize_for_display(pred_yz)
            target_yz_norm = normalize_for_display(target_yz)
            error_yz_norm = normalize_for_display(error_yz)
            
            mode = "Enhanced" if self.use_tissue_patches else "Baseline"
            wandb.log({
                f"Reconstructions/epoch_{epoch}/predicted_xy_slice": wandb.Image(pred_xy_norm),
                f"Reconstructions/epoch_{epoch}/target_xy_slice": wandb.Image(target_xy_norm),
                f"Reconstructions/epoch_{epoch}/error_xy_slice": wandb.Image(error_xy_norm),
                f"Reconstructions/epoch_{epoch}/predicted_xz_slice": wandb.Image(pred_xz_norm),
                f"Reconstructions/epoch_{epoch}/target_xz_slice": wandb.Image(target_xz_norm),
                f"Reconstructions/epoch_{epoch}/error_xz_slice": wandb.Image(error_xz_norm),
                f"Reconstructions/epoch_{epoch}/predicted_yz_slice": wandb.Image(pred_yz_norm),
                f"Reconstructions/epoch_{epoch}/target_yz_slice": wandb.Image(target_yz_norm),
                f"Reconstructions/epoch_{epoch}/error_yz_slice": wandb.Image(error_yz_norm),
            })
            
            logger.debug(f"✅ Successfully logged Stage 2 reconstruction images for epoch {epoch}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log Stage 2 reconstruction images: {e}")
            logger.debug(f"Error details: {str(e)}")
    
    def validate(self, data_loader):
        """
        Evaluate the hybrid model on the validation dataset with enhanced metrics.
        
        This method performs forward propagation without gradient computation
        to assess the performance of the enhanced model on unseen data.
        Includes tissue patch processing when enabled and collects transformer-specific metrics.
        
        Args:
            data_loader: DataLoader containing validation batches with
                        'nir_measurements', 'ground_truth', and optionally 'tissue_patches'
        
        Returns:
            tuple: (average_validation_loss, metrics_dict)
        """
        logger.debug("🔍 Starting Stage 2 validation epoch...")
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Initialize metrics tracking (includes feature analysis for Stage 2)
        epoch_metrics = {
            'ssim': 0.0, 'psnr': 0.0, 'rmse_overall': 0.0,
            'rmse_absorption': 0.0, 'rmse_scattering': 0.0,
            'feature_enhancement_ratio': 0.0, 'attention_entropy': 0.0
        }
        
        logger.debug(f"📊 Processing {len(data_loader)} Stage 2 validation batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                logger.debug(f"🔍 Validating Stage 2 batch {batch_idx + 1}/{len(data_loader)}")
                
                nir_measurements = batch['nir_measurements'].to(self.device)  # Shape: (batch_size, 256_subsampled, 8)
                targets = batch['ground_truth'].to(self.device)               # Shape: (batch_size, 2, 64, 64, 64)
                
                logger.debug(f"📦 Stage 2 validation NIR shape: {nir_measurements.shape}")
                logger.debug(f"📦 Stage 2 validation target shape: {targets.shape}")
                
                tissue_patches = None
                if self.use_tissue_patches and 'tissue_patches' in batch:
                    tissue_patches = batch['tissue_patches'].to(self.device)
                    logger.debug(f"🧬 Validation tissue patches: {tissue_patches.shape}")
                else:
                    logger.debug("🧬 No tissue patches in validation")
                
                logger.debug("⚡ Stage 2 validation forward pass (no gradients)...")
                with autocast():
                    outputs = self.model(nir_measurements, tissue_patches)
                    loss = self.criterion(outputs['reconstruction'], targets)
                    logger.debug(f"💰 Stage 2 validation batch loss: {loss.item():.6f}")
                
                # Calculate enhanced metrics including feature analysis
                batch_metrics = calculate_batch_metrics(
                    self.metrics, outputs, targets, "stage2"
                )
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                
                total_loss += loss.item()
                num_batches += 1
                
                # Show validation batch progress with enhanced metrics
                mode = "Enhanced" if self.use_tissue_patches else "Baseline"
                logger.info(f"🔍 Stage 2 {mode} Val Batch {batch_idx + 1}/{len(data_loader)}: "
                           f"Loss = {loss.item():.6f}, SSIM = {batch_metrics.get('ssim', 0):.4f}, "
                           f"Enhancement = {batch_metrics.get('feature_enhancement_ratio', 0):.4f}")
        
        avg_loss = total_loss / num_batches
        
        # Average metrics across epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        logger.debug(f"✅ Stage 2 validation completed. Average loss: {avg_loss:.6f}")
        logger.info(f"📊 Stage 2 val metrics - SSIM: {epoch_metrics['ssim']:.4f}, "
                   f"PSNR: {epoch_metrics['psnr']:.2f}dB, Enhancement: {epoch_metrics['feature_enhancement_ratio']:.4f}")
        
        return avg_loss, epoch_metrics
    
    def train(self, data_loaders, epochs=EPOCHS_STAGE2):
        """
        Execute the complete Stage 2 training pipeline.
        
        This method orchestrates the full transformer enhancement training process,
        including epoch-wise training and validation, progress monitoring, and
        automatic checkpoint saving. Supports both baseline and enhanced modes.
        
        Args:
            data_loaders (dict): Dictionary containing 'train' and 'val' DataLoaders
            epochs (int): Number of training epochs to execute. Default from constants
        
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
        mode = ENHANCED_MODE if self.use_tissue_patches else BASELINE_MODE
        logger.info(f"🏋️ Starting Stage 2 training ({mode}) for {epochs} epochs")
        logger.debug(f"📊 Stage 2 configuration: device={self.device}, lr={self.learning_rate}, tissue_patches={self.use_tissue_patches}")
        logger.debug(f"📈 Stage 2 data loaders: train_batches={len(data_loaders['train'])}, val_batches={len(data_loaders['val'])}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"📅 Starting Stage 2 Epoch {epoch + 1}/{epochs}")
            
            # Train: Update transformer parameters (CNN decoder frozen)
            logger.debug(f"🏋️  Beginning Stage 2 training phase for epoch {epoch+1}")
            train_loss, train_metrics = self.train_epoch(data_loaders['train'])
            logger.info(f"🏋️  Stage 2 Training completed - Average Loss: {train_loss:.6f}")
            
            # Validate: Evaluate hybrid model on unseen data (no parameter updates)
            logger.debug(f"🔍 Beginning Stage 2 validation phase for epoch {epoch+1}")
            val_loss, val_metrics = self.validate(data_loaders['val'])
            logger.info(f"🔍 Stage 2 Validation completed - Average Loss: {val_loss:.6f}")
            
            # Log enhanced metrics to W&B
            if self.use_wandb:
                mode = "Enhanced" if self.use_tissue_patches else "Baseline"
                
                # Log training metrics
                train_metrics['loss'] = train_loss
                self.metrics.log_to_wandb(train_metrics, epoch + 1, "train", self.use_wandb)
                
                # Log validation metrics
                val_metrics['loss'] = val_loss
                self.metrics.log_to_wandb(val_metrics, epoch + 1, "val", self.use_wandb)
                
                # Legacy charts for backward compatibility
                wandb.log({
                    "Charts/train_loss": train_loss,
                    "Charts/val_loss": val_loss,
                    "Charts/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "Charts/train_val_loss_ratio": train_loss / val_loss if val_loss > 0 else 0,
                    "Charts/ssim_diff": val_metrics['ssim'] - train_metrics['ssim'],
                    "Charts/psnr_diff": val_metrics['psnr'] - train_metrics['psnr'],
                    "Charts/enhancement_ratio": val_metrics['feature_enhancement_ratio'],
                    "Charts/attention_entropy": val_metrics['attention_entropy'],
                    "System/mode": mode,
                }, step=epoch + 1)
                
                # Log reconstruction images periodically (and always on first/last epoch)
                should_log_images = (epoch % LOG_IMAGES_EVERY == 0) or (epoch == 0) or (epoch == epochs - 1)
                if should_log_images:
                    try:
                        sample_batch = next(iter(data_loaders['val']))
                        # Stage 2 uses NIR measurements as input and ground truth as target
                        measurements = sample_batch['nir_measurements'].to(self.device)  # Fixed key
                        targets = sample_batch['ground_truth'].to(self.device)          # Fixed key
                        tissue_patches = sample_batch.get('tissue_patches', None)
                        if tissue_patches is not None:
                            tissue_patches = tissue_patches.to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(measurements, tissue_patches)
                        self._log_reconstruction_images(outputs['reconstruction'], targets, measurements, epoch + 1)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to log Stage 2 images at epoch {epoch + 1}: {e}")
            
            # Print progress
            if epoch % PROGRESS_LOG_INTERVAL == 0 or epoch == epochs - 1:
                logger.info(f"📈 Stage 2 Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                checkpoint_filename = CHECKPOINT_STAGE2_ENHANCED if self.use_tissue_patches else CHECKPOINT_STAGE2_BASELINE
                checkpoint_path = f"{CHECKPOINT_BASE_DIR}/{checkpoint_filename}"
                logger.info(f"🎉 New best Stage 2 model! Improvement: {improvement:.6f} -> Saving checkpoint")
                logger.debug(f"💾 Stage 2 checkpoint path: {checkpoint_path}")
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                logger.debug(f"💾 New best Stage 2 model saved at epoch {epoch}")
            else:
                logger.debug(f"📊 Stage 2 no improvement. Current: {val_loss:.6f}, Best: {best_val_loss:.6f}")
        
        logger.info(f"✅ Stage 2 training complete! Best val loss: {best_val_loss:.6f}")
        logger.debug(f"🏁 Stage 2 training summary: Total epochs: {epochs}, Final best loss: {best_val_loss:.6f}")
        
        # Finish W&B run
        if self.use_wandb:
            mode = "enhanced" if self.use_tissue_patches else "baseline"
            mode = "Enhanced" if self.use_tissue_patches else "Baseline"
            wandb.log({f"System/final_best_val_loss": best_val_loss, f"System/final_mode": mode})
            wandb.finish()
            logger.info("🔬 W&B Stage 2 experiment finished")
        
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
