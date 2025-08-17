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
Classes:
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
from typing import Dict
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
from code.utils.standardizers import PerChannelZScore, fit_standardizer_on_dataloader
from .training_config import *  # Import all training config

# =============================================================================
# STAGE-SPECIFIC CONFIGURATION
# =============================================================================

# Initialize module logger
logger = get_training_logger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _assert_raw_chfirst(x):
    """Assert tensor is RAW [B,2,D,H,W] format for viz."""
    assert x.ndim==5 and x.shape[1]==2, f"Require RAW [B,2,D,H,W], got {tuple(x.shape)}"

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
            weight_decay (float): L2 regularization strength. Default: 7e-4
            early_stopping_patience (int): Early stopping patience in epochs. Default: 25
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
        
        # Initialize ground truth standardizer for normalized training
        self.standardizer = PerChannelZScore(device=self.device)
        self.standardizer_fitted = False
        
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
        
        # EMA for evaluation only (optional quality improvement)
        self.use_ema = globals().get("USE_EMA_EVAL", False)
        self.ema_decay = globals().get("EMA_DECAY", 0.999)
        self.ema = None
        if self.use_ema:
            from torch.optim.swa_utils import AveragedModel
            # EMA with exponential decay
            avg_fn = (lambda avg_p, p, n:
                      avg_p.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay))
            self.ema = AveragedModel(self.model, avg_fn=avg_fn)
            logger.info(f"🎯 EMA enabled for evaluation with decay={self.ema_decay}")
        
        # Loss function - Standard RMSE loss
        self.criterion = RMSELoss()
        logger.info("📊 Using standard RMSE loss")
        
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
        
        # Log stage-specific model info
        total_params = sum(p.numel() for p in self.model.parameters())
        cnn_params = sum(p.numel() for p in self.model.cnn_autoencoder.parameters())
        
        logger.info("📊 STAGE 1 PARAMETER USAGE:")
        logger.info(f"   ├─ Total Model: {total_params:,} params")
        logger.info(f"   ├─ CNN Autoencoder: {cnn_params:,} params (ACTIVE)")
        logger.info(f"   └─ Transformer Pipeline: {total_params - cnn_params:,} params (INACTIVE)")
        logger.info("🎯 Stage 1 trains ONLY the CNN autoencoder for spatial feature learning")
        
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
                "loss_function": "RMSE_Standardized",
                "ground_truth_normalization": "PerChannel_ZScore",
                
                # Model specifications (Stage 1: Autoencoder training)
                "input_data": "standardized_ground_truth_volumes",
                "input_shape": "64x64x64x2_channels",
                "target_data": "standardized_ground_truth_volumes", 
                "reconstruction_task": "autoencoder_identity_mapping_standardized",
                "use_tissue_patches": USE_TISSUE_PATCHES_STAGE1,
                "validation_metrics": "raw_space_for_interpretation",
                
                # Architecture details
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        )
        
        # Define custom metrics with proper x-axes for clean dissertation graphs
        wandb.define_metric("LR_Scheduler/*", step_metric="training_step")
        wandb.define_metric("train/*", step_metric="training_step")  # Training logs use training_step
        wandb.define_metric("val/*", step_metric="epoch")           # Validation logs use epoch
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
        # Create parameter groups with proper weight decay hygiene
        param_groups = self._create_parameter_groups()
        
        # Create AdamW optimizer with CNN-optimized parameters
        # Based on "Fixing Weight Decay Regularization in Adam" (Loshchilov & Hutter, 2019)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=STAGE1_BASE_LR,                    # Base LR (overridden by OneCycleLR)
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
    
    def _create_parameter_groups(self):
        """
        Create parameter groups for CNN training with proper weight decay hygiene.
        
        AdamW hygiene: no weight decay on norms, biases, or embeddings.
        This prevents scale drift and stabilizes training in Transformers & CNNs.
        
        Based on "Fixing Weight Decay Regularization in Adam" (Loshchilov & Hutter, 2019)
        and modern deep learning best practices.
        
        Returns:
            list: Parameter groups for AdamW optimizer
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for: LayerNorm weights, all bias terms, embedding parameters
            if (name.endswith(".bias") or 
                "norm" in name.lower() or 
                "LayerNorm" in name or 
                "layernorm" in name or 
                name.endswith("embedding.weight") or
                "pos_embed" in name or
                "token_type_embedding" in name):
                no_decay_params.append(param)
                logger.debug(f"🚫 No decay: {name}")
            else:
                decay_params.append(param)
                logger.debug(f"✅ With decay: {name}")
        
        logger.info(f"[AdamW Groups] decay: {len(decay_params)} params, no_decay: {len(no_decay_params)} params")
        
        return [
            {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    
    def _fit_standardizer_on_train_data(self, train_loader):
        """
        Fit the ground truth standardizer on training data only.
        
        This method computes per-channel mean and std from the training dataset
        and stores them for consistent normalization across train/val splits.
        The standardizer ensures stable training by normalizing μₐ and μ′ₛ
        channels independently using z-score normalization.
        
        Args:
            train_loader: Training DataLoader containing ground truth volumes
        """
        if self.standardizer_fitted:
            logger.info("📊 Standardizer already fitted - skipping")
            return
        
        logger.info("")
        logger.info("="*60)
        logger.info("📊 FITTING GROUND TRUTH STANDARDIZER")
        logger.info("="*60)
        logger.info("🔧 Computing per-channel z-score statistics from training data...")
        
        # Collect all training ground truth volumes
        all_volumes = []
        self.model.eval()  # Ensure model is in eval mode during data collection
        
        logger.info(f"📦 Processing {len(train_loader)} training batches...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                ground_truth = batch['ground_truth'].cpu()  # Keep on CPU for memory efficiency
                all_volumes.append(ground_truth)
                
                if (batch_idx + 1) % 20 == 0:
                    logger.info(f"   Processed {batch_idx + 1}/{len(train_loader)} batches...")
        
        # Concatenate all volumes and fit standardizer
        all_volumes = torch.cat(all_volumes, dim=0)
        logger.info(f"📊 Fitting standardizer on {all_volumes.shape[0]} training volumes...")
        logger.info(f"📐 Volume shape: {all_volumes.shape[1:]} (channels, depth, height, width)")
        
        # Fit standardizer and move to correct device
        self.standardizer.fit(all_volumes)
        self.standardizer.to(self.device)
        self.standardizer_fitted = True
        
        logger.info("✅ Ground truth standardizer fitted successfully!")
        logger.info("="*60)
        logger.info("")
    
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
        Execute one complete training epoch with standardized ground truth.
        
        This method trains the CNN autoencoder on z-score normalized ground truth
        volumes (μₐ, μ′ₛ) for stable optimization. Only standardized RMSE and
        learning rate are logged during training to keep console output clean.
        
        Key Changes for Stage 1 with Standardization:
        - Ground truth is standardized before forward pass
        - Loss computed in normalized space (standardized RMSE) 
        - Training logs show only standardized RMSE and LR
        - No Dice/Contrast during training (validation only)
        
        Args:
            data_loader: DataLoader containing training batches with 'ground_truth' key
            epoch (int): Current epoch number for W&B logging
        
        Returns:
            float: Average standardized RMSE loss for epoch monitoring
        """
        if not self.standardizer_fitted:
            raise RuntimeError("Standardizer must be fitted before training. Call _fit_standardizer_on_train_data() first.")
        
        logger.debug("🔄 Starting training epoch with standardized targets...")
        self.model.train()
        total_standardized_loss = 0
        num_batches = 0
        
        logger.debug(f"📊 Processing {len(data_loader)} batches in training epoch")
        
        for batch_idx, batch in enumerate(data_loader):
            logger.debug(f"🔍 Processing batch {batch_idx + 1}/{len(data_loader)}")
            
            # Get raw ground truth and standardize it
            raw_ground_truth = batch['ground_truth'].to(self.device)  # Shape: (batch_size, 2, 64, 64, 64)
            standardized_ground_truth = self.standardizer.transform(raw_ground_truth)
            
            logger.debug(f"📦 Ground truth batch shape: {raw_ground_truth.shape}")
            logger.debug(f"� Standardized ground truth range: [{standardized_ground_truth.min():.3f}, {standardized_ground_truth.max():.3f}]")
            
            # Forward pass with mixed precision for A100 optimization
            logger.debug("⚡ Starting forward pass...")
            self.optimizer.zero_grad()
            
            if self.scaler:  # Mixed precision training
                with autocast():
                    outputs = self.model(standardized_ground_truth, tissue_patches=None)
                    logger.debug(f"📤 Model output shape: {outputs['reconstructed'].shape}")
                    
                    # Compute loss in STANDARDIZED space
                    logger.debug("📏 Computing standardized RMSE loss...")
                    loss = self.criterion(outputs['reconstructed'], standardized_ground_truth)
                
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
                
                # Update EMA after successful optimizer step (mixed precision)
                took_step = True  # In mixed precision, step may be skipped on overflow, but we assume success for simplicity
                if self.use_ema and took_step and self.ema is not None:
                    self.ema.update_parameters(self.model)
            else:  # Standard precision training (CPU fallback)
                outputs = self.model(standardized_ground_truth, tissue_patches=None)
                logger.debug(f"📤 Model output shape: {outputs['reconstructed'].shape}")
                
                # Compute loss in STANDARDIZED space
                logger.debug("📏 Computing standardized RMSE loss...")
                loss = self.criterion(outputs['reconstructed'], standardized_ground_truth)
                
                # Standard backward pass
                loss.backward()
                
                # Apply gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                
                # Monitor gradient norm for training health
                if grad_norm > GRADIENT_MONITOR_THRESHOLD:
                    logger.warning(f"⚠️ High gradient norm detected: {grad_norm:.4f} > {GRADIENT_MONITOR_THRESHOLD}")
                
                self.optimizer.step()
                
                # Update EMA after successful optimizer step (standard precision)
                took_step = True
                if self.use_ema and took_step and self.ema is not None:
                    self.ema.update_parameters(self.model)
                
            # OneCycleLR updates per-batch (essential for proper LR scheduling)
            self.scheduler.step()
            
            # Log learning rate every 5 batches to avoid W&B buffer warnings
            if batch_idx % LOG_LR_EVERY_N_BATCHES == 0:
                self._log_learning_rate_to_wandb(epoch, batch_idx, len(data_loader))
            
            # Log to W&B during training: only standardized loss and LR
            if self.use_wandb and wandb.run:
                current_lr = self.optimizer.param_groups[0]['lr']
                global_step = epoch * len(data_loader) + batch_idx + 1
                
                wandb.log({
                    "training_step": global_step,
                    "train/loss_std": loss.item(),  # Standardized RMSE loss
                    "train/lr": current_lr
                })
                
            logger.debug(f"💰 Standardized batch loss: {loss.item():.6f}")
            logger.debug("✅ Optimizer step completed")
            
            total_standardized_loss += loss.item()
            num_batches += 1
            
            # Clean training batch progress: only standardized RMSE and LR
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"🏋️  TRAIN | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                       f"Std_RMSE: {loss.item():.4f} | LR: {current_lr:.2e}")
            
            # Log gradient norm at debug level for monitoring training health
            logger.debug(f"🔧 Batch {batch_idx + 1} | Gradient Norm: {grad_norm:.3f}")
            
            # Additional detailed logging at DEBUG level
            if batch_idx % 10 == 0:  # Log every 10 batches for stability monitoring
                logger.debug(f"🔍 Detailed: Batch {batch_idx}: Std_Loss = {loss.item():.6f}, "
                           f"Running Avg = {total_standardized_loss/num_batches:.6f}")
        
        avg_standardized_loss = total_standardized_loss / num_batches
        
        logger.debug(f"✅ Training epoch completed. Average standardized loss: {avg_standardized_loss:.6f}")
        logger.info(f"📊 TRAIN SUMMARY | Std_RMSE: {avg_standardized_loss:.4f}")
        
        return avg_standardized_loss
    
    def _log_reconstruction_images(self, predictions, targets, epoch, phantom_ids=None, step=None):
        """
        Log 3D reconstruction slices to W&B for visualization.
        
        VALIDATION-ONLY METHOD: This method must only be called during validation,
        never during training loops, to prevent training set leakage in visualizations.
        """
        if not self.use_wandb:
            return
            
        try:
            from code.utils import viz_recon as viz

            # Ensure channel-first [B,2,D,H,W]
            pred_std = predictions
            if pred_std.shape[-1] == 2:  # channels-last to channels-first
                pred_std = pred_std.permute(0, 4, 1, 2, 3).contiguous()
            tgt_raw = targets
            if tgt_raw.shape[-1] == 2:
                tgt_raw = tgt_raw.permute(0, 4, 1, 2, 3).contiguous()

            # Convert predictions from standardized to RAW mm⁻¹
            pred_raw = self.standardizer.inverse_transform_gt_chfirst(pred_std)

            # Safety assertions for RAW channel-first format
            _assert_raw_chfirst(pred_raw)
            _assert_raw_chfirst(tgt_raw)

            # Log telemetry for debugging
            mu_a_rng = (float(pred_raw[:,0].min()), float(pred_raw[:,0].max()))
            mu_s_rng = (float(pred_raw[:,1].min()), float(pred_raw[:,1].max()))
            logger.info(f"[VIZ RAW] pred μₐ {mu_a_rng[0]:.4f}..{mu_a_rng[1]:.4f}, μ′ₛ {mu_s_rng[0]:.3f}..{mu_s_rng[1]:.3f}")

            # Call RAW-only visualization (no standardizer dependency)
            pred_raw, tgt_raw = viz.prepare_raw_DHW(pred_raw, tgt_raw)
            viz.log_recon_slices_raw(pred_raw, tgt_raw, epoch, phantom_ids=phantom_ids, prefix="Reconstructions")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log reconstruction images: {e}")
    
    def validate(self, data_loader):
        """
        Evaluate the model on validation data with both standardized and raw metrics.
        
        This method runs the forward pass on standardized targets (like training)
        but computes human-interpretable metrics in raw space after inverse
        transformation. Logs both standardized loss and raw metrics for W&B.
        
        Key Changes for Stage 1 with Standardization:
        - Forward pass on standardized ground truth (consistency with training)
        - Inverse transform predictions and targets for raw metrics
        - Log standardized loss + raw metrics (RMSE, Dice, Contrast)
        - Console shows raw metrics for human interpretation
        
        Args:
            data_loader: DataLoader containing validation batches with 'ground_truth' key
        
        Returns:
            tuple: (standardized_loss, raw_metrics_dict) for monitoring
        """
        if not self.standardizer_fitted:
            raise RuntimeError("Standardizer must be fitted before validation. Call _fit_standardizer_on_train_data() first.")
        
        logger.debug("🔍 Starting validation epoch with standardized targets...")
        
        # Mark as validation phase - ensures visualization is validation-only
        is_validation_phase = True  # makes intent explicit for future contributors
        
        # Use EMA model for evaluation if available, otherwise use training model
        model_for_eval = self.ema.module if (self.use_ema and self.ema is not None) else self.model
        model_for_eval.eval()
        
        total_standardized_loss = 0
        num_batches = 0
        
        # Initialize RAW metrics tracking (human-interpretable)
        raw_metrics = {
            'raw_rmse_total': 0.0, 'raw_rmse_mu_a': 0.0, 'raw_rmse_mu_s': 0.0,
            'raw_dice': 0.0, 'raw_contrast': 0.0
        }
        
        logger.debug(f"📊 Processing {len(data_loader)} validation batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                logger.debug(f"🔍 Validating batch {batch_idx + 1}/{len(data_loader)}")
                
                # Get raw ground truth and standardize for forward pass
                raw_ground_truth = batch['ground_truth'].to(self.device)  # Shape: (batch_size, 2, 64, 64, 64)
                standardized_ground_truth = self.standardizer.transform(raw_ground_truth)
                
                logger.debug(f"📦 Validation batch shape: {raw_ground_truth.shape}")
                
                logger.debug("⚡ Forward pass on standardized targets (no gradients)...")
                if self.scaler:  # Mixed precision validation
                    with autocast():
                        outputs = model_for_eval(standardized_ground_truth, tissue_patches=None)
                        # Loss computed in STANDARDIZED space (consistency with training)
                        standardized_loss = self.criterion(outputs['reconstructed'], standardized_ground_truth)
                else:  # Standard precision validation
                    outputs = model_for_eval(standardized_ground_truth, tissue_patches=None)
                    # Loss computed in STANDARDIZED space (consistency with training) 
                    standardized_loss = self.criterion(outputs['reconstructed'], standardized_ground_truth)
                
                # Inverse transform predictions to RAW space for human-interpretable metrics
                raw_predictions = self.standardizer.inverse_transform(outputs['reconstructed'])
                # raw_ground_truth already in raw space (no transformation needed)
                
                logger.debug(f"💰 Standardized validation loss: {standardized_loss.item():.6f}")
                
                # Calculate RAW metrics in original μₐ/μ′ₛ space for human interpretation
                with torch.no_grad():
                    # Create outputs dict with raw predictions for metrics calculation
                    raw_outputs = {'reconstructed': raw_predictions}
                    batch_raw_metrics = calculate_batch_metrics(
                        self.metrics, raw_outputs, raw_ground_truth, "stage1"
                    )
                    
                    # Map batch metrics to our raw metrics naming
                    raw_metrics['raw_rmse_total'] += batch_raw_metrics.get('rmse_overall', 0)
                    raw_metrics['raw_rmse_mu_a'] += batch_raw_metrics.get('rmse_absorption', 0)
                    raw_metrics['raw_rmse_mu_s'] += batch_raw_metrics.get('rmse_scattering', 0)
                    raw_metrics['raw_dice'] += batch_raw_metrics.get('dice', 0)
                    raw_metrics['raw_contrast'] += batch_raw_metrics.get('contrast_ratio', 0)
                
                total_standardized_loss += standardized_loss.item()
                num_batches += 1
                
                # Show validation batch progress with RAW metrics (human-interpretable)
                logger.info(f"🔍 VALID | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                           f"Raw_RMSE: {batch_raw_metrics.get('rmse_overall', 0):.4f} | "
                           f"Dice: {batch_raw_metrics.get('dice', 0):.4f} | "
                           f"Contrast: {batch_raw_metrics.get('contrast_ratio', 0):.4f}")
        
        # Average all metrics
        avg_standardized_loss = total_standardized_loss / num_batches
        for key in raw_metrics:
            raw_metrics[key] /= num_batches
        
        logger.debug(f"✅ Validation completed. Average standardized loss: {avg_standardized_loss:.6f}")
        logger.info(f"📊 VALID SUMMARY | Raw_RMSE: {raw_metrics['raw_rmse_total']:.4f} | "
                   f"Dice: {raw_metrics['raw_dice']:.4f} | Contrast: {raw_metrics['raw_contrast']:.4f} | "
                   f"μₐ: {raw_metrics['raw_rmse_mu_a']:.4f} | μ′ₛ: {raw_metrics['raw_rmse_mu_s']:.4f}")
        
        # Log reconstruction images during validation only
        if hasattr(self, '_current_epoch') and self.use_wandb:
            try:
                # Use EMA model for visualization if available
                model_for_viz = self.ema.module if (self.use_ema and self.ema is not None) else self.model
                model_for_viz.eval()
                
                # Get a sample batch for visualization
                sample_batch = next(iter(data_loader))
                raw_ground_truth = sample_batch['ground_truth'].to(self.device)
                phantom_ids = sample_batch.get('phantom_id', torch.arange(raw_ground_truth.shape[0])).cpu().numpy()
                
                # Forward pass on standardized input using EMA model if available
                standardized_ground_truth = self.standardizer.transform(raw_ground_truth)
                with torch.no_grad():
                    outputs = model_for_viz(standardized_ground_truth, tissue_patches=None)
                
                # Use the centralized logging method with RAW targets
                self._log_reconstruction_images(
                    predictions=outputs['reconstructed'], 
                    targets=raw_ground_truth, 
                    epoch=self._current_epoch, 
                    phantom_ids=phantom_ids
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to log validation images: {e}")
        
        return avg_standardized_loss, raw_metrics
    
    def train(self, data_loaders, epochs=EPOCHS_STAGE1):
        """
        Execute the complete Stage 1 training pipeline with ground truth standardization.
        
        This method orchestrates the full training process including:
        1. Standardizer fitting on training data only
        2. Epoch-wise training on standardized targets
        3. Validation with both standardized and raw metrics
        4. Enhanced W&B logging and checkpoint management
        
        Key Changes for Standardization:
        - Fits standardizer once on training data before training begins
        - Trains on standardized targets, validates with raw metrics
        - Logs both standardized loss and raw metrics to W&B
        - Saves standardizer state in checkpoint for Stage 2 reuse
        
        Args:
            data_loaders (dict): Dictionary containing 'train' and 'val' DataLoaders
            epochs (int): Number of training epochs to execute. Default from constants
        
        Returns:
            dict: Training results containing 'best_val_loss' and standardizer info
        """
        # Initialize AdamW optimizer and OneCycleLR scheduler
        if self.optimizer is None:
            steps_per_epoch = len(data_loaders['train'])
            self._create_optimizer_and_scheduler(epochs, steps_per_epoch)
        
        # Fit standardizer on training data BEFORE training begins
        if not self.standardizer_fitted:
            self._fit_standardizer_on_train_data(data_loaders['train'])
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"🚀 STARTING STAGE 1 TRAINING | {epochs} Epochs")
        logger.info(f"{'='*80}")
        logger.debug(f"📊 Training configuration: device={self.device}, lr={self.learning_rate}, epochs={epochs}")
        logger.debug(f"📈 Data loaders: train_batches={len(data_loaders['train'])}, val_batches={len(data_loaders['val'])}")
        logger.info(f"📅 AdamW + OneCycleLR | Steps per epoch: {len(data_loaders['train'])}")
        logger.info(f"📊 Ground truth standardization: ENABLED (z-score per channel)")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"")
            logger.info(f"📅 EPOCH {epoch + 1}/{epochs}")
            logger.info(f"{'-'*40}")
            
            # Train: Update model parameters using STANDARDIZED targets
            logger.debug(f"🏋️  Beginning training phase for epoch {epoch+1}")
            train_std_loss = self.train_epoch(data_loaders['train'], epoch)
            logger.info(f"🏋️  TRAIN COMPLETE | Std_RMSE: {train_std_loss:.4f}")
            
            # Validate: Evaluate on STANDARDIZED targets, compute RAW metrics
            logger.debug(f"🔍 Beginning validation phase for epoch {epoch+1}")
            self._current_epoch = epoch  # Store current epoch for visualization
            val_std_loss, val_raw_metrics = self.validate(data_loaders['val'])
            logger.info(f"🔍 VALID COMPLETE | Std_RMSE: {val_std_loss:.4f} | Raw_RMSE: {val_raw_metrics['raw_rmse_total']:.4f}")
            
            # Log enhanced metrics to W&B with proper separation of standardized vs raw
            if self.use_wandb and wandb.run:
                wandb.log({
                    "epoch": epoch + 1,  # Custom x-axis for epoch metrics
                    
                    # === STANDARDIZED LOSSES (optimization space) ===
                    "val/loss_std": val_std_loss,           # Main validation metric for early stopping
                    
                    # === RAW METRICS (human-interpretable) ===
                    "val/raw_rmse_total": val_raw_metrics['raw_rmse_total'],
                    "val/raw_rmse_mu_a": val_raw_metrics['raw_rmse_mu_a'],
                    "val/raw_rmse_mu_s": val_raw_metrics['raw_rmse_mu_s'],
                    "val/raw_dice": val_raw_metrics['raw_dice'],
                    "val/raw_contrast": val_raw_metrics['raw_contrast'],
                    
                    # === ANALYSIS METRICS ===
                    "Analysis/Std_Train_Val_Ratio": train_std_loss / val_std_loss if val_std_loss > 0 else 1.0,
                })
                
            # Print epoch summary every epoch for important milestones
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"🚀 EPOCH {epoch+1:3d}/{epochs} SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"📈 Std Loss | Train: {train_std_loss:.4f} | Valid: {val_std_loss:.4f}")
            logger.info(f"📊 Raw RMSE | Total: {val_raw_metrics['raw_rmse_total']:.4f} | μₐ: {val_raw_metrics['raw_rmse_mu_a']:.4f} | μ′ₛ: {val_raw_metrics['raw_rmse_mu_s']:.4f}")
            logger.info(f"📊 Raw Metrics | Dice: {val_raw_metrics['raw_dice']:.4f} | Contrast: {val_raw_metrics['raw_contrast']:.4f}")
            logger.info(f"📈 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"{'='*80}")
            
            # Log GPU stats every 5 epochs
            if epoch % 5 == 0 and torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                    logger.debug(f"🖥️ GPU Memory: {gpu_memory:.1f}GB")
                except:
                    pass
            
            # Early stopping based on STANDARDIZED validation loss with min_delta
            improvement = self.best_val_loss - val_std_loss
            improved = improvement > EARLY_STOPPING_MIN_DELTA  # NEW: minimum improvement threshold
            if improved:
                self.best_val_loss = val_std_loss
                self.patience_counter = 0  # Reset patience counter
                checkpoint_path = f"{CHECKPOINT_BASE_DIR}/{CHECKPOINT_STAGE1}"
                logger.info(f"🎉 NEW BEST MODEL | Improvement: {improvement:.4f} | Best Std_RMSE: {self.best_val_loss:.4f}")
                self.save_checkpoint(checkpoint_path, epoch, val_std_loss, val_raw_metrics)
            else:
                self.patience_counter += 1
                logger.debug(f"📊 No significant improvement. Current: {val_std_loss:.6f}, Best: {self.best_val_loss:.6f}, Improvement: {improvement:.6f} < {EARLY_STOPPING_MIN_DELTA:.6f}, Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                # Check for early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"")
                    logger.info(f"🛑 EARLY STOPPING TRIGGERED")
                    logger.info(f"🔄 No significant improvement (>{EARLY_STOPPING_MIN_DELTA:.6f}) for {self.early_stopping_patience} epochs")
                    logger.info(f"🏆 Best Std_RMSE achieved: {self.best_val_loss:.4f}")
                    self.early_stopped = True
                    break
        
        # Training completion message
        logger.info(f"")
        logger.info(f"{'='*80}")
        if self.early_stopped:
            logger.info(f"✅ STAGE 1 TRAINING COMPLETED (Early Stopped)")
        else:
            logger.info(f"✅ STAGE 1 TRAINING COMPLETED (Full {epochs} Epochs)")
        logger.info(f"🏆 Best Std_RMSE Loss: {self.best_val_loss:.4f}")
        logger.info(f"📊 Final Epoch: {epoch+1}")
        logger.info(f"{'='*80}")
        
        logger.debug(f"🏁 Training summary: Completed epochs: {epoch+1}, Final best loss: {self.best_val_loss:.6f}")
        
        # Finish W&B run
        if self.use_wandb and wandb.run:
            wandb.log({"System/final_best_val_loss": self.best_val_loss, "System/early_stopped": self.early_stopped}, commit=False)
            wandb.finish()
            logger.info("🔬 W&B experiment finished")
        
        return {
            'best_val_loss': self.best_val_loss, 
            'early_stopped': self.early_stopped,
            'standardizer_fitted': self.standardizer_fitted
        }
    
    def save_checkpoint(self, path, epoch, val_loss, val_raw_metrics=None):
        """
        Save model checkpoint with training state and standardizer information.
        
        This method creates a comprehensive checkpoint containing the model state,
        optimizer state, standardizer state, and training metadata for resuming 
        training or transferring to Stage 2 training.
        
        Args:
            path (str): File path for saving the checkpoint
            epoch (int): Current training epoch number
            val_loss (float): Current validation loss value (standardized space)
            val_raw_metrics (dict, optional): Raw validation metrics for reference
        
        The checkpoint includes:
        - Model state dictionary (learned parameters)
        - Optimizer state dictionary (for training resumption)
        - Standardizer state dictionary (normalization parameters)
        - Training metadata (epoch, validation loss, raw metrics)
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
            'val_loss': val_loss,  # Standardized validation loss
            'standardizer': self.standardizer.state_dict() if self.standardizer_fitted else None,
            'standardizer_fitted': self.standardizer_fitted,
        }
        
        # Add raw metrics if provided
        if val_raw_metrics is not None:
            checkpoint_data['val_raw_metrics'] = val_raw_metrics
        
        torch.save(checkpoint_data, path)
        logger.info(f"💾 ✅ CHECKPOINT SAVED | Path: {path} | Epoch: {epoch+1} | Std_Loss: {val_loss:.6f}")
        if val_raw_metrics:
            logger.info(f"📊 Raw metrics saved: RMSE={val_raw_metrics.get('raw_rmse_total', 0):.4f}, Dice={val_raw_metrics.get('raw_dice', 0):.4f}")
        logger.debug(f"📊 Checkpoint data keys: {list(checkpoint_data.keys())}")
