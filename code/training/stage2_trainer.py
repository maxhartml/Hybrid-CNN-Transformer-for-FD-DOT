#!/usr/bin/env python3
"""
Stage 2 Training: Transformer Enhancement with Frozen CNN Decoder.

This module implements the second stage of the two-stage training pipeline,
focusing on training transformer components while keeping the pre-trained
CNN decoder frozen. This approach leverages the robust feature representations
learned in Stage 1 while adding sophisticated spatial modeling capabilities.

The training pipeline supports both baseline and enhanced modes:
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
import math
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
        ...     learning_rate=STAGE2_BASE_LR,
        ...     device="cuda"
        ... )
        >>> trainer.train(data_loaders, epochs=EPOCHS_STAGE2)
    """
    
    def __init__(self, stage1_checkpoint_path, use_tissue_patches=USE_TISSUE_PATCHES_STAGE2, 
                 learning_rate=STAGE2_BASE_LR, device=CPU_DEVICE, use_wandb=True,
                 early_stopping_patience=EARLY_STOPPING_PATIENCE):
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
            early_stopping_patience (int): Early stopping patience in epochs. Default: 8
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.use_tissue_patches = use_tissue_patches
        self.use_wandb = use_wandb
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
            training_stage=TRAINING_STAGE2,  # IMPORTANT: Set to stage 2 for correct forward pass
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
        
        # Load Stage 1 checkpoint
        self.load_stage1_checkpoint(stage1_checkpoint_path)
        
        # Freeze CNN decoder (Robin Dale's approach)
        self.freeze_cnn_decoder()
        
        # Loss function - Choose between standard RMSE and channel-weighted RMSE (same as Stage 1)
        if USE_CHANNEL_WEIGHTED_LOSS:
            from code.utils.metrics import ChannelWeightedRMSELoss
            self.criterion = ChannelWeightedRMSELoss(channel_weights=CHANNEL_WEIGHTS)
            logger.info(f"🎯 Stage 2 using channel-weighted RMSE loss with weights: {CHANNEL_WEIGHTS}")
        else:
            self.criterion = RMSELoss()
            logger.info("📊 Stage 2 using standard RMSE loss")
        
        # NOTE: Optimizer and scheduler are created in this stage using research-validated
        # AdamW + Linear Warmup + Cosine Decay for optimal transformer fine-tuning.
        self.optimizer = None
        self.scheduler = None
        
        # Mixed precision training for A100 optimization (2x speedup + memory savings)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        mode = "enhanced" if use_tissue_patches else "baseline"
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"🚀 TRANSFORMER TRAINING INITIALIZATION ({mode})")
        logger.info(f"{'='*80}")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"📈 Base Learning Rate: {learning_rate}")
        logger.info(f"🔒 L2 Regularization: {weight_decay}")
        logger.info(f"⏰ Early Stopping Patience: {early_stopping_patience}")
        logger.info(f"🧬 Tissue Patches: {use_tissue_patches}")
        if self.scaler:
            logger.info(f"🚀 Mixed Precision: Enabled (A100 Optimized)")
        logger.info(f"📅 Scheduler: Will be created during training initialization")
        
        # Log GPU info if available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"🔧 Current Batch Size: {BATCH_SIZE}")
        else:
            logger.info(f"💻 CPU Mode: Enabled")
        logger.info(f"{'='*80}")
        
        # Initialize Weights & Biases (deferred until training starts)
        # Note: W&B initialization happens in train() method when epochs and data_loader are available
        self._wandb_initialized = False
    
    def _init_wandb(self, epochs: int, steps_per_epoch: int):
        """Initialize Weights & Biases experiment tracking for Stage 2."""
        mode_suffix = "Enhanced" if self.use_tissue_patches else "Baseline"
        experiment_name = f"Transformer_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Choose tags based on mode
        tags = WANDB_TAGS_STAGE2_ENHANCED if self.use_tissue_patches else WANDB_TAGS_STAGE2_BASELINE
        
        # Calculate warmup steps for logging
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(STAGE2_WARMUP_PCT * total_steps)
        
        wandb.init(
            project=WANDB_PROJECT,
            name=experiment_name,
            tags=tags,
            config={
                # Model architecture
                "stage": "Transformer_Enhancement",
                "model_type": "Hybrid_CNN_Transformer", 
                "training_stage": "Stage_2_Sequence_Modeling",
                "mode": "enhanced" if self.use_tissue_patches else "baseline",
                
                # Training hyperparameters
                "learning_rate": self.learning_rate,
                "device": str(self.device),
                "optimizer": "AdamW",
                "optimizer_betas": ADAMW_BETAS_STAGE2,
                "scheduler": "LinearWarmupCosineDecay",
                "warmup_steps": warmup_steps,
                "warmup_pct": STAGE2_WARMUP_PCT,
                "max_steps": total_steps,
                "weight_decay": WEIGHT_DECAY_TRANSFORMER,
                "loss_function": "RMSE",
                
                # Model specifications (Transformer: NIR measurements → transformer → decoder)
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
        
        # Define separate x-axes for different metric types to avoid step conflicts
        wandb.define_metric("training_step")  # Custom x-axis for learning rate
        wandb.define_metric("epoch")  # Custom x-axis for all other metrics
        
        # Map metric categories to their respective x-axes
        wandb.define_metric("LR_Scheduler/*", step_metric="training_step")
        wandb.define_metric("Metrics/*", step_metric="epoch")
        wandb.define_metric("RMSE_Details/*", step_metric="epoch")
        wandb.define_metric("System/*", step_metric="epoch")
        wandb.define_metric("Analysis/*", step_metric="epoch")
        wandb.define_metric("Reconstructions/*", step_metric="epoch")
        
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
    
    def _create_parameter_groups(self):
        """
        Create parameter groups for transformer training with differential weight decay.
        
        This method implements the standard transformer training approach where
        different parameter types receive different weight decay values:
        - Layer norms and biases: No weight decay (prevents regularization of scale/shift)
        - All other parameters: Standard weight decay for regularization
        
        Based on BERT, GPT, and ViT training procedures.
        
        Returns:
            list: Parameter groups for AdamW optimizer
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No weight decay for layer norms, biases, and position embeddings
                if ('bias' in name or 
                    'LayerNorm' in name or 
                    'norm' in name or 
                    'pos_embed' in name):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        logger.info(f"📊 Parameter Groups:")
        logger.info(f"   ├─ With weight decay: {len(decay_params)} groups")
        logger.info(f"   └─ No weight decay: {len(no_decay_params)} groups")
        
        return [
            {'params': decay_params, 'weight_decay': WEIGHT_DECAY_TRANSFORMER},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    
    def _create_optimizer_and_scheduler(self, epochs: int, steps_per_epoch: int):
        """
        Create AdamW optimizer and Linear Warmup + Cosine Decay scheduler for Stage 2.
        
        This method implements research-validated optimization for transformer fine-tuning
        based on "Attention Is All You Need", BERT, and ViT papers.
        
        AdamW Configuration:
        - Separate parameter groups for differential weight decay
        - Transformer-optimized betas
        - Conservative learning rate for fine-tuning
        
        Linear Warmup + Cosine Decay Configuration:
        - 10% warmup (transformer standard)
        - Smooth cosine decay to 3% of peak LR
        - No aggressive exploration (preserves frozen CNN features)
        
        Args:
            epochs (int): Total training epochs
            steps_per_epoch (int): Batches per epoch
            
        Returns:
            tuple: (optimizer, scheduler) configured for Stage 2
        """
        # Create parameter groups for differential weight decay
        param_groups = self._create_parameter_groups()
        
        # Create AdamW optimizer with transformer-optimized parameters
        # Based on "Fixing Weight Decay Regularization in Adam" (Loshchilov & Hutter, 2019)
        # and transformer training best practices
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=STAGE2_BASE_LR,                   # Conservative for fine-tuning
            betas=ADAMW_BETAS_STAGE2,           # Transformer-standard betas (0.9, 0.98)
            eps=ADAMW_EPS_STAGE2                # Numerical stability
        )
        
        # Create Linear Warmup + Cosine Decay scheduler
        # Based on "Attention Is All You Need" and BERT/ViT training procedures
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(STAGE2_WARMUP_PCT * total_steps)
        
        def get_cosine_schedule_with_warmup(step):
            """Learning rate schedule function for transformer fine-tuning."""
            if step < warmup_steps:
                # Linear warmup: 0 → peak_lr
                return step / max(1, warmup_steps)
            
            # Cosine decay: peak_lr → eta_min
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            eta_min = STAGE2_ETA_MIN_PCT  # Final LR = 3% of peak
            return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            get_cosine_schedule_with_warmup
        )
        
        logger.info(f"🚀 STAGE 2 ADAMW OPTIMIZER:")
        logger.info(f"   ├─ Base LR: {STAGE2_BASE_LR:.0e}")
        logger.info(f"   ├─ Weight Decay (Transformer): {WEIGHT_DECAY_TRANSFORMER}")
        logger.info(f"   └─ Betas: {ADAMW_BETAS_STAGE2}")
        
        logger.info(f"🚀 STAGE 2 LINEAR WARMUP + COSINE DECAY:")
        logger.info(f"   ├─ Total Steps: {total_steps:,}")
        logger.info(f"   ├─ Warmup Steps: {warmup_steps:,} ({STAGE2_WARMUP_PCT*100:.0f}%)")
        logger.info(f"   ├─ Final LR: {STAGE2_ETA_MIN_PCT*100:.0f}% of peak")
        logger.info(f"   └─ Schedule: Linear Warmup → Cosine Decay")
        
        return self.optimizer, self.scheduler
    
    def _log_learning_rate_to_wandb(self, epoch: int, batch_idx: int, total_batches: int):
        """
        Log current learning rate to W&B for Linear Warmup + Cosine Decay visualization.
        
        Unlike OneCycleLR, this scheduler is designed for per-batch updates with
        smooth, monotonic progression suitable for transformer fine-tuning.
        
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
            
            # Calculate global step for scheduler tracking (dissertation graphs need this!)
            global_step = epoch * total_batches + batch_idx + 1  # Start from 1, not 0
            
            # Log detailed learning rate for smooth LinearWarmupCosineDecay visualization
            # Essential for dissertation graphs showing scheduler behavior
            wandb.log({
                "LR_Scheduler/Learning_Rate": current_lr,
                "LR_Scheduler/Training_Progress": epoch + (batch_idx / total_batches),
                "training_step": global_step
            })
                
        except Exception as e:
            logger.debug(f"W&B LR logging failed: {e}")
    
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
    
    def train_epoch(self, data_loader, epoch=0):
        """
        Execute one complete training epoch for transformer components with enhanced metrics.
        
        This method performs forward propagation through the hybrid model,
        with tissue patch integration when enabled. Only unfrozen transformer
        parameters are updated during backpropagation.
        
        Args:
            data_loader: DataLoader containing training batches with
                        'measurements', 'volumes', and optionally 'tissue_patches'
            epoch (int): Current epoch number for W&B logging
        
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
            
            # Get tissue patches if using them (now implemented with tissue patch extraction!)
            tissue_patches = None
            if self.use_tissue_patches and 'tissue_patches' in batch:
                tissue_patches = batch['tissue_patches'].to(self.device)
                logger.debug(f"🧬 Using tissue patches: {tissue_patches.shape}")
                logger.debug(f"🧬 Tissue patch format: (batch_size={tissue_patches.shape[0]}, "
                           f"patches_per_measurement={tissue_patches.shape[1]}, "
                           f"patch_data={tissue_patches.shape[2]}) = 16³×2 channels flattened")
            else:
                logger.debug("🧬 No tissue patches used (baseline mode)")
                
            # Log data flow for tissue patch debugging
            if tissue_patches is not None:
                logger.debug(f"🔍 Tissue patch stats: min={tissue_patches.min():.4f}, "
                           f"max={tissue_patches.max():.4f}, mean={tissue_patches.mean():.4f}")
            
            
            # Forward pass through hybrid model with mixed precision
            logger.debug("⚡ Starting Stage 2 forward pass (NIR → features → reconstruction)...")
            self.optimizer.zero_grad()
            
            with autocast():
                # The hybrid model handles: NIR measurements (batch, 256_subsampled, 8) → transformer → CNN decoder → reconstruction
                # Note: 256 measurements are subsampled from 1000 generated measurements for data augmentation
                outputs = self.model(nir_measurements, tissue_patches)
                logger.debug(f"📤 Stage 2 model output shape: {outputs['reconstructed'].shape}")
                
                # Compute loss
                logger.debug("📏 Computing Stage 2 RMSE loss...")
                loss = self.criterion(outputs['reconstructed'], targets)
                logger.debug(f"💰 Stage 2 batch loss: {loss.item():.6f}")
            
            # Backward pass with mixed precision scaling
            logger.debug("🔙 Starting Stage 2 backward pass (only transformer gradients)...")
            try:
                self.scaler.scale(loss).backward()
                
                # Apply gradient clipping before optimizer step
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
                
                # Monitor gradient norm for training health
                if grad_norm > GRADIENT_MONITOR_THRESHOLD:
                    logger.warning(f"⚠️ High gradient norm detected: {grad_norm:.4f} > {GRADIENT_MONITOR_THRESHOLD}")
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                logger.debug("✅ Stage 2 mixed precision optimizer step completed")
                
                # Linear Warmup + Cosine Decay updates per-batch (essential for smooth scheduling)
                self.scheduler.step()
                
                # Log learning rate every 5 batches to avoid W&B buffer warnings
                if batch_idx % LOG_LR_EVERY_N_BATCHES == 0:
                    self._log_learning_rate_to_wandb(epoch, batch_idx, len(data_loader))
            except RuntimeError as e:
                logger.error(f"🚨 Gradient error: {e}")
                raise e
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate enhanced metrics for this batch
            with torch.no_grad():
                batch_metrics = calculate_batch_metrics(
                    self.metrics, outputs, targets, "stage2"
                )
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
            
            # Show batch progress with standardized metrics format (including LR for LinearWarmupCosineDecay monitoring)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"🏋️ TRAIN | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                       f"RMSE: {loss.item():.4f} | SSIM: {batch_metrics.get('ssim', 0):.4f} | "
                       f"PSNR: {batch_metrics.get('psnr', 0):.1f}dB | LR: {current_lr:.2e}")
            
            # Log gradient norm at debug level for monitoring training health (match Stage 1)
            logger.debug(f"🔧 Batch {batch_idx + 1} | Gradient Norm: {grad_norm:.3f}")
            
            # Additional detailed logging at DEBUG level
            if batch_idx % 10 == 0:  # Log every 10 batches for stability monitoring
                logger.debug(f"🔍 Detailed: Batch {batch_idx}: Loss = {loss.item():.6f}, "
                           f"Running Avg = {total_loss/num_batches:.6f}")
        
        avg_loss = total_loss / num_batches
        
        # Average metrics across epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        logger.debug(f"✅ Stage 2 training epoch completed. Average loss: {avg_loss:.6f}")
        logger.info(f"📊 TRAIN SUMMARY | RMSE: {avg_loss:.4f} | SSIM: {epoch_metrics['ssim']:.4f} | "
                   f"PSNR: {epoch_metrics['psnr']:.1f}dB | Abs: {epoch_metrics['rmse_absorption']:.4f} | "
                   f"Scat: {epoch_metrics['rmse_scattering']:.4f}")
        
        return avg_loss, epoch_metrics
    
    def _log_reconstruction_images(self, predictions, targets, nir_measurements, epoch, step=None, phantom_ids=None):
        """Log 3D reconstruction slices to W&B for visualization (using shared function)."""
        if not self.use_wandb:
            return
            
        # Use shared visualization function to avoid code duplication
        from code.utils.visualization import log_reconstruction_images_to_wandb
        log_reconstruction_images_to_wandb(predictions, targets, epoch, "Reconstructions", step, phantom_ids=phantom_ids)
    
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
                    loss = self.criterion(outputs['reconstructed'], targets)
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
                
                # Show validation batch progress with standardized format (match Stage 1)
                logger.info(f"🔍 VALID | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                           f"RMSE: {loss.item():.4f} | SSIM: {batch_metrics.get('ssim', 0):.4f} | "
                           f"PSNR: {batch_metrics.get('psnr', 0):.1f}dB")
        
        avg_loss = total_loss / num_batches
        
        # Average metrics across epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        logger.debug(f"✅ Stage 2 validation completed. Average loss: {avg_loss:.6f}")
        logger.info(f"📊 VALID SUMMARY | RMSE: {avg_loss:.4f} | SSIM: {epoch_metrics['ssim']:.4f} | "
                   f"PSNR: {epoch_metrics['psnr']:.1f}dB | Abs: {epoch_metrics['rmse_absorption']:.4f} | "
                   f"Scat: {epoch_metrics['rmse_scattering']:.4f}")
        
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
            >>> trainer.train(data_loaders, epochs=EPOCHS_STAGE2)
        """
        # Initialize AdamW optimizer and Linear Warmup + Cosine Decay scheduler
        if self.optimizer is None:
            steps_per_epoch = len(data_loaders['train'])
            self._create_optimizer_and_scheduler(epochs, steps_per_epoch)
        
        # Initialize W&B logging with proper parameters
        if self.use_wandb and not self._wandb_initialized:
            steps_per_epoch = len(data_loaders['train'])
            self._init_wandb(epochs, steps_per_epoch)
            self._wandb_initialized = True
        
        mode = "enhanced" if self.use_tissue_patches else "baseline"
        logger.info(f"🏋️ Starting Transformer training ({mode}) for {epochs} epochs")
        logger.debug(f"📊 Stage 2 configuration: device={self.device}, lr={self.learning_rate}, tissue_patches={self.use_tissue_patches}")
        logger.debug(f"📈 Stage 2 data loaders: train_batches={len(data_loaders['train'])}, val_batches={len(data_loaders['val'])}")
        logger.info(f"📅 AdamW + Linear Warmup + Cosine Decay | Steps per epoch: {len(data_loaders['train'])}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"📅 Starting Transformer Epoch {epoch + 1}/{epochs}")
            
            # Train: Update transformer parameters (CNN decoder frozen)
            logger.debug(f"🏋️  Beginning transformer training phase for epoch {epoch+1}")
            train_loss, train_metrics = self.train_epoch(data_loaders['train'], epoch)
            logger.info(f"🏋️  TRAIN COMPLETE | Avg RMSE: {train_loss:.4f}")
            
            # Validate: Evaluate hybrid model on unseen data (no parameter updates)
            logger.debug(f"🔍 Beginning transformer validation phase for epoch {epoch+1}")
            val_loss, val_metrics = self.validate(data_loaders['val'])
            logger.info(f"🔍 VALID COMPLETE | Avg RMSE: {val_loss:.4f}")
            
            # Log enhanced metrics to W&B
            if self.use_wandb:
                mode = "Enhanced" if self.use_tissue_patches else "Baseline"
                # Use consistent step value throughout this epoch (epoch + 1)
                # Use high epoch-based step value to avoid conflict with batch-level LR logging
                # Batch LR logging goes up to ~3100, so use 10000+ for epochs
                epoch_step = (epoch + 1) * 10000
                
                # Log comprehensive metrics in organized format
                wandb.log({
                    # === PRIMARY METRICS (most important) ===
                    "Metrics/RMSE_Overall_Train": train_loss,
                    "Metrics/RMSE_Overall_Valid": val_loss,
                    "Metrics/SSIM_Train": train_metrics['ssim'],
                    "Metrics/SSIM_Valid": val_metrics['ssim'],
                    "Metrics/PSNR_Train": train_metrics['psnr'],
                    "Metrics/PSNR_Valid": val_metrics['psnr'],
                    
                    # === TRANSFORMER SPECIFIC METRICS ===
                    "Transformer/Feature_Enhancement_Train": train_metrics['feature_enhancement_ratio'],
                    "Transformer/Feature_Enhancement_Valid": val_metrics['feature_enhancement_ratio'],
                    "Transformer/Attention_Entropy_Train": train_metrics['attention_entropy'],
                    "Transformer/Attention_Entropy_Valid": val_metrics['attention_entropy'],
                    
                    # === DETAILED RMSE BREAKDOWN ===
                    "RMSE_Details/Absorption_Train": train_metrics['rmse_absorption'],
                    "RMSE_Details/Absorption_Valid": val_metrics['rmse_absorption'],
                    "RMSE_Details/Scattering_Train": train_metrics['rmse_scattering'],
                    "RMSE_Details/Scattering_Valid": val_metrics['rmse_scattering'],
                    
                    # === TRAINING SYSTEM ===
                    "System/Epoch": epoch + 1,
                    "System/Mode": mode,
                    
                    # === ANALYSIS METRICS ===
                    "Analysis/Train_Valid_RMSE_Ratio": train_loss / val_loss if val_loss > 0 else 0,
                    "Analysis/SSIM_Improvement": val_metrics['ssim'] - train_metrics['ssim'],
                    "Analysis/PSNR_Improvement": val_metrics['psnr'] - train_metrics['psnr'],
                    "epoch": epoch + 1
                })
                
                # Log reconstruction images periodically (and always on first/last epoch)
                should_log_images = (epoch % LOG_IMAGES_EVERY == 0) or (epoch == 0) or (epoch == epochs - 1)
                if should_log_images:
                    try:
                        sample_batch = next(iter(data_loaders['val']))
                        # Stage 2 uses NIR measurements as input and ground truth as target
                        measurements = sample_batch['nir_measurements'].to(self.device)  # Fixed key
                        targets = sample_batch['ground_truth'].to(self.device)          # Fixed key
                        phantom_ids = sample_batch['phantom_id'].cpu().numpy()          # Extract phantom IDs
                        tissue_patches = sample_batch.get('tissue_patches', None)
                        if tissue_patches is not None:
                            tissue_patches = tissue_patches.to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(measurements, tissue_patches)
                        self._log_reconstruction_images(outputs['reconstructed'], targets, measurements, epoch, phantom_ids=phantom_ids)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to log Stage 2 images at epoch {epoch + 1}: {e}")
            
            # Print epoch summary every epoch (always log progress)
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"🚀 EPOCH {epoch+1:3d}/{epochs} SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"📈 Train RMSE: {train_loss:.4f} | Valid RMSE: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"📊 Train SSIM: {train_metrics['ssim']:.4f} | Valid SSIM: {val_metrics['ssim']:.4f}")
            logger.info(f"📊 Train PSNR: {train_metrics['psnr']:.1f}dB | Valid PSNR: {val_metrics['psnr']:.1f}dB")
            logger.info(f"🔮 Feature Enhancement: {val_metrics['feature_enhancement_ratio']:.4f} | Attention: {val_metrics['attention_entropy']:.4f} | Mode: {mode}")
            logger.info(f"{'='*80}")
            
            # Log GPU stats every 5 epochs
            if epoch % 5 == 0 and torch.cuda.is_available():
                log_gpu_stats()
            
            # Learning rate scheduling (Linear Warmup + Cosine Decay updates per-batch)
            # No epoch-level action needed - scheduler updates per-batch automatically
            
            # Save best model
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                checkpoint_filename = CHECKPOINT_STAGE2_ENHANCED if self.use_tissue_patches else CHECKPOINT_STAGE2_BASELINE
                checkpoint_path = f"{CHECKPOINT_BASE_DIR}/{checkpoint_filename}"
                logger.info(f"🎉 NEW BEST MODEL | Improvement: {improvement:.4f} | Best RMSE: {best_val_loss:.4f}")
                logger.debug(f"💾 Stage 2 checkpoint path: {checkpoint_path}")
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                logger.debug(f"💾 New best Stage 2 model saved at epoch {epoch}")
            else:
                logger.debug(f"📊 Stage 2 no improvement. Current: {val_loss:.6f}, Best: {best_val_loss:.6f}")
        
        mode = "Enhanced" if self.use_tissue_patches else "Baseline"
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"✅ TRANSFORMER TRAINING COMPLETED ({mode})")
        logger.info(f"🏆 Best RMSE Loss: {best_val_loss:.4f}")
        logger.info(f"📊 Total Epochs: {epochs}")
        logger.info(f"{'='*80}")
        
        logger.debug(f"🏁 Stage 2 training summary: Total epochs: {epochs}, Final best loss: {best_val_loss:.6f}")
        
        # Finish W&B run
        if self.use_wandb:
            wandb.log({f"System/final_best_val_loss": best_val_loss, f"System/final_mode": mode, "epoch": epochs})
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
