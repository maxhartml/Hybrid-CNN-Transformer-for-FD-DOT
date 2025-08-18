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
import torch.nn as nn
import numpy as np
import wandb
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training for A100 optimization

# Project imports
from code.models.hybrid_model import HybridCNNTransformer
from code.utils.logging_config import get_training_logger
from code.utils.metrics import NIRDOTMetrics, create_metrics_for_stage, calculate_batch_metrics, RMSELoss, dice_per_channel, contrast_ratio_per_channel
from code.utils.standardizers import Stage2StandardizerCollection
from code.data_processing.data_loader import create_phantom_dataloaders  # For standardizer fitting
from .training_config import *  # Import all training config
from .training_utils import get_or_create_run_id, get_checkpoint_path, save_checkpoint, find_best_checkpoint

# Initialize module logger
logger = get_training_logger(__name__)

# =============================================================================
# PERFORMANCE OPTIMIZATIONS
# =============================================================================

# Enable TensorFloat32 for better A100 performance (suppresses the warning)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    logger.debug("✅ Enabled TensorFloat32 for optimized A100 matrix multiplication")

# =============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# =============================================================================

class EMAModel:
    """
    Exponential Moving Average of model parameters for better generalization.
    
    Maintains a moving average of model weights that often performs better
    than the final training weights, especially for transformer models.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.997):
        """
        Initialize EMA model.
        
        Args:
            model: The model to track
            decay: EMA decay rate (higher = slower update)
        """
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update EMA weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA weights to model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

# =============================================================================
# STAGE-SPECIFIC CONFIGURATION
# =============================================================================

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
    
    def __init__(self, stage1_checkpoint_path=None, use_tissue_patches=USE_TISSUE_PATCHES_STAGE2, 
                 learning_rate=STAGE2_BASE_LR, device=CPU_DEVICE, use_wandb=True,
                 early_stopping_patience=EARLY_STOPPING_PATIENCE, checkpoint_dir=None):
        """
        Initialize the Stage 2 trainer with pre-trained CNN components.
        
        Args:
            stage1_checkpoint_path (str, optional): Path to Stage 1 checkpoint file containing
                                        pre-trained CNN autoencoder weights. If None, 
                                        automatically selects the best Stage 1 checkpoint
            use_tissue_patches (bool): Whether to enable tissue patch integration
                                     for enhanced spatial modeling. Default from constants
            learning_rate (float): Learning rate for transformer optimization.
                                 Typically lower than Stage 1. Default from constants
            device (str): Training device ('cpu' or 'cuda'). Default from constants
            use_wandb (bool): Whether to use Weights & Biases logging. Default: True
            early_stopping_patience (int): Early stopping patience in epochs. Default: 25
            checkpoint_dir (str, optional): Directory to search for Stage 1 checkpoints.
                                          Default: uses CHECKPOINT_BASE_DIR from config
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.use_tissue_patches = use_tissue_patches
        self.use_wandb = use_wandb
        self.early_stopping_patience = early_stopping_patience
        
        # Automatic checkpoint selection if path not provided
        if stage1_checkpoint_path is None:
            checkpoint_search_dir = checkpoint_dir or CHECKPOINT_BASE_DIR
            logger.info("🔍 No Stage 1 checkpoint path provided - searching for best checkpoint...")
            stage1_checkpoint_path = find_best_checkpoint("stage1", checkpoint_search_dir)
            
            if stage1_checkpoint_path is None:
                raise FileNotFoundError(
                    f"No valid Stage 1 checkpoints found in {checkpoint_search_dir}. "
                    f"Please run Stage 1 training first or provide a specific checkpoint path."
                )
        
        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Initialize enhanced metrics for Stage 2 (includes feature analysis)
        self.metrics = create_metrics_for_stage("stage2")
        
        # Initialize Stage 2 standardizer collection
        self.standardizers = Stage2StandardizerCollection(device=self.device)
        
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
        
        # Freeze CNN decoder (ECBO 2025 approach)
        self.freeze_cnn_decoder()
        
        # Log stage-specific parameter usage after freezing
        self._log_stage2_parameter_breakdown()
        
        # Loss function - Standard RMSE loss
        self.criterion = RMSELoss()
        logger.info("📊 Stage 2 using standard RMSE loss")
        
        # NOTE: Optimizer and scheduler are created in this stage using research-validated
        # AdamW + Linear Warmup + Cosine Decay for optimal transformer fine-tuning.
        self.optimizer = None
        self.scheduler = None
        
        # Mixed precision training with conservative scaler settings (prevents crashes)
        if self.device.type == 'cuda':
            self.scaler = GradScaler(
                init_scale=GRADSCALER_INIT_SCALE,
                growth_factor=GRADSCALER_GROWTH_FACTOR, 
                backoff_factor=GRADSCALER_BACKOFF_FACTOR,
                growth_interval=GRADSCALER_GROWTH_INTERVAL
            )
        else:
            self.scaler = None
        
        mode = "enhanced" if use_tissue_patches else "baseline"
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"🚀 TRANSFORMER TRAINING INITIALIZATION ({mode})")
        logger.info(f"{'='*80}")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"📈 Base Learning Rate: {learning_rate}")
        logger.info(f"🔒 L2 Regularization: {WEIGHT_DECAY_TRANSFORMER}")
        logger.info(f"⏰ Early Stopping Patience: {early_stopping_patience}")
        logger.info(f"🧬 Tissue Patches: {use_tissue_patches}")
        if self.scaler:
            # Check AMP dtype for ChatGPT's gradient underflow fix
            bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            amp_dtype = "bfloat16" if bf16_supported else "float16"
            logger.info(f"🚀 Mixed Precision: Enabled ({amp_dtype}) - ChatGPT underflow fix")
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
        
        # =============================================================================
        # LATENT-ONLY TRAINING SETUP
        # =============================================================================
        
        if TRAIN_STAGE2_LATENT_ONLY:
            logger.info("🎯 LATENT-ONLY TRAINING MODE ENABLED")
            logger.info(f"   ├─ Latent dimension: {LATENT_DIM}")
            logger.info(f"   ├─ E2E validation every: {VAL_E2E_EVERY_K_EPOCHS} epochs")
            logger.info(f"   └─ Training on latent RMSE only (no decoder)")
            
            # Import latent-only training components
            from .teacher_stage1 import load_teacher_stage1
            from .latent_stats import LatentStats, compute_latent_rmse
            
            # Initialize teacher model for latent targets
            self.teacher = load_teacher_stage1(checkpoint_path=stage1_checkpoint_path, device=self.device)
            logger.info("✅ Stage 1 teacher model loaded for latent targets")
            
            # Initialize latent affine aligner (learned scale + bias transformation)
            self.latent_align = nn.Linear(LATENT_DIM, LATENT_DIM, bias=True)
            with torch.no_grad():
                self.latent_align.weight.copy_(torch.eye(LATENT_DIM))
                self.latent_align.bias.zero_()
            self.latent_align.to(self.device)
            logger.info(f"🔧 Latent affine aligner initialized (identity transform)")
            
            # Initialize latent statistics tracker
            self.latent_stats = LatentStats()
            
            # Override loss function to latent RMSE
            self.criterion = compute_latent_rmse
            logger.info("📊 Stage 2 using latent RMSE loss (teacher-student)")
        else:
            logger.info("🔄 STANDARD TRAINING MODE (end-to-end reconstruction)")
            self.teacher = None
            self.latent_stats = None
            self.latent_align = None
        
        # Initialize EMA if enabled - will be created after optimizer/scheduler in train()
        self.ema = None
            
        # Initialize checkpoint system for Stage 2
        self.run_id = get_or_create_run_id("stage2")
        self.checkpoint_path = get_checkpoint_path("stage2", self.run_id)
        logger.info(f"📁 Stage 2 Checkpoint: {self.checkpoint_path}")
    
    def _create_lightweight_dataloader_for_standardizer_fitting(self):
        """
        Create a lightweight dataloader specifically for standardizer fitting.
        
        This dataloader skips tissue patch extraction to dramatically reduce memory
        usage and processing time during standardizer fitting. Standardizer fitting
        only needs NIR measurements to compute mean/std statistics.
        
        Returns:
            DataLoader: Training dataloader with tissue patches disabled
        """
        logger.info("🔧 Creating lightweight dataloader for standardizer fitting (no tissue patches)")
        
        # Create dataloaders with tissue patches explicitly disabled
        dataloaders = create_phantom_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            use_tissue_patches=False,  # Always False for standardizer fitting
            stage='stage2'
        )
        
        return dataloaders['train']
    
    def _calculate_attention_entropy(self) -> Dict[str, float]:
        """
        Calculate attention entropy for each transformer layer to track learning progress.
        
        Key diagnostic from ChatGPT: Attention should move from uniform (~ln(256)=5.545) 
        to focused patterns within first few epochs. Staying at 5.545 indicates frozen attention.
        
        Returns:
            Dict[str, float]: Attention entropy per layer
        """
        entropies = {}
        
        # Access transformer layers through the model
        if hasattr(self.model, 'transformer_encoder') and hasattr(self.model.transformer_encoder, 'layers'):
            for i, layer in enumerate(self.model.transformer_encoder.layers):
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'last_attention_weights'):
                    # Get the last computed attention weights [batch, heads, seq, seq]
                    attn_weights = layer.attention.last_attention_weights
                    if attn_weights is not None:
                        # Calculate entropy: -sum(p * log(p)) where p is attention probability
                        # Average over batch and heads, compute entropy over sequence dimension
                        attn_probs = attn_weights.mean(dim=(0, 1))  # [seq, seq]
                        # Focus on diagonal-ish attention patterns
                        attn_probs = attn_probs + 1e-12  # Numerical stability
                        entropy = -(attn_probs * torch.log(attn_probs)).sum().item()
                        entropies[f'layer_{i}_entropy'] = entropy
        
        return entropies
    
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
        wandb.define_metric("ema/*", step_metric="epoch")  # EMA metrics vs epoch
        wandb.define_metric("Metrics/*", step_metric="epoch")
        wandb.define_metric("RMSE_Details/*", step_metric="epoch")
        wandb.define_metric("System/*", step_metric="epoch")
        wandb.define_metric("Analysis/*", step_metric="epoch")
        wandb.define_metric("Reconstructions/*", step_metric="epoch")
        
        logger.info(f"🔬 W&B experiment initialized: {experiment_name}")
    
    def _compute_ema_decay(self, global_step: int) -> float:
        """
        Compute progressive EMA decay using cosine ramp from EMA_DECAY_START to EMA_DECAY_END.
        
        Args:
            global_step: Current training step
            
        Returns:
            Current EMA decay value
        """
        if not hasattr(self, 'total_training_steps') or self.total_training_steps <= 1:
            return EMA_DECAY_START
        
        # Calculate progress through training (0.0 to 1.0)
        progress = min(global_step / self.total_training_steps, 1.0)
        
        # Cosine ramp from START to END
        cur_decay = EMA_DECAY_START + (EMA_DECAY_END - EMA_DECAY_START) * 0.5 * (1 - math.cos(math.pi * progress))
        
        return cur_decay
    
    def _log_stage2_parameter_breakdown(self):
        """
        Log detailed parameter breakdown for Stage 2 after freezing.
        """
        # Count parameters by component
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        
        # Component-wise breakdown
        cnn_total = sum(p.numel() for p in self.model.cnn_autoencoder.parameters())
        cnn_decoder = sum(p.numel() for p in self.model.cnn_autoencoder.decoder.parameters())
        cnn_encoder = sum(p.numel() for p in self.model.cnn_autoencoder.encoder.parameters())
        
        embedding_total = sum(p.numel() for p in self.model.spatially_aware_encoder.parameters())
        transformer_total = sum(p.numel() for p in self.model.transformer_encoder.parameters())
        pooling_total = sum(p.numel() for p in self.model.global_pooling_encoder.parameters())
        calibrator_total = sum(p.numel() for p in self.model.range_calibrator.parameters())
        
        # Count trainable in each component
        cnn_decoder_trainable = sum(p.numel() for p in self.model.cnn_autoencoder.decoder.parameters() if p.requires_grad)
        embedding_trainable = sum(p.numel() for p in self.model.spatially_aware_encoder.parameters() if p.requires_grad)
        transformer_trainable = sum(p.numel() for p in self.model.transformer_encoder.parameters() if p.requires_grad)
        pooling_trainable = sum(p.numel() for p in self.model.global_pooling_encoder.parameters() if p.requires_grad)
        calibrator_trainable = sum(p.numel() for p in self.model.range_calibrator.parameters() if p.requires_grad)
        
        logger.info("📊 STAGE 2 PARAMETER BREAKDOWN (AFTER FREEZING):")
        logger.info("   ┌─────────────────────────────────────────────────────────────┐")
        logger.info(f"   │ CNN Decoder:           {cnn_decoder:>8,} params ({cnn_decoder_trainable:>8,} trainable) │")
        logger.info(f"   │ Spatially-Aware Embed: {embedding_total:>8,} params ({embedding_trainable:>8,} trainable) │")
        logger.info(f"   │ Transformer Encoder:   {transformer_total:>8,} params ({transformer_trainable:>8,} trainable) │")
        logger.info(f"   │ Global Pooling (Attn): {pooling_total:>8,} params ({pooling_trainable:>8,} trainable) │")
        logger.info(f"   │ Range Calibrator:      {calibrator_total:>8,} params ({calibrator_trainable:>8,} trainable) │")
        logger.info("   ├─────────────────────────────────────────────────────────────┤")
        logger.info(f"   │ TOTAL MODEL:           {total_params:>8,} params                 │")
        logger.info(f"   │ TRAINABLE:             {trainable_params:>8,} params                 │")
        logger.info(f"   │ FROZEN:                {frozen_params:>8,} params                 │")
        logger.info("   └─────────────────────────────────────────────────────────────┘")
        
        # Enhanced features logging
        logger.info("🚀 ENHANCED TRAINING FEATURES:")
        logger.info(f"   ├─ Attention Pooling: ✅ Active (learnable)")
        logger.info(f"   ├─ Range Calibrator: ✅ Active with L2 regularization")
        logger.info(f"   ├─ Decoder Unfreezing: {'✅ Enabled' if UNFREEZE_LAST_DECODER_BLOCK else '❌ Disabled'}")
        logger.info(f"   ├─ EMA Training: {'✅ Active (decay=' + str(EMA_DECAY) + ')' if USE_EMA else '❌ Disabled'}")
        logger.info(f"   ├─ Attention Entropy Reg: {'✅ Active (λ=' + str(ATTENTION_ENTROPY_LAMBDA) + ')' if ATTENTION_ENTROPY_LAMBDA > 0 else '❌ Disabled'}")
        logger.info(f"   └─ Training Schedule: Cosine decay with floor ({STAGE2_MIN_LR})")
        
        logger.info(f"❄️  DISCARDED: CNN Encoder ({cnn_encoder:,} params - not used in Stage 2)")
        logger.info("🎯 Stage 2 trains transformer pipeline with frozen CNN decoder")
    
    def load_stage1_checkpoint(self, checkpoint_path):
        """
        Load pre-trained Stage 1 checkpoint into the model and initialize standardizers.
        
        This method loads:
        1. CNN autoencoder weights from Stage 1 training
        2. Ground truth standardizer from Stage 1 for inverse transformation
        3. Prepares Stage 2 standardizer collection for training data fitting
        
        Args:
            checkpoint_path (str): Path to the Stage 1 checkpoint file
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint format is invalid
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dict with compatibility for missing tissue encoder parameters
        model_state = checkpoint['model_state_dict']
        
        # Handle compiled model state dict (remove _orig_mod. prefix)
        if any(key.startswith('_orig_mod.') for key in model_state.keys()):
            logger.info("🔧 Removing compilation prefixes from Stage 1 checkpoint...")
            model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
        
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
        
        # Load ground truth standardizer from Stage 1 checkpoint
        if 'standardizer' in checkpoint:
            self.standardizers.ground_truth_standardizer.load_state_dict(checkpoint['standardizer'])
            logger.info("✅ Loaded ground truth standardizer from Stage 1 checkpoint")
        else:
            logger.warning("⚠️ No standardizer found in Stage 1 checkpoint - will need manual initialization")
        
        # Store checkpoint path for later use in standardizer fitting
        self.stage1_checkpoint_path = checkpoint_path
        
        logger.info(f"📂 Loaded Stage 1 checkpoint: {checkpoint_path}")
        
        epoch_info = checkpoint.get('epoch', 'N/A')
        val_loss_info = checkpoint.get('val_loss', None)
        if val_loss_info is not None:
            logger.info(f"📊 Checkpoint epoch: {epoch_info}, val_loss: {val_loss_info:.6f}")
        else:
            logger.info(f"📊 Checkpoint epoch: {epoch_info}, val_loss: N/A")
    
    def _create_parameter_groups(self):
        """
        Create parameter groups for transformer training with differential weight decay.
        
        This method implements the critical transformer training approach where
        LayerNorm weights, biases, and embeddings receive NO weight decay to prevent
        gradient flow issues and "frozen attention" problems.
        
        Additionally supports decoder fine-tuning with reduced learning rate.
        
        Based on BERT, GPT, and ViT training procedures + ChatGPT recommendations.
        
        Returns:
            list: Parameter groups for AdamW optimizer
        """
        decay_params = []
        no_decay_params = []
        decoder_params = []  # For unfrozen decoder parameters with reduced LR
        decoder_no_decay_params = []
        
        # Unfreeze last decoder block if enabled
        if UNFREEZE_LAST_DECODER_BLOCK:
            self.model.unfreeze_last_decoder_block()
        
        # Add latent aligner parameters to no_decay group (if using latent-only training)
        if TRAIN_STAGE2_LATENT_ONLY and hasattr(self, 'latent_align'):
            no_decay_params.append(self.latent_align.weight)
            no_decay_params.append(self.latent_align.bias)
            logger.debug(f"🔧 Added latent aligner to no_decay group")
        
        # Get unfrozen decoder parameters for separate group
        unfrozen_decoder_params = []
        if UNFREEZE_LAST_DECODER_BLOCK:
            unfrozen_decoder_params = self.model.get_unfrozen_decoder_parameters()
            unfrozen_param_names = {id(p) for p in unfrozen_decoder_params}
            logger.info(f"🔓 Found {len(unfrozen_decoder_params)} unfrozen decoder parameters")
        else:
            unfrozen_param_names = set()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Check if this parameter is from unfrozen decoder
                is_decoder_param = id(param) in unfrozen_param_names
                
                # NO weight decay for: biases, norms, and specific embedding parameters (critical for gradient flow)
                if (name.endswith(".bias") or 
                    "norm" in name.lower() or 
                    "ln" in name.lower() or
                    "layer_norm" in name.lower() or 
                    name.endswith("embedding.weight") or
                    "pos_embed" in name.lower() or
                    "token_type_embedding" in name or
                    "range_calibrator" in name):  # Range calibrator gets no weight decay
                    if is_decoder_param:
                        decoder_no_decay_params.append(param)
                        logger.debug(f"🔓🚫 Decoder no decay: {name}")
                    else:
                        no_decay_params.append(param)
                        logger.debug(f"🚫 No decay: {name}")
                else:
                    if is_decoder_param:
                        decoder_params.append(param)
                        logger.debug(f"🔓✅ Decoder with decay: {name}")
                    else:
                        decay_params.append(param)
                        logger.debug(f"✅ With decay: {name}")
        
        param_groups = [
            {'params': decay_params, 'weight_decay': WEIGHT_DECAY_TRANSFORMER},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Add decoder parameter groups with reduced LR if any unfrozen decoder params exist
        if decoder_params or decoder_no_decay_params:
            reduced_lr = STAGE2_BASE_LR * DECODER_FINETUNING_LR_SCALE
            if decoder_params:
                param_groups.append({
                    'params': decoder_params, 
                    'weight_decay': WEIGHT_DECAY_TRANSFORMER, 
                    'lr': reduced_lr
                })
            if decoder_no_decay_params:
                param_groups.append({
                    'params': decoder_no_decay_params, 
                    'weight_decay': 0.0, 
                    'lr': reduced_lr
                })
            logger.info(f"🔓 Added {len(decoder_params + decoder_no_decay_params)} decoder params with {DECODER_FINETUNING_LR_SCALE}x LR")
        
        logger.info(f"[AdamW Groups] decay: {len(decay_params)} params, no_decay: {len(no_decay_params)} params")
        logger.info(f"📊 Parameter Groups (CRITICAL for transformer training):")
        logger.info(f"   ├─ With weight decay: {len(decay_params)} params")
        logger.info(f"   ├─ No weight decay: {len(no_decay_params)} params (norms/biases/embeddings)")
        if decoder_params or decoder_no_decay_params:
            logger.info(f"   ├─ Decoder with decay: {len(decoder_params)} params")
            logger.info(f"   └─ Decoder no decay: {len(decoder_no_decay_params)} params")
        
        return param_groups
    
    def _create_optimizer_and_scheduler(self, steps_per_epoch: int, total_epochs: int):
        """
        Create AdamW optimizer and single scheduler for Stage 2 with linear warmup + cosine decay.
        
        Features:
        - Linear warmup from start_factor*base_lr to base_lr
        - Cosine decay from base_lr to min_lr floor
        - Single unified scheduler (no dual scheduler complexity)
        
        Args:
            steps_per_epoch (int): Batches per epoch
            total_epochs (int): Total training epochs
        """
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        warmup_steps = int(max(1, STAGE2_WARMUP_PCT * self.total_steps))
        self.warmup_steps = warmup_steps

        base_lr = STAGE2_BASE_LR
        min_lr  = STAGE2_MIN_LR
        start_factor = getattr(globals(), "SCHEDULER_START_FACTOR", 0.01)

        # Create parameter groups for differential weight decay
        param_groups = self._create_parameter_groups()
        
        # Create AdamW optimizer with transformer-optimized parameters
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,                         # Base LR will be managed by scheduler
            betas=ADAMW_BETAS_STAGE2,          # Transformer-standard betas (0.9, 0.98)
            eps=ADAMW_EPS_STAGE2               # Numerical stability
        )

        # Make sure optimizer param_groups all share base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = base_lr

        def lr_lambda(global_step: int):
            # Linear warmup from start_factor*base_lr -> base_lr
            if global_step < warmup_steps:
                frac = global_step / max(1, warmup_steps)
                return start_factor + (1.0 - start_factor) * frac

            # Cosine decay from base_lr -> min_lr
            decay_steps = max(1, self.total_steps - warmup_steps)
            progress = (global_step - warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            # cosine multiplier in [0,1]
            cos_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            # map to [min_lr/base_lr, 1]
            floor = min_lr / base_lr
            return floor + (1.0 - floor) * cos_mult

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=-1)
        
        logger.info(f"🚀 STAGE 2 LINEAR WARMUP + COSINE DECAY SCHEDULER:")
        logger.info(f"   ├─ Warmup Steps: {warmup_steps:,} ({STAGE2_WARMUP_PCT*100:.1f}%)")
        logger.info(f"   ├─ Base LR: {base_lr:.2e}")
        logger.info(f"   ├─ Min LR: {min_lr:.2e}")
        logger.info(f"   └─ Start Factor: {start_factor} (warmup begins at {start_factor*100:.1f}% of base)")
        
        logger.info(f"🚀 STAGE 2 ADAMW OPTIMIZER:")
        logger.info(f"   ├─ Base LR: {base_lr:.0e}")
        logger.info(f"   ├─ Weight Decay (Transformer): {WEIGHT_DECAY_TRANSFORMER}")
        logger.info(f"   └─ Betas: {ADAMW_BETAS_STAGE2}")
        
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
        Freeze entire CNN autoencoder parameters to preserve Stage 1 learned features.
        
        This method implements the core strategy of the two-stage approach by
        freezing all CNN autoencoder parameters, ensuring that the robust
        feature representations learned in Stage 1 are preserved while only
        the transformer components are optimized.
        
        The freezing strategy:
        - Disables gradient computation for all CNN autoencoder parameters
        - Reduces the parameter space for efficient transformer optimization
        - Preserves stable feature extraction capabilities
        """
        logger.debug("🔒 Starting CNN autoencoder freezing process...")
        
        # Freeze the entire CNN autoencoder
        frozen_params = 0
        for name, param in self.model.cnn_autoencoder.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            logger.debug(f"🔒 Frozen: {name} ({param.numel():,} params)")
        
        # Count trainable parameters after freezing
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"🔒 Entire CNN autoencoder frozen. Frozen: {frozen_params:,}, "
                   f"Trainable: {trainable_params:,}/{total_params:,} "
                   f"({100 * trainable_params / total_params:.1f}%)")
        
        # Verify we have trainable parameters
        if trainable_params == 0:
            logger.error("🚨 ERROR: No trainable parameters found! All parameters are frozen!")
            raise RuntimeError("No trainable parameters - all model parameters are frozen")
        
        logger.info(f"✅ Parameter freezing completed successfully")
    
    def train_epoch(self, data_loader, epoch=0):
        """
        Execute one complete training epoch for transformer components.
        
        This method performs forward propagation through the hybrid model,
        with tissue patch integration when enabled. Only unfrozen transformer
        parameters are updated during backpropagation.
        
        Logging matches Stage 1 exactly:
        - Per batch: Std_RMSE and LR only
        - Summary: Std_RMSE mean
        
        Args:
            data_loader: DataLoader containing training batches with
                        'measurements', 'volumes', and optionally 'tissue_patches'
            epoch (int): Current epoch number for W&B logging
        
        Returns:
            float: Average standardized RMSE loss across all batches
        """
        if DEBUG_VERBOSE:
            logger.debug("🔄 Starting Stage 2 training epoch...")
        
        # Ensure model is in correct stage mode
        self.model.set_training_stage("stage2")
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Reset latent statistics for this epoch
        if TRAIN_STAGE2_LATENT_ONLY:
            self.latent_stats.reset()
        
        if DEBUG_VERBOSE:
            logger.debug(f"📊 Processing {len(data_loader)} batches in Stage 2 training epoch")
        
        for batch_idx, batch in enumerate(data_loader):
            logger.debug(f"🔍 Processing Stage 2 batch {batch_idx + 1}/{len(data_loader)}")
            
            # Extract raw inputs from batch
            nir_measurements_raw = batch['nir_measurements'].to(self.device)  # Shape: (batch_size, 256_subsampled, 8)
            targets_raw = batch['ground_truth'].to(self.device)               # Shape: (batch_size, 2, 64, 64, 64)
            
            logger.debug(f"📦 Raw NIR measurements shape: {nir_measurements_raw.shape}")
            logger.debug(f"📦 Raw ground truth targets shape: {targets_raw.shape}")
            
            # Apply Stage 2 standardization to inputs
            nir_measurements = self.standardizers.transform_nir_inputs(nir_measurements_raw)
            targets = self.standardizers.ground_truth_standardizer.transform(targets_raw)  # Train on standardized targets
            
            logger.debug(f"📦 Standardized NIR measurements shape: {nir_measurements.shape}")
            logger.debug(f"📦 Standardized ground truth targets shape: {targets.shape}")
            
            # Get and standardize tissue patches if using them
            tissue_patches = None
            if self.use_tissue_patches and 'tissue_patches' in batch:
                tissue_patches_raw = batch['tissue_patches'].to(self.device)
                tissue_patches = self.standardizers.transform_tissue_patches(tissue_patches_raw)
                logger.debug(f"🧬 Using standardized tissue patches: {tissue_patches.shape}")
                logger.debug(f"🧬 Tissue patch format: (batch_size={tissue_patches.shape[0]}, "
                           f"n_measurements={tissue_patches.shape[1]}, "
                           f"patches={tissue_patches.shape[2]}, channels={tissue_patches.shape[3]}, "
                           f"spatial={tissue_patches.shape[4]}×{tissue_patches.shape[5]}×{tissue_patches.shape[6]})")
            else:
                logger.debug("🧬 No tissue patches used (baseline mode)")
                
            # Log data flow for tissue patch debugging
            if tissue_patches is not None:
                logger.debug(f"🔍 Tissue patch stats: min={tissue_patches.min():.4f}, "
                           f"max={tissue_patches.max():.4f}, mean={tissue_patches.mean():.4f}")
            
            
            # Forward pass through hybrid model with mixed precision (bfloat16 if available)
            logger.debug("⚡ Starting Stage 2 forward pass (NIR → features → reconstruction)...")
            self.optimizer.zero_grad()
            
            # Guard AMP autocast cleanly
            use_amp = (self.device.type == "cuda")
            amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
            
            with autocast(enabled=use_amp, dtype=amp_dtype):
                if TRAIN_STAGE2_LATENT_ONLY:
                    # LATENT-ONLY TRAINING MODE
                    # Forward pass only through encoder to get student latent
                    student_latent = self.model.encode(nir_measurements, tissue_patches)
                    
                    # Apply latent affine aligner (learned scale + bias transformation)
                    student_latent_aligned = self.latent_align(student_latent)
                    
                    # Get teacher latent from standardized ground truth (NOT NIR measurements)
                    with torch.no_grad():
                        teacher_latent = self.teacher.encode_from_gt_std(targets)
                    
                    # Compute latent RMSE loss using aligned student latent
                    loss = self.criterion(student_latent_aligned, teacher_latent)
                    
                    # Update latent statistics using aligned student latent
                    batch_latent_stats = self.latent_stats.update(teacher_latent, student_latent_aligned)
                    
                    # Debug: Assert correct shapes and no NaNs
                    assert teacher_latent.shape == student_latent_aligned.shape == (targets.shape[0], LATENT_DIM), \
                        f"Latent shape mismatch: teacher={teacher_latent.shape}, student_aligned={student_latent_aligned.shape}, expected=({targets.shape[0]}, {LATENT_DIM})"
                    assert not torch.isnan(student_latent_aligned).any(), "Student aligned latent contains NaNs"
                    assert not torch.isnan(teacher_latent).any(), "Teacher latent contains NaNs"
                    
                    # Log latent stats at debug level
                    if DEBUG_VERBOSE:
                        logger.debug(f"🎯 Latent RMSE: {batch_latent_stats['latent_rmse']:.4f}, "
                                   f"Cosine Sim: {batch_latent_stats['latent_cosine_sim']:.4f}")
                        logger.debug(f"📊 Teacher latent: mean={teacher_latent.mean():.4f}, std={teacher_latent.std():.4f}, "
                                   f"min={teacher_latent.min():.4f}, max={teacher_latent.max():.4f}, norm={teacher_latent.norm():.4f}")
                        logger.debug(f"📊 Student latent (aligned): mean={student_latent_aligned.mean():.4f}, std={student_latent_aligned.std():.4f}, "
                                   f"min={student_latent_aligned.min():.4f}, max={student_latent_aligned.max():.4f}, norm={student_latent_aligned.norm():.4f}")
                        logger.debug(f"📊 Student latent (raw): mean={student_latent.mean():.4f}, std={student_latent.std():.4f}, "
                                   f"min={student_latent.min():.4f}, max={student_latent.max():.4f}, norm={student_latent.norm():.4f}")
                else:
                    # STANDARD END-TO-END TRAINING MODE
                    outputs = self.model(nir_measurements, tissue_patches)
                    logger.debug(f"📤 Stage 2 model output shape: {outputs['reconstructed'].shape}")
                    
                    # Compute standardized RMSE loss (matches Stage 1)
                    loss = self.criterion(outputs['reconstructed'], targets)
                    
                    # Add attention entropy regularization if available
                    if 'attention_weights' in outputs and ATTENTION_ENTROPY_LAMBDA > 0:
                        attention_weights = outputs['attention_weights']
                        # Compute entropy of attention weights across sequence dimension
                        # attention_weights shape: [batch, num_heads, seq_len, seq_len] or [batch, seq_len, seq_len]
                        if attention_weights.dim() == 4:
                            # Multi-head case: average over heads
                            attn_probs = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
                        else:
                            attn_probs = attention_weights  # [batch, seq_len, seq_len]
                        
                        # Compute entropy for each batch item and position
                        # Focus on entropy of each position's attention distribution
                        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1)  # [batch, seq_len]
                        mean_entropy = entropy.mean()  # Average across batch and positions
                        
                        # Add entropy regularization to loss (non-in-place)
                        loss = loss + ATTENTION_ENTROPY_LAMBDA * mean_entropy
                        
                        # Store for logging
                        mean_entropy_for_logging = float(mean_entropy)
                        
                        if DEBUG_VERBOSE:
                            logger.debug(f"🧠 Attention entropy: {mean_entropy:.4f}, regularization: {ATTENTION_ENTROPY_LAMBDA * mean_entropy:.6f}")
                    
                    # DEBUGGING: Log transformer activity details only at the final batch of epoch
                    if (batch_idx + 1) == len(data_loader):
                        attn_weights = outputs['attention_weights']
                        logger.debug(f"🧠 Transformer attention stats: shape={attn_weights.shape}, "
                                   f"min={attn_weights.min():.4f}, max={attn_weights.max():.4f}, "
                                   f"mean={attn_weights.mean():.4f}")
                    
                    if (batch_idx + 1) == len(data_loader) and 'enhanced_features' in outputs and outputs['enhanced_features'] is not None:
                        features = outputs['enhanced_features']
                        logger.debug(f"✨ Enhanced features stats: shape={features.shape}, "
                                   f"min={features.min():.4f}, max={features.max():.4f}, "
                                   f"mean={features.mean():.4f}, std={features.std():.4f}")
                    
                    # SAFETY: Check for NaN values immediately after forward pass
                    if torch.isnan(outputs['reconstructed']).any():
                        logger.error(f"🚨 NaN detected in model output at batch {batch_idx}")
                        raise ValueError(f"NaN detected in model output - stopping training at batch {batch_idx}")
                
                # Add range calibrator regularization (non-in-place)
                if hasattr(self.model, 'get_calibrator_regularization'):
                    calibrator_reg = self.model.get_calibrator_regularization()
                    loss = loss + calibrator_reg
                
                # SAFETY: Check for NaN loss immediately
                if torch.isnan(loss):
                    logger.error(f"🚨 NaN loss detected at batch {batch_idx}")
                    raise ValueError(f"NaN loss detected - stopping training at batch {batch_idx}")
            
            # Backward pass and optimization (Mixed Precision enabled)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_MAX_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_MAX_NORM)
                self.optimizer.step()
            
            # Step scheduler - single scheduler only
            self.scheduler.step()
            
            # One-time scheduler info log
            if epoch == 0 and batch_idx == 0:
                logger.info(f"📅 LR schedule: warmup {self.warmup_steps} steps "
                           f"({STAGE2_WARMUP_PCT*100:.1f}%), "
                           f"base={STAGE2_BASE_LR:.3e}, floor={STAGE2_MIN_LR:.3e}")
            
            # SAFETY: Check for extreme gradient norms (sign of training instability)
            if grad_norm > GRADIENT_MONITOR_THRESHOLD:
                if DEBUG_VERBOSE:
                    logger.warning(f"⚠️ High gradient norm detected: {grad_norm:.3f} (threshold: {GRADIENT_MONITOR_THRESHOLD})")
            
            # Update EMA weights with progressive decay ramp
            if self.ema is not None:
                global_step = epoch * len(data_loader) + batch_idx + 1
                current_ema_decay = self._compute_ema_decay(global_step)
                self.ema.decay = current_ema_decay  # Update decay dynamically
                self.ema.update()
            
            # Wandb logging (matches Stage 1 pattern exactly)
            if self.use_wandb and wandb.run:
                current_lr = self.optimizer.param_groups[0]['lr']
                global_step = epoch * len(data_loader) + batch_idx + 1
                
                log_data = {
                    "training_step": global_step,
                    "train/lr": current_lr
                }
                
                # Add EMA decay if enabled
                if self.ema is not None:
                    log_data["ema/decay"] = current_ema_decay
                
                if TRAIN_STAGE2_LATENT_ONLY:
                    log_data["train/latent_rmse"] = batch_latent_stats["latent_rmse"]
                else:
                    log_data["train/loss_std"] = float(loss.item())
                
                # Add attention entropy if available
                if 'mean_entropy_for_logging' in locals():
                    log_data["train/attention_entropy"] = mean_entropy_for_logging
                
                # Add latent-specific metrics if in latent-only mode
                if TRAIN_STAGE2_LATENT_ONLY:
                    log_data.update({
                        "train/latent_cosine_sim": batch_latent_stats['latent_cosine_sim'],
                        "train/teacher_magnitude": batch_latent_stats['teacher_magnitude'],
                        "train/student_magnitude": batch_latent_stats['student_magnitude']
                    })
                
                wandb.log(log_data)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clean training batch progress: only loss and LR (matches Stage 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            loss_name = "Latent_RMSE" if TRAIN_STAGE2_LATENT_ONLY else "Std_RMSE"
            logger.info(f"🏋️  TRAIN | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                       f"{loss_name}: {loss.item():.4f} | LR: {current_lr:.2e}")
            
            # Log gradient norm at debug level for monitoring training health
            if DEBUG_VERBOSE:
                logger.debug(f"🔧 Batch {batch_idx + 1} | Gradient Norm: {grad_norm:.3f}")
            
            # Additional detailed logging at DEBUG level
            if DEBUG_VERBOSE and batch_idx % TRAINING_BATCH_LOG_INTERVAL == 0:
                logger.debug(f"🔍 Detailed: Batch {batch_idx}: Loss = {loss.item():.6f}, "
                           f"Running Avg = {total_loss/num_batches:.6f}")
        
        avg_loss = total_loss / num_batches
        
        # Log epoch summary
        if TRAIN_STAGE2_LATENT_ONLY:
            epoch_latent_stats = self.latent_stats.compute_epoch_stats()
            logger.info(f"📊 TRAIN SUMMARY | Latent_RMSE: {avg_loss:.4f} | "
                       f"Cosine_Sim: {epoch_latent_stats.get('latent_cosine_sim', 0):.4f}")
            
            # Log latent statistics to wandb
            if self.use_wandb and wandb.run:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_latent_rmse": epoch_latent_stats.get('latent_rmse', 0),
                    "train/epoch_cosine_sim": epoch_latent_stats.get('latent_cosine_sim', 0),
                    "train/epoch_teacher_mag": epoch_latent_stats.get('teacher_magnitude', 0),
                    "train/epoch_student_mag": epoch_latent_stats.get('student_magnitude', 0),
                })
        else:
            logger.info(f"📊 TRAIN SUMMARY | Std_RMSE: {avg_loss:.4f}")
        
        if DEBUG_VERBOSE:
            logger.debug(f"✅ Training epoch completed. Average loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def _log_reconstruction_images(self, predictions, targets, nir_measurements, epoch, step=None, phantom_ids=None):
        """Log 3D reconstruction slices to W&B for visualization (using hardened function)."""
        if not self.use_wandb:
            return
            
        try:
            from code.utils.viz_recon import prepare_raw_DHW, log_recon_slices_raw
            
            # Convert tensors from channels-last [B,D,H,W,2] to channel-first [B,2,D,H,W]
            if predictions.shape[-1] == 2:  # Check if channels-last
                predictions = predictions.permute(0, 4, 1, 2, 3).contiguous()  # [B,D,H,W,2] -> [B,2,D,H,W]
            if targets.shape[-1] == 2:  # Check if channels-last
                targets = targets.permute(0, 4, 1, 2, 3).contiguous()  # [B,D,H,W,2] -> [B,2,D,H,W]
            
            # Prepare raw mm^-1 volumes with strict [B,2,D,H,W] format
            # Target is already raw from dataloader, only inverse-standardize prediction
            pred_raw, tgt_raw = prepare_raw_DHW(
                predictions, targets,
                standardizer=self.standardizers.ground_truth_standardizer,
                tgt_is_std=False  # Target is already raw from dataloader
            )
            
            # Log absorption and scattering ranges for visualization insight
            pred_mu_a_min, pred_mu_a_max = float(pred_raw[:,0].min()), float(pred_raw[:,0].max())
            pred_mu_s_min, pred_mu_s_max = float(pred_raw[:,1].min()), float(pred_raw[:,1].max())
            tgt_mu_a_min, tgt_mu_a_max = float(tgt_raw[:,0].min()), float(tgt_raw[:,0].max())
            tgt_mu_s_min, tgt_mu_s_max = float(tgt_raw[:,1].min()), float(tgt_raw[:,1].max())
            
            logger.info(f"[VIZ PRED] μₐ: {pred_mu_a_min:.4f}..{pred_mu_a_max:.4f} mm⁻¹, μ′ₛ: {pred_mu_s_min:.3f}..{pred_mu_s_max:.3f} mm⁻¹")
            logger.info(f"[VIZ TGT]  μₐ: {tgt_mu_a_min:.4f}..{tgt_mu_a_max:.4f} mm⁻¹, μ′ₛ: {tgt_mu_s_min:.3f}..{tgt_mu_s_max:.3f} mm⁻¹")
            
            # Acceptance check
            assert pred_raw[:,0].max() > 1e-5 and pred_raw[:,1].max() > 1e-3, "Zeros after prep; abort recon logging."
            
            # Log exactly 24 images
            log_recon_slices_raw(pred_raw, tgt_raw, epoch, phantom_ids=phantom_ids, prefix="Reconstructions")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log reconstruction images: {e}")
    
    def _compute_raw_rmse_total_only(self, data_loader) -> float:
        """
        Compute only raw RMSE total using current model weights (no EMA).
        
        This helper mirrors the E2E validation path but computes only
        the raw_rmse_total metric for non-EMA sanity checking.
        
        Args:
            data_loader: DataLoader containing validation batches
            
        Returns:
            float: Average raw RMSE total across all batches
        """
        self.model.eval()
        total_rmse = 0.0
        count = 0
        
        # Use same AMP settings as main validation
        use_amp = (self.device.type == "cuda")
        dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        
        with torch.no_grad():
            for batch in data_loader:
                nir_measurements_raw = batch['nir_measurements'].to(self.device)
                raw_ground_truth = batch['ground_truth'].to(self.device)
                
                nir_measurements = self.standardizers.transform_nir_inputs(nir_measurements_raw)
                
                # Get tissue patches if using them
                tissue_patches = None
                if self.use_tissue_patches and 'tissue_patches' in batch:
                    tissue_patches_raw = batch['tissue_patches'].to(self.device)
                    tissue_patches = self.standardizers.transform_tissue_patches(tissue_patches_raw)
                
                # Full forward for student (same as E2E path)
                with autocast(enabled=use_amp, dtype=dtype):
                    student_outputs = self.model(nir_measurements, tissue_patches)
                    student_latent = student_outputs.get('encoded_scan')
                    student_latent_aligned = self.latent_align(student_latent) if hasattr(self, 'latent_align') else student_latent
                    with torch.no_grad():
                        student_reconstruction_std = self.teacher.decode_from_latent(student_latent_aligned)
                
                # Inverse to RAW for RMSE calculation
                raw_predictions = self.standardizers.inverse_transform_ground_truth(student_reconstruction_std)
                
                batch_metrics = calculate_batch_metrics(
                    self.metrics,
                    {'reconstructed': raw_predictions},
                    raw_ground_truth,
                    "stage2"
                )
                total_rmse += float(batch_metrics.get('rmse_overall', 0.0))
                count += 1
        
        return total_rmse / max(count, 1)

    def validate(self, data_loader, epoch=0):
        """
        Evaluate the hybrid model on validation data.
        
        Supports two validation modes:
        1. Latent-only validation: Compute latent space metrics only
        2. End-to-end validation: Full reconstruction validation (every K epochs or standard mode)
        
        Args:
            data_loader: DataLoader containing validation batches
            epoch: Current epoch number for determining validation type
        
        Returns:
            tuple: (loss, metrics_dict) for monitoring
        """
        if not self.standardizers.fitted:
            raise RuntimeError("Standardizers must be fitted before validation.")
        
        logger.debug("🔍 Starting validation epoch...")
        self.model.eval()
        
        # Apply EMA weights if available for better evaluation performance
        if self.ema is not None:
            self.ema.apply_shadow()
            logger.debug("🔄 Applied EMA weights for validation")
        
        total_loss = 0
        num_batches = 0
        
        # Determine validation mode
        if TRAIN_STAGE2_LATENT_ONLY:
            do_e2e_validation = (epoch % VAL_E2E_EVERY_K_EPOCHS == 0)
            if do_e2e_validation:
                logger.debug(f"🎯 E2E validation epoch {epoch} (every {VAL_E2E_EVERY_K_EPOCHS} epochs)")
            else:
                logger.debug(f"🎯 Latent-only validation epoch {epoch}")
        else:
            do_e2e_validation = True  # Always do E2E in standard mode
        
        # Initialize metrics tracking (always initialize, but only use in E2E mode)
        raw_metrics = {
            'raw_rmse_total': 0.0, 'raw_rmse_mu_a': 0.0, 'raw_rmse_mu_s': 0.0,
            'raw_dice': 0.0, 'raw_contrast': 0.0,
            'raw_dice_mu_a': 0.0, 'raw_dice_mu_s': 0.0,  # New per-channel Dice
            'raw_contrast_mu_a': 0.0, 'raw_contrast_mu_s': 0.0  # New per-channel Contrast Ratio
        }
        
        # Initialize latent statistics if in latent mode
        if TRAIN_STAGE2_LATENT_ONLY:
            from .latent_stats import LatentStats
            val_latent_stats = LatentStats()
        
        # Add transformer metrics for Stage 2
        transformer_metrics = {
            'feature_enhancement_ratio': 0.0, 'attention_entropy': 0.0
        }
        
        logger.debug(f"📊 Processing {len(data_loader)} validation batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                logger.debug(f"🔍 Validating batch {batch_idx + 1}/{len(data_loader)}")
                
                # Extract inputs and targets
                nir_measurements_raw = batch['nir_measurements'].to(self.device)
                raw_ground_truth = batch['ground_truth'].to(self.device)  # Shape: (batch_size, 2, 64, 64, 64)
                
                # Standardize inputs for forward pass
                nir_measurements = self.standardizers.transform_nir_inputs(nir_measurements_raw)
                standardized_ground_truth = self.standardizers.ground_truth_standardizer.transform(raw_ground_truth)
                
                # Get and standardize tissue patches if using them
                tissue_patches = None
                if self.use_tissue_patches and 'tissue_patches' in batch:
                    tissue_patches_raw = batch['tissue_patches'].to(self.device)
                    tissue_patches = self.standardizers.transform_tissue_patches(tissue_patches_raw)
                
                logger.debug(f"📦 Validation batch shape: {raw_ground_truth.shape}")
                
                # Guard AMP autocast cleanly
                use_amp = (self.device.type == "cuda")
                amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
                
                logger.debug("⚡ Validation forward pass...")
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    if TRAIN_STAGE2_LATENT_ONLY and not do_e2e_validation:
                        # LATENT-ONLY VALIDATION
                        student_latent = self.model.encode(nir_measurements, tissue_patches)
                        
                        # Apply latent affine aligner for validation consistency
                        student_latent_aligned = self.latent_align(student_latent)
                        
                        # Get teacher latent from standardized ground truth (NOT NIR measurements)
                        with torch.no_grad():
                            teacher_latent = self.teacher.encode_from_gt_std(standardized_ground_truth)
                        
                        # Compute latent loss using aligned student latent
                        loss = self.criterion(student_latent_aligned, teacher_latent)
                        
                        # Update latent statistics using aligned student latent
                        batch_latent_stats = val_latent_stats.update(teacher_latent, student_latent_aligned)
                        
                        logger.info(f"🔍 VALID | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                                   f"Latent_RMSE: {batch_latent_stats['latent_rmse']:.4f} | "
                                   f"Cosine_Sim: {batch_latent_stats['latent_cosine_sim']:.4f}")
                    else:
                        # END-TO-END VALIDATION - Use teacher decoder for both teacher and student
                        # 1. Get teacher baseline (ground truth → teacher encode → teacher decode)
                        with torch.no_grad():
                            teacher_latent = self.teacher.encode_from_gt_std(standardized_ground_truth)
                            teacher_reconstruction_std = self.teacher.decode_from_latent(teacher_latent)
                        
                        # 2. Get student reconstruction via full forward pass to get transformer features
                        student_outputs = self.model(nir_measurements, tissue_patches)
                        student_latent = student_outputs['encoded_scan']  # Get latent from full forward pass
                        student_latent_aligned = self.latent_align(student_latent)  # Apply aligner for consistent decoding
                        with torch.no_grad():
                            student_reconstruction_std = self.teacher.decode_from_latent(student_latent_aligned)
                        
                        # 3. Use student reconstruction for loss (comparison against ground truth)
                        loss = self.criterion(student_reconstruction_std, standardized_ground_truth)
                        
                        # 4. Set outputs for metrics calculation with transformer features
                        outputs = {
                            'reconstructed': student_reconstruction_std,
                            'enhanced_features': student_outputs.get('enhanced_features'),
                            'cnn_features': student_outputs.get('cnn_features'),
                            'attention_weights': student_outputs.get('attention_weights')
                        }
                        
                        # 5. Debug assertions and logging
                        assert teacher_latent.shape == student_latent_aligned.shape, f"Latent shape mismatch: teacher={teacher_latent.shape}, student_aligned={student_latent_aligned.shape}"
                        assert student_reconstruction_std.shape == standardized_ground_truth.shape, f"Reconstruction shape mismatch: {student_reconstruction_std.shape} vs {standardized_ground_truth.shape}"
                        assert not torch.isnan(student_reconstruction_std).any(), "Student reconstruction contains NaNs"
                        
                        if DEBUG_VERBOSE:
                            logger.debug(f"📊 Teacher reconstruction (std): mean={teacher_reconstruction_std.mean():.4f}, std={teacher_reconstruction_std.std():.4f}")
                            logger.debug(f"📊 Student reconstruction (std): mean={student_reconstruction_std.mean():.4f}, std={student_reconstruction_std.std():.4f}")
                        
                        # Removed teacher vs student comparison - using unified visualization instead
                
                # Inverse transform predictions to RAW space for human-interpretable metrics (only for E2E)
                if 'outputs' in locals() and outputs is not None:
                    raw_predictions = self.standardizers.inverse_transform_ground_truth(outputs['reconstructed'])
                    # raw_ground_truth already in raw space (no transformation needed)
                else:
                    raw_predictions = None
                
                logger.debug(f"📊 Standardized validation loss: {loss.item():.6f}")
                
                # Calculate RAW metrics in original μₐ/μ′ₛ space for human interpretation (only for E2E)
                if raw_predictions is not None and 'outputs' in locals() and outputs is not None:
                    with torch.no_grad():
                        # Create outputs dict with raw predictions for metrics calculation (include transformer outputs)
                        raw_outputs = {
                            'reconstructed': raw_predictions,
                            'attention_weights': outputs.get('attention_weights'),
                            'enhanced_features': outputs.get('enhanced_features'),
                            'cnn_features': outputs.get('cnn_features')  # Add this too
                        }
                        batch_raw_metrics = calculate_batch_metrics(
                            self.metrics, raw_outputs, raw_ground_truth, "stage2"
                        )
                        
                        # Map batch metrics to our raw metrics naming
                        raw_metrics['raw_rmse_total'] += batch_raw_metrics.get('rmse_overall', 0)
                        raw_metrics['raw_rmse_mu_a'] += batch_raw_metrics.get('rmse_absorption', 0)
                        raw_metrics['raw_rmse_mu_s'] += batch_raw_metrics.get('rmse_scattering', 0)
                        raw_metrics['raw_dice'] += batch_raw_metrics.get('dice', 0)
                        raw_metrics['raw_contrast'] += batch_raw_metrics.get('contrast_ratio', 0)
                        
                        # Compute per-channel metrics in raw physical space
                        dice_mu_a = dice_per_channel(raw_predictions, raw_ground_truth, channel=0)
                        dice_mu_s = dice_per_channel(raw_predictions, raw_ground_truth, channel=1)
                        cr_mu_a = contrast_ratio_per_channel(raw_predictions, raw_ground_truth, channel=0)
                        cr_mu_s = contrast_ratio_per_channel(raw_predictions, raw_ground_truth, channel=1)
                        
                        # Accumulate per-channel metrics
                        raw_metrics['raw_dice_mu_a'] += dice_mu_a.item()
                        raw_metrics['raw_dice_mu_s'] += dice_mu_s.item()
                        raw_metrics['raw_contrast_mu_a'] += cr_mu_a.item()
                        raw_metrics['raw_contrast_mu_s'] += cr_mu_s.item()
                        
                        # Collect transformer metrics
                        transformer_metrics['feature_enhancement_ratio'] += batch_raw_metrics.get('feature_enhancement_ratio', 0)
                        transformer_metrics['attention_entropy'] += batch_raw_metrics.get('attention_entropy', 0)
                        
                        # Show validation batch progress with RAW metrics (human-interpretable) - match Stage 1
                        logger.info(f"🔍 VALID | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                                   f"Raw_RMSE: {batch_raw_metrics.get('rmse_overall', 0):.4f} | "
                                   f"Dice: {batch_raw_metrics.get('dice', 0):.4f} | "
                                   f"Contrast: {batch_raw_metrics.get('contrast_ratio', 0):.4f}")
                else:
                    # For latent-only validation, set default values for batch_raw_metrics
                    batch_raw_metrics = {'rmse_overall': 0.0, 'dice': 0.0, 'contrast_ratio': 0.0}
                
                total_loss += loss.item()
                num_batches += 1
        
        # Average all metrics
        avg_loss = total_loss / num_batches
        
        # Only average raw metrics if we did E2E validation
        if do_e2e_validation:
            for key in raw_metrics:
                raw_metrics[key] /= num_batches
            for key in transformer_metrics:
                transformer_metrics[key] /= num_batches
                
            logger.debug(f"✅ Validation completed. Average loss: {avg_loss:.6f}")
            logger.info(f"📊 VALID SUMMARY | Raw_RMSE: {raw_metrics['raw_rmse_total']:.4f} | "
                       f"Dice: {raw_metrics['raw_dice']:.4f} | Contrast: {raw_metrics['raw_contrast']:.4f} | "
                       f"μₐ: {raw_metrics['raw_rmse_mu_a']:.4f} | μ′ₛ: {raw_metrics['raw_rmse_mu_s']:.4f}")
            logger.info(f"📊 VALID PER-CHANNEL | Dice μₐ: {raw_metrics['raw_dice_mu_a']:.4f} | Dice μ′ₛ: {raw_metrics['raw_dice_mu_s']:.4f} | "
                       f"CR μₐ: {raw_metrics['raw_contrast_mu_a']:.4f} | CR μ′ₛ: {raw_metrics['raw_contrast_mu_s']:.4f}")
            logger.info(f"🔮 TRANSFORMER | Feature Enhancement: {transformer_metrics['feature_enhancement_ratio']:.4f} | "
                       f"Attention Entropy: {transformer_metrics['attention_entropy']:.4f}")
        else:
            # For latent-only validation
            logger.debug(f"✅ Latent validation completed. Average loss: {avg_loss:.6f}")
            logger.info(f"📊 LATENT SUMMARY | RMSE: {avg_loss:.4f}")
        
        # Combine metrics for return (maintain Stage 1 compatibility but add transformer metrics)
        combined_metrics = {**raw_metrics, **transformer_metrics}
        
        # === Non-EMA sanity check: one metric, one log line ===
        val_rmse_nonema = None
        if self.ema is not None and do_e2e_validation:
            # Switch to raw weights
            self.ema.restore()
            
            try:
                val_rmse_nonema = self._compute_raw_rmse_total_only(data_loader)
                logger.info(f"📊 Validation RMSE (non-EMA): {val_rmse_nonema:.4f}")
                if self.use_wandb and wandb.run:
                    wandb.log({"val/rmse_nonema": val_rmse_nonema, "epoch": epoch})
            finally:
                # Re-apply EMA so the existing end-of-function restore()
                # returns the model to raw weights as before.
                self.ema.apply_shadow()
        
        # Restore original weights if EMA was used
        if self.ema is not None:
            self.ema.restore()
            logger.debug("🔄 Restored original weights after EMA validation")
        
        return avg_loss, combined_metrics
    
    def train(self, data_loaders, epochs=EPOCHS_STAGE2):
        """
        Execute the complete Stage 2 training pipeline.
        
        This method orchestrates the full transformer enhancement training process,
        including standardizer fitting, epoch-wise training and validation, 
        progress monitoring, and automatic checkpoint saving.
        
        Args:
            data_loaders (dict): Dictionary containing 'train' and 'val' DataLoaders
            epochs (int): Number of training epochs to execute. Default from constants
        
        The training process:
        - Fits Stage 2 standardizers on training data (measurements, positions, tissue patches)
        - Trains only unfrozen transformer parameters with standardized inputs
        - Computes validation metrics in physical units via inverse standardization
        - Monitors validation loss for model selection
        - Saves mode-specific checkpoints (baseline/enhanced)
        
        Example:
            >>> # Enhanced mode with tissue patches
            >>> trainer = Stage2Trainer(checkpoint_path, use_tissue_patches=True)
            >>> trainer.train(data_loaders, epochs=EPOCHS_STAGE2)
        """
        # Fit Stage 2 standardizers on training data before training begins
        if not self.standardizers.fitted:
            logger.info("🔧 Fitting Stage 2 standardizers on training data...")
            
            # Use lightweight dataloader for standardizer fitting (no tissue patches)
            standardizer_fit_loader = self._create_lightweight_dataloader_for_standardizer_fitting()
            
            self.standardizers.fit_from_stage1_checkpoint(
                stage1_checkpoint_path=self.stage1_checkpoint_path,
                train_dataloader=standardizer_fit_loader
            )
            logger.info("✅ Stage 2 standardizers fitted successfully!")
        
        # Initialize AdamW optimizer and Linear Warmup + Cosine Decay scheduler
        if self.optimizer is None:
            steps_per_epoch = len(data_loaders['train'])
            self._create_optimizer_and_scheduler(epochs, steps_per_epoch)
        
        # AFTER optimizer/scheduler creation (and after any unfreezing in param-groups)
        if USE_EMA and self.ema is None:
            self.ema = EMAModel(self.model, decay=EMA_DECAY_START)  # Start with initial decay
            self.total_training_steps = epochs * len(data_loaders['train'])  # For EMA ramp
            logger.info(f"🔄 EMA initialized with progressive decay: {EMA_DECAY_START} → {EMA_DECAY_END} over {self.total_training_steps} steps")
        
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
        best_metric_name = "Raw_RMSE"  # Default, will be updated based on final selection
        
        for epoch in range(epochs):
            logger.info(f"📅 Starting Transformer Epoch {epoch + 1}/{epochs}")
            
            # Train: Update transformer parameters (CNN decoder frozen)
            logger.debug(f"🏋️  Beginning transformer training phase for epoch {epoch+1}")
            train_std_loss = self.train_epoch(data_loaders['train'], epoch)
            logger.info(f"🏋️  TRAIN COMPLETE | {'Latent_RMSE' if TRAIN_STAGE2_LATENT_ONLY else 'Std_RMSE'}: {train_std_loss:.4f}")
            
            # Validate: Evaluate hybrid model on unseen data (no parameter updates)
            logger.debug(f"🔍 Beginning transformer validation phase for epoch {epoch+1}")
            val_std_loss, val_raw_metrics = self.validate(data_loaders['val'], epoch)
            
            # Determine actual validation type performed (same logic as in validate())
            if TRAIN_STAGE2_LATENT_ONLY:
                was_e2e_validation = (epoch % VAL_E2E_EVERY_K_EPOCHS == 0)
            else:
                was_e2e_validation = True  # Always E2E in standard mode
            
            # Log with correct metric name based on actual validation performed
            if was_e2e_validation:
                val_loss_name = "Std_RMSE"  # E2E validation returns standardized RMSE
            else:
                val_loss_name = "Latent_RMSE"  # Latent-only validation returns latent RMSE
                
            logger.info(f"🔍 VALID COMPLETE | {val_loss_name}: {val_std_loss:.4f} | Raw_RMSE: {val_raw_metrics['raw_rmse_total']:.4f}")
            
            # Log enhanced metrics to W&B with proper separation of standardized vs raw - MATCH STAGE 1 EXACTLY
            if self.use_wandb and wandb.run:
                # Use correct validation metric name based on actual validation performed
                val_wandb_key = "val/std_rmse" if was_e2e_validation else "val/latent_rmse"
                
                wandb.log({
                    "epoch": epoch + 1,  # Custom x-axis for epoch metrics
                    
                    # === TRAINING LOSS ===
                    "train/latent_rmse" if TRAIN_STAGE2_LATENT_ONLY else "train/loss_std": train_std_loss,
                    val_wandb_key: val_std_loss,               # Main validation metric labeled correctly
                    
                    # === RAW METRICS (human-interpretable) ===
                    "val/raw_rmse_total": val_raw_metrics['raw_rmse_total'],
                    "val/raw_rmse_mu_a": val_raw_metrics['raw_rmse_mu_a'], 
                    "val/raw_rmse_mu_s": val_raw_metrics['raw_rmse_mu_s'],
                    "val/raw_dice": val_raw_metrics['raw_dice'],
                    "val/raw_contrast": val_raw_metrics['raw_contrast'],
                    
                    # === PER-CHANNEL METRICS (new) ===
                    "val/raw_dice_mu_a": val_raw_metrics['raw_dice_mu_a'],
                    "val/raw_dice_mu_s": val_raw_metrics['raw_dice_mu_s'],
                    "val/raw_contrast_mu_a": val_raw_metrics['raw_contrast_mu_a'],
                    "val/raw_contrast_mu_s": val_raw_metrics['raw_contrast_mu_s'],
                    
                    # === TRANSFORMER METRICS (Stage 2 specific) ===
                    "transformer/feature_enhancement_valid": val_raw_metrics['feature_enhancement_ratio'],
                    "transformer/attention_entropy_valid": val_raw_metrics['attention_entropy'],
                    
                    # === LEARNING RATE TRACKING ===
                    "train/lr": self.optimizer.param_groups[0]['lr']
                })
                
                # Log current EMA decay value for monitoring
                if self.ema is not None:
                    current_global_step = (epoch + 1) * len(data_loaders['train'])
                    current_ema_decay = self._compute_ema_decay(current_global_step)
                    wandb.log({
                        "epoch": epoch + 1,
                        "ema/decay": current_ema_decay
                    })
                    logger.info(f"📊 EMA Decay: {current_ema_decay:.6f}")
                
                # Log reconstruction images every epoch
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
                        # Always use teacher decoder for visualization to avoid zero tensors
                        # Standardize inputs for proper model forward
                        nir_measurements = self.standardizers.transform_nir_inputs(measurements)
                        tissue_patches_std = None
                        if tissue_patches is not None:
                            tissue_patches_std = self.standardizers.transform_tissue_patches(tissue_patches)
                        
                        # Get student latent encoding
                        student_latent = self.model.encode(nir_measurements, tissue_patches_std)
                        
                        # Apply latent affine aligner for consistent reconstruction
                        student_latent_aligned = self.latent_align(student_latent)
                        
                        # Decode with teacher to get standardized reconstruction using aligned latent
                        pred_std = self.teacher.decode_from_latent(student_latent_aligned)
                        
                    self._log_reconstruction_images(pred_std, targets, measurements, epoch, phantom_ids=phantom_ids)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log Stage 2 images at epoch {epoch + 1}: {e}")
            
            # Print epoch summary with clear visual formatting - MATCH STAGE 1 EXACTLY
            if epoch % PROGRESS_LOG_INTERVAL == 0 or epoch == epochs - FINAL_EPOCH_OFFSET:
                logger.info(f"")
                logger.info(f"{'='*80}")
                logger.info(f"🚀 EPOCH {epoch+1:3d}/{epochs} SUMMARY")
                logger.info(f"{'='*80}")
                logger.info(f"📈 Train {'Latent_RMSE' if TRAIN_STAGE2_LATENT_ONLY else 'Std_RMSE'}: {train_std_loss:.4f} | Valid {'Latent_RMSE' if TRAIN_STAGE2_LATENT_ONLY else 'Std_RMSE'}: {val_std_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"📊 Valid Raw_RMSE: {val_raw_metrics['raw_rmse_total']:.4f} | "
                        f"Dice: {val_raw_metrics['raw_dice']:.4f} | Contrast: {val_raw_metrics['raw_contrast']:.4f}")
                logger.info(f"📊 Raw μₐ: {val_raw_metrics['raw_rmse_mu_a']:.4f} | Raw μ′ₛ: {val_raw_metrics['raw_rmse_mu_s']:.4f}")
                logger.info(f"🔮 TRANSFORMER | Feature Enhancement: {val_raw_metrics['feature_enhancement_ratio']:.4f} | "
                           f"Attention Entropy: {val_raw_metrics['attention_entropy']:.4f}")
                logger.info(f"{'='*80}")
                
                
            # Log GPU stats every 5 epochs
            if epoch % GPU_MEMORY_LOG_INTERVAL == 0 and torch.cuda.is_available():
                try:
                    log_gpu_stats()
                except NameError:
                    logger.debug("log_gpu_stats() not available - skipping GPU memory logging")
            
            # Learning rate scheduling (Linear Warmup + Cosine Decay updates per-batch)
            # No epoch-level action needed - scheduler updates per-batch automatically
            
            # Determine validation mode for metric selection
            if TRAIN_STAGE2_LATENT_ONLY:
                is_e2e_epoch = (epoch % VAL_E2E_EVERY_K_EPOCHS == 0)
            else:
                is_e2e_epoch = True  # Always E2E in standard mode
            
            # Save best model - use EMA Raw RMSE when available, otherwise standardized RMSE
            if is_e2e_epoch and val_raw_metrics['raw_rmse_total'] != 0.0:
                selection_metric = val_raw_metrics['raw_rmse_total']   # EMA path for E2E validation
                metric_name = "Raw_RMSE"
            else:
                selection_metric = val_std_loss   # Fallback for latent-only validation
                metric_name = "Latent_RMSE" if TRAIN_STAGE2_LATENT_ONLY else "Std_RMSE"
                
            if selection_metric < best_val_loss:
                improvement = best_val_loss - selection_metric
                best_val_loss = selection_metric
                best_metric_name = metric_name  # Store for final summary
                self.patience_counter = 0  # Reset patience counter on improvement
                logger.info(f"🎉 NEW BEST MODEL | Improvement: {improvement:.4f} | Best {metric_name}: {best_val_loss:.4f}")
                
                # Prepare checkpoint data
                metrics = {
                    'val_std_rmse': val_std_loss,  # Always log for visibility
                    'epoch': epoch
                }
                if is_e2e_epoch:
                    metrics['val_raw_rmse'] = val_raw_metrics['raw_rmse_total']  # Selection metric when available
                
                checkpoint_data = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': metrics,
                    'use_tissue_patches': self.use_tissue_patches
                }
                
                save_checkpoint(self.checkpoint_path, checkpoint_data, selection_metric)
            else:
                self.patience_counter += 1
                logger.debug(f"📊 Stage 2 no improvement. Current: {selection_metric:.6f}, Best: {best_val_loss:.6f}, Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                # Check for early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"")
                    logger.info(f"🛑 EARLY STOPPING TRIGGERED")
                    logger.info(f"🔄 No improvement for {self.early_stopping_patience} epochs")
                    logger.info(f"🏆 Best {metric_name}: {best_val_loss:.4f}")
                    self.early_stopped = True
                    break
        
        mode = "Enhanced" if self.use_tissue_patches else "Baseline"
        logger.info(f"")
        logger.info(f"{'='*80}")
        if self.early_stopped:
            logger.info(f"✅ TRANSFORMER TRAINING COMPLETED ({mode}) - Early Stopped")
        else:
            logger.info(f"✅ TRANSFORMER TRAINING COMPLETED ({mode}) - Full Training")
        logger.info(f"🏆 Best {best_metric_name}: {best_val_loss:.4f}")
        logger.info(f"📊 Final Epoch: {epoch+1}/{epochs}")
        logger.info(f"{'='*80}")
        
        logger.debug(f"🏁 Training summary: Completed epochs: {epoch+1}, Final best loss: {best_val_loss:.6f}")
        
        # Finish W&B run
        if self.use_wandb:
            wandb.log({"System/final_best_val_loss": best_val_loss, "System/final_mode": mode, "System/early_stopped": self.early_stopped}, commit=False)
            wandb.finish()
            logger.info("🔬 W&B experiment finished")
        
        return {'best_val_loss': best_val_loss, 'early_stopped': self.early_stopped}
