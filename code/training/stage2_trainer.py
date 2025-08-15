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
from code.utils.metrics import NIRDOTMetrics, create_metrics_for_stage, calculate_batch_metrics, RMSELoss, dice_per_channel, contrast_ratio_per_channel
from code.utils.standardizers import Stage2StandardizerCollection
from code.data_processing.data_loader import create_phantom_dataloaders  # For standardizer fitting
from .training_config import *  # Import all training config

# =============================================================================
# PERFORMANCE OPTIMIZATIONS
# =============================================================================

# Enable TensorFloat32 for better A100 performance (suppresses the warning)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    logger.debug("✅ Enabled TensorFloat32 for optimized A100 matrix multiplication")

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
            early_stopping_patience (int): Early stopping patience in epochs. Default: 25
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
            
            # Initialize latent statistics tracker
            self.latent_stats = LatentStats()
            
            # Override loss function to latent RMSE
            self.criterion = compute_latent_rmse
            logger.info("📊 Stage 2 using latent RMSE loss (teacher-student)")
        else:
            logger.info("🔄 STANDARD TRAINING MODE (end-to-end reconstruction)")
            self.teacher = None
            self.latent_stats = None
    
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
        wandb.define_metric("Metrics/*", step_metric="epoch")
        wandb.define_metric("RMSE_Details/*", step_metric="epoch")
        wandb.define_metric("System/*", step_metric="epoch")
        wandb.define_metric("Analysis/*", step_metric="epoch")
        wandb.define_metric("Reconstructions/*", step_metric="epoch")
        
        logger.info(f"🔬 W&B experiment initialized: {experiment_name}")
    
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
        
        # Count trainable in each component
        cnn_decoder_trainable = sum(p.numel() for p in self.model.cnn_autoencoder.decoder.parameters() if p.requires_grad)
        embedding_trainable = sum(p.numel() for p in self.model.spatially_aware_encoder.parameters() if p.requires_grad)
        transformer_trainable = sum(p.numel() for p in self.model.transformer_encoder.parameters() if p.requires_grad)
        pooling_trainable = sum(p.numel() for p in self.model.global_pooling_encoder.parameters() if p.requires_grad)
        
        logger.info("📊 STAGE 2 PARAMETER BREAKDOWN (AFTER FREEZING):")
        logger.info("   ┌─────────────────────────────────────────────────────────────┐")
        logger.info(f"   │ CNN Decoder:           {cnn_decoder:>8,} params ({cnn_decoder_trainable:>8,} trainable) │")
        logger.info(f"   │ Spatially-Aware Embed: {embedding_total:>8,} params ({embedding_trainable:>8,} trainable) │")
        logger.info(f"   │ Transformer Encoder:   {transformer_total:>8,} params ({transformer_trainable:>8,} trainable) │")
        logger.info(f"   │ Global Pooling:        {pooling_total:>8,} params ({pooling_trainable:>8,} trainable) │")
        logger.info("   ├─────────────────────────────────────────────────────────────┤")
        logger.info(f"   │ TOTAL MODEL:           {total_params:>8,} params                 │")
        logger.info(f"   │ TRAINABLE:             {trainable_params:>8,} params                 │")
        logger.info(f"   │ FROZEN:                {frozen_params:>8,} params                 │")
        logger.info("   └─────────────────────────────────────────────────────────────┘")
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
        
        Based on BERT, GPT, and ViT training procedures + ChatGPT recommendations.
        
        Returns:
            list: Parameter groups for AdamW optimizer
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # NO weight decay for: biases, norms, and specific embedding parameters (critical for gradient flow)
                if (name.endswith(".bias") or 
                    "norm" in name.lower() or 
                    "ln" in name.lower() or
                    "layer_norm" in name.lower() or 
                    name.endswith("embedding.weight") or
                    "pos_embed" in name.lower() or
                    "token_type_embedding" in name):
                    no_decay_params.append(param)
                    logger.debug(f"🚫 No decay: {name}")
                else:
                    decay_params.append(param)
                    logger.debug(f"✅ With decay: {name}")
        
        logger.info(f"[AdamW Groups] decay: {len(decay_params)} params, no_decay: {len(no_decay_params)} params")
        logger.info(f"📊 Parameter Groups (CRITICAL for transformer training):")
        logger.info(f"   ├─ With weight decay: {len(decay_params)} groups")
        logger.info(f"   └─ No weight decay: {len(no_decay_params)} groups (norms/biases/embeddings)")
        
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
                # Linear warmup: Start at 1% of base LR, grow to 100% of base LR
                # This prevents the learning rate from being too small at the start
                min_lr_factor = 0.01  # Start at 1% of base LR instead of 0%
                warmup_factor = min_lr_factor + (1.0 - min_lr_factor) * (step / max(1, warmup_steps))
                logger.debug(f"LR Schedule - Step {step}: Warmup factor = {warmup_factor:.6f}")
                return warmup_factor
            
            # Cosine decay: 1.0 → eta_min
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            eta_min = STAGE2_ETA_MIN_PCT  # Final LR = 5% of peak
            cosine_factor = eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * progress))
            logger.debug(f"LR Schedule - Step {step}: Cosine factor = {cosine_factor:.6f}, Progress = {progress:.3f}")
            return cosine_factor
        
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
        logger.info(f"   ├─ Starting LR (step 0): {STAGE2_BASE_LR * 0.01:.2e}")
        logger.info(f"   ├─ Peak LR (after warmup): {STAGE2_BASE_LR:.2e}")
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
            
            logger.debug(f"� Standardized NIR measurements shape: {nir_measurements.shape}")
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
            
            # Use bfloat16 if available to avoid gradient underflow issues (ChatGPT recommendation)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with autocast(dtype=dtype):
                if TRAIN_STAGE2_LATENT_ONLY:
                    # LATENT-ONLY TRAINING MODE
                    # Forward pass only through encoder to get student latent
                    student_latent = self.model.encode(nir_measurements, tissue_patches)
                    
                    # Get teacher latent from standardized ground truth (NOT NIR measurements)
                    with torch.no_grad():
                        teacher_latent = self.teacher.encode_from_gt_std(targets)
                    
                    # Compute latent RMSE loss
                    loss = self.criterion(student_latent, teacher_latent)
                    
                    # Update latent statistics
                    batch_latent_stats = self.latent_stats.update(teacher_latent, student_latent)
                    
                    # Debug: Assert correct shapes and no NaNs
                    assert teacher_latent.shape == student_latent.shape == (targets.shape[0], LATENT_DIM), \
                        f"Latent shape mismatch: teacher={teacher_latent.shape}, student={student_latent.shape}, expected=({targets.shape[0]}, {LATENT_DIM})"
                    assert not torch.isnan(student_latent).any(), "Student latent contains NaNs"
                    assert not torch.isnan(teacher_latent).any(), "Teacher latent contains NaNs"
                    
                    # Log latent stats at debug level
                    if DEBUG_VERBOSE:
                        logger.debug(f"🎯 Latent RMSE: {batch_latent_stats['latent_rmse']:.4f}, "
                                   f"Cosine Sim: {batch_latent_stats['latent_cosine_sim']:.4f}")
                        logger.debug(f"📊 Teacher latent: mean={teacher_latent.mean():.4f}, std={teacher_latent.std():.4f}, "
                                   f"min={teacher_latent.min():.4f}, max={teacher_latent.max():.4f}, norm={teacher_latent.norm():.4f}")
                        logger.debug(f"📊 Student latent: mean={student_latent.mean():.4f}, std={student_latent.std():.4f}, "
                                   f"min={student_latent.min():.4f}, max={student_latent.max():.4f}, norm={student_latent.norm():.4f}")
                else:
                    # STANDARD END-TO-END TRAINING MODE
                    outputs = self.model(nir_measurements, tissue_patches)
                    logger.debug(f"📤 Stage 2 model output shape: {outputs['reconstructed'].shape}")
                    
                    # DEBUGGING: Log transformer activity details only at the final batch of epoch
                    if (batch_idx + 1) == len(data_loader):
                        attn_weights = outputs['attention_weights']
                        logger.info(f"🧠 Transformer attention stats: shape={attn_weights.shape}, "
                                  f"min={attn_weights.min():.4f}, max={attn_weights.max():.4f}, "
                                  f"mean={attn_weights.mean():.4f}")
                    
                    if (batch_idx + 1) == len(data_loader) and 'enhanced_features' in outputs and outputs['enhanced_features'] is not None:
                        features = outputs['enhanced_features']
                        logger.info(f"✨ Enhanced features stats: shape={features.shape}, "
                                  f"min={features.min():.4f}, max={features.max():.4f}, "
                                  f"mean={features.mean():.4f}, std={features.std():.4f}")
                    
                    # SAFETY: Check for NaN values immediately after forward pass
                    if torch.isnan(outputs['reconstructed']).any():
                        logger.error(f"🚨 NaN detected in model output at batch {batch_idx}")
                        raise ValueError(f"NaN detected in model output - stopping training at batch {batch_idx}")
                    
                    # Compute standardized RMSE loss (matches Stage 1)
                    loss = self.criterion(outputs['reconstructed'], targets)
                
                # SAFETY: Check for NaN loss immediately
                if torch.isnan(loss):
                    logger.error(f"🚨 NaN loss detected at batch {batch_idx}")
                    raise ValueError(f"NaN loss detected - stopping training at batch {batch_idx}")
            
            # Backward pass and optimization (Mixed Precision enabled)
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (prevents exploding gradients in transformer training)
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_MAX_NORM)
            
            # SAFETY: Check for extreme gradient norms (sign of training instability)
            if grad_norm > GRADIENT_MONITOR_THRESHOLD:
                if DEBUG_VERBOSE:
                    logger.warning(f"⚠️ High gradient norm detected: {grad_norm:.3f} (threshold: {GRADIENT_MONITOR_THRESHOLD})")
            
            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()  # Step-wise LR scheduling for linear warmup + cosine decay
            
            # Wandb logging (matches Stage 1 pattern exactly)
            if self.use_wandb and wandb.run:
                current_lr = self.optimizer.param_groups[0]['lr']
                global_step = epoch * len(data_loader) + batch_idx + 1
                
                log_data = {
                    "training_step": global_step,
                    "train/latent_rmse": loss.item(),  # Use latent_rmse as primary training metric
                    "train/lr": current_lr
                }
                
                # Add latent-specific metrics if in latent-only mode
                if TRAIN_STAGE2_LATENT_ONLY:
                    log_data.update({
                        "train/latent_rmse": batch_latent_stats['latent_rmse'],
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
            if DEBUG_VERBOSE and batch_idx % 10 == 0:
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
        """Log 3D reconstruction slices to W&B for visualization (using shared function)."""
        if not self.use_wandb:
            return
            
        # Use shared visualization function with proper inverse standardization
        from code.utils.visualization import log_reconstruction_images_to_wandb
        log_reconstruction_images_to_wandb(
            predictions=predictions, 
            targets=targets, 
            epoch=epoch, 
            prefix="Reconstructions", 
            step=step, 
            phantom_ids=phantom_ids,
            gt_standardizer=self.standardizers.ground_truth_standardizer,  # CRITICAL: Pass standardizer for inverse transform
            add_autocontrast_preview=True
        )
    
    def _log_teacher_student_comparison(self, student_recon, teacher_recon, targets, epoch, phantom_ids=None):
        """Log teacher vs student reconstruction comparison for E2E validation."""
        if not self.use_wandb:
            return
            
        try:
            import wandb
            
            # Apply inverse standardization for visualization using standardizers
            student_raw = self.standardizers.ground_truth_standardizer.inverse_transform(student_recon)
            teacher_raw = self.standardizers.ground_truth_standardizer.inverse_transform(teacher_recon) 
            targets_raw = self.standardizers.ground_truth_standardizer.inverse_transform(targets)
            
            # Physical range clamping for proper visualization
            def clamp_to_physical_range(tensor):
                # mu_a: [0.01, 0.1] mm^-1, mu_s: [5, 25] mm^-1 (typical values)
                mu_a_min, mu_a_max = 0.001, 0.2   # Conservative bounds
                mu_s_min, mu_s_max = 1.0, 50.0
                
                clamped = tensor.clone()
                clamped[:, 0] = torch.clamp(clamped[:, 0], mu_a_min, mu_a_max)  # mu_a
                clamped[:, 1] = torch.clamp(clamped[:, 1], mu_s_min, mu_s_max)  # mu_s
                return clamped
            
            student_raw = clamp_to_physical_range(student_raw)
            teacher_raw = clamp_to_physical_range(teacher_raw)
            targets_raw = clamp_to_physical_range(targets_raw)
            
            # Log comparison images (first sample in batch)
            images = []
            
            # Extract middle slice for visualization (Z//2)
            z_mid = student_raw.shape[-1] // 2
            
            # Student reconstruction
            student_slice = student_raw[0, :, :, :, z_mid].cpu().numpy()  # [2, H, W]
            # Teacher reconstruction  
            teacher_slice = teacher_raw[0, :, :, :, z_mid].cpu().numpy()  # [2, H, W]
            # Ground truth
            target_slice = targets_raw[0, :, :, :, z_mid].cpu().numpy()   # [2, H, W]
            
            phantom_id = phantom_ids[0] if phantom_ids is not None else "unknown"
            
            # Log mu_a comparison
            images.append(wandb.Image(
                student_slice[0], 
                caption=f"Student μa (Phantom {phantom_id})"
            ))
            images.append(wandb.Image(
                teacher_slice[0], 
                caption=f"Teacher μa (Phantom {phantom_id})"
            ))
            images.append(wandb.Image(
                target_slice[0], 
                caption=f"GT μa (Phantom {phantom_id})"
            ))
            
            # Log mu_s comparison
            images.append(wandb.Image(
                student_slice[1], 
                caption=f"Student μs (Phantom {phantom_id})"
            ))
            images.append(wandb.Image(
                teacher_slice[1], 
                caption=f"Teacher μs (Phantom {phantom_id})"
            ))
            images.append(wandb.Image(
                target_slice[1], 
                caption=f"GT μs (Phantom {phantom_id})"
            ))
            
            wandb.log({
                f"Teacher_vs_Student_E2E_Epoch_{epoch}": images
            })  # Remove step parameter to use global step
            
            if DEBUG_VERBOSE:
                logger.debug(f"✅ Logged teacher vs student comparison for epoch {epoch}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to log teacher vs student comparison: {e}")
    
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
        total_loss = 0
        num_batches = 0
        
        # Determine validation mode
        if TRAIN_STAGE2_LATENT_ONLY:
            do_e2e_validation = (epoch % VAL_E2E_EVERY_K_EPOCHS == 0)
            if do_e2e_validation:
                logger.info(f"🎯 E2E validation epoch {epoch} (every {VAL_E2E_EVERY_K_EPOCHS} epochs)")
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
                
                # Use same dtype as training for consistency
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                logger.debug("⚡ Validation forward pass...")
                with autocast(dtype=dtype):
                    if TRAIN_STAGE2_LATENT_ONLY and not do_e2e_validation:
                        # LATENT-ONLY VALIDATION
                        student_latent = self.model.encode(nir_measurements, tissue_patches)
                        
                        # Get teacher latent from standardized ground truth (NOT NIR measurements)
                        with torch.no_grad():
                            teacher_latent = self.teacher.encode_from_gt_std(standardized_ground_truth)
                        
                        # Compute latent loss
                        loss = self.criterion(student_latent, teacher_latent)
                        
                        # Update latent statistics
                        batch_latent_stats = val_latent_stats.update(teacher_latent, student_latent)
                        
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
                        with torch.no_grad():
                            student_reconstruction_std = self.teacher.decode_from_latent(student_latent)
                        
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
                        assert teacher_latent.shape == student_latent.shape, f"Latent shape mismatch: teacher={teacher_latent.shape}, student={student_latent.shape}"
                        assert student_reconstruction_std.shape == standardized_ground_truth.shape, f"Reconstruction shape mismatch: {student_reconstruction_std.shape} vs {standardized_ground_truth.shape}"
                        assert not torch.isnan(student_reconstruction_std).any(), "Student reconstruction contains NaNs"
                        
                        if DEBUG_VERBOSE:
                            logger.debug(f"📊 Teacher reconstruction (std): mean={teacher_reconstruction_std.mean():.4f}, std={teacher_reconstruction_std.std():.4f}")
                            logger.debug(f"📊 Student reconstruction (std): mean={student_reconstruction_std.mean():.4f}, std={student_reconstruction_std.std():.4f}")
                        
                        # 6. Log teacher vs student comparison for first batch of epoch
                        if batch_idx == 0 and epoch % VAL_E2E_EVERY_K_EPOCHS == 0:
                            phantom_ids = batch.get('phantom_id', [f"batch_{batch_idx}_sample_{i}" for i in range(raw_ground_truth.size(0))])
                            if hasattr(phantom_ids, 'cpu'):
                                phantom_ids = phantom_ids.cpu().numpy()
                            self._log_teacher_student_comparison(
                                student_reconstruction_std, 
                                teacher_reconstruction_std, 
                                standardized_ground_truth, 
                                epoch, 
                                phantom_ids=phantom_ids
                            )
                
                # Inverse transform predictions to RAW space for human-interpretable metrics (only for E2E)
                if 'outputs' in locals() and outputs is not None:
                    raw_predictions = self.standardizers.inverse_transform_ground_truth(outputs['reconstructed'])
                    # raw_ground_truth already in raw space (no transformation needed)
                else:
                    raw_predictions = None
                
                logger.debug(f"� Standardized validation loss: {loss.item():.6f}")
                
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
            logger.info(f"� VALID CHAN | Dice μₐ: {raw_metrics['raw_dice_mu_a']:.4f} | Dice μ′ₛ: {raw_metrics['raw_dice_mu_s']:.4f} | "
                       f"CR μₐ: {raw_metrics['raw_contrast_mu_a']:.4f} | CR μ′ₛ: {raw_metrics['raw_contrast_mu_s']:.4f}")
            logger.info(f"�🔮 TRANSFORMER | Feature Enhancement: {transformer_metrics['feature_enhancement_ratio']:.4f} | "
                       f"Attention Entropy: {transformer_metrics['attention_entropy']:.4f}")
        else:
            # For latent-only validation
            logger.debug(f"✅ Latent validation completed. Average loss: {avg_loss:.6f}")
            logger.info(f"📊 LATENT SUMMARY | RMSE: {avg_loss:.4f}")
        
        # Combine metrics for return (maintain Stage 1 compatibility but add transformer metrics)
        combined_metrics = {**raw_metrics, **transformer_metrics}
        
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
            train_std_loss = self.train_epoch(data_loaders['train'], epoch)
            logger.info(f"🏋️  TRAIN COMPLETE | Std_RMSE: {train_std_loss:.4f}")
            
            # Validate: Evaluate hybrid model on unseen data (no parameter updates)
            logger.debug(f"🔍 Beginning transformer validation phase for epoch {epoch+1}")
            val_std_loss, val_raw_metrics = self.validate(data_loaders['val'])
            logger.info(f"🔍 VALID COMPLETE | Std_RMSE: {val_std_loss:.4f} | Raw_RMSE: {val_raw_metrics['raw_rmse_total']:.4f}")
            
            # Log enhanced metrics to W&B with proper separation of standardized vs raw - MATCH STAGE 1 EXACTLY
            if self.use_wandb and wandb.run:
                wandb.log({
                    "epoch": epoch + 1,  # Custom x-axis for epoch metrics
                    
                    # === TRAINING LOSS (latent-only) ===
                    "train/latent_rmse": train_std_loss,        # Primary training metric (latent RMSE)
                    "val/loss_std": val_std_loss,               # Main validation metric for early stopping
                    
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
                        outputs = self.model(measurements, tissue_patches)
                    self._log_reconstruction_images(outputs['reconstructed'], targets, measurements, epoch, phantom_ids=phantom_ids)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log Stage 2 images at epoch {epoch + 1}: {e}")
            
            # Print epoch summary with clear visual formatting - MATCH STAGE 1 EXACTLY
            if epoch % PROGRESS_LOG_INTERVAL == 0 or epoch == epochs - FINAL_EPOCH_OFFSET:
                logger.info(f"")
                logger.info(f"{'='*80}")
                logger.info(f"🚀 EPOCH {epoch+1:3d}/{epochs} SUMMARY")
                logger.info(f"{'='*80}")
                logger.info(f"📈 Train Std_RMSE: {train_std_loss:.4f} | Valid Std_RMSE: {val_std_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"📊 Valid Raw_RMSE: {val_raw_metrics['raw_rmse_total']:.4f} | "
                        f"Dice: {val_raw_metrics['raw_dice']:.4f} | Contrast: {val_raw_metrics['raw_contrast']:.4f}")
                logger.info(f"📊 Raw μₐ: {val_raw_metrics['raw_rmse_mu_a']:.4f} | Raw μ′ₛ: {val_raw_metrics['raw_rmse_mu_s']:.4f}")
                logger.info(f"🔮 TRANSFORMER | Feature Enhancement: {val_raw_metrics['feature_enhancement_ratio']:.4f} | "
                           f"Attention Entropy: {val_raw_metrics['attention_entropy']:.4f}")
                logger.info(f"{'='*80}")
                
                
            # Log GPU stats every 5 epochs
            if epoch % 5 == 0 and torch.cuda.is_available():
                log_gpu_stats()
            
            # Learning rate scheduling (Linear Warmup + Cosine Decay updates per-batch)
            # No epoch-level action needed - scheduler updates per-batch automatically
            
            # Save best model - use standardized loss for consistency with Stage 1
            if val_std_loss < best_val_loss:
                improvement = best_val_loss - val_std_loss
                best_val_loss = val_std_loss
                checkpoint_filename = CHECKPOINT_STAGE2_ENHANCED if self.use_tissue_patches else CHECKPOINT_STAGE2_BASELINE
                checkpoint_path = f"{CHECKPOINT_BASE_DIR}/{checkpoint_filename}"
                logger.info(f"🎉 NEW BEST MODEL | Improvement: {improvement:.4f} | Best Std_RMSE: {best_val_loss:.4f}")
                self.save_checkpoint(checkpoint_path, epoch, val_std_loss)
            else:
                logger.debug(f"📊 Stage 2 no improvement. Current: {val_std_loss:.6f}, Best: {best_val_loss:.6f}")
        
        mode = "Enhanced" if self.use_tissue_patches else "Baseline"
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"✅ TRANSFORMER TRAINING COMPLETED ({mode})")
        logger.info(f"🏆 Best Std_RMSE Loss: {best_val_loss:.4f}")
        logger.info(f"📊 Total Epochs: {epochs}")
        logger.info(f"{'='*80}")
        
        logger.debug(f"🏁 Training summary: Completed epochs: {epochs}, Final best loss: {best_val_loss:.6f}")
        
        # Finish W&B run
        if self.use_wandb:
            wandb.log({"System/final_best_val_loss": best_val_loss, "System/final_mode": mode}, commit=False)
            wandb.finish()
            logger.info("🔬 W&B experiment finished")
        
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
        mode = "Enhanced" if self.use_tissue_patches else "Baseline"
        logger.info(f"💾 ✅ CHECKPOINT SAVED | Mode: {mode} | Path: {path} | Epoch: {epoch+1} | Val Loss: {val_loss:.6f}")
