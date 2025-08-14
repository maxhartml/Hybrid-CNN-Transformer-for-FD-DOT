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
                # NO weight decay for: biases, norms, and embeddings (critical for gradient flow)
                if (name.endswith(".bias") or 
                    "norm" in name.lower() or 
                    "ln" in name.lower() or
                    "layer_norm" in name.lower() or 
                    "embedding" in name.lower() or
                    "pos_embed" in name.lower()):
                    no_decay_params.append(param)
                    logger.debug(f"🚫 No decay: {name}")
                else:
                    decay_params.append(param)
                    logger.debug(f"✅ With decay: {name}")
        
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
            'dice': 0.0, 'contrast_ratio': 0.0, 'rmse_overall': 0.0,
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
                # The hybrid model handles: NIR measurements (batch, 256_subsampled, 8) → transformer → CNN decoder → reconstruction
                # Note: 256 measurements are subsampled from 1000 generated measurements for data augmentation
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
                    logger.error(f"🔍 NIR measurements stats: min={nir_measurements.min():.6f}, max={nir_measurements.max():.6f}, mean={nir_measurements.mean():.6f}")
                    logger.error(f"🔍 Output stats: min={outputs['reconstructed'].min():.6f}, max={outputs['reconstructed'].max():.6f}")
                    raise ValueError(f"NaN detected in model output - stopping training at batch {batch_idx}")
                
                # Compute loss
                logger.debug("📏 Computing Stage 2 RMSE loss...")
                loss = self.criterion(outputs['reconstructed'], targets)
                
                # SAFETY: Check for NaN loss immediately
                if torch.isnan(loss):
                    logger.error(f"🚨 NaN loss detected at batch {batch_idx}")
                    logger.error(f"🔍 Loss value: {loss}")
                    raise ValueError(f"NaN loss detected - stopping training at batch {batch_idx}")
                    
                logger.debug(f"💰 Stage 2 batch loss: {loss.item():.6f}")
            
            # Backward pass with safe AMP pattern (prevents unscale_() crashes)
            logger.debug("🔙 Starting Stage 2 backward pass (only transformer gradients)...")
            try:
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # CRITICAL: Unscale ONCE per step → then clip → decide to step/skip
                self.scaler.unscale_(self.optimizer)
                
                # Apply adaptive gradient clipping (ChatGPT recommendation)
                # Tighter clipping for first 5 epochs, then relax
                clip_norm = 0.5 if epoch < 5 else 1.0
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=clip_norm, 
                    norm_type=2.0
                )
                
                # Log adaptive clipping info
                if batch_idx == 0:  # Log once per epoch
                    logger.info(f"🎯 Adaptive clipping: epoch {epoch} using max_norm={clip_norm}")
                
                # SAFETY: Check for extremely high gradients or non-finite values
                skip_step = False
                if not torch.isfinite(grad_norm) or grad_norm > 10.0:
                    logger.warning(f"⚡ Skipping optimizer step - grad_norm: {grad_norm:.4f}")
                    skip_step = True
                elif grad_norm > GRADIENT_MONITOR_THRESHOLD:
                    logger.warning(f"⚠️ High gradient norm detected: {grad_norm:.4f} > {GRADIENT_MONITOR_THRESHOLD}")
                
                # SAFETY: Check for NaN gradients before optimizer step
                if not skip_step:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            logger.warning(f"⚡ NaN gradient detected in {name} - skipping step")
                            skip_step = True
                            break
                
                # Execute or skip optimizer step
                if skip_step:
                    # Clear gradients and update scaler (important for scale backoff)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()  # Update scaler so it backs off scale
                else:
                    # Normal optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Linear Warmup + Cosine Decay updates per-batch
                    self.scheduler.step()
                
                # DEBUGGING: Log transformer parameter statistics only at final batch of epoch
                if (batch_idx + 1) == len(data_loader):
                    transformer_grad_norm = 0.0
                    transformer_param_count = 0
                    for name, param in self.model.transformer_encoder.named_parameters():
                        if param.grad is not None:
                            transformer_grad_norm += param.grad.norm().item() ** 2
                            transformer_param_count += 1
                    
                    if transformer_param_count > 0:
                        transformer_grad_norm = (transformer_grad_norm ** 0.5)
                        logger.info(f"� Transformer gradients: norm={transformer_grad_norm:.4f}, "
                                  f"params_updated={transformer_param_count}")
                    
                    # Log parameter statistics for transformer
                    for name, param in self.model.transformer_encoder.named_parameters():
                        if param.requires_grad and 'weight' in name:
                            grad_norm_val = param.grad.norm().item() if param.grad is not None else 0.0
                            logger.info(f"🎛️ {name}: mean={param.data.mean().item():.6f}, "
                                      f"std={param.data.std().item():.6f}, "
                                      f"grad_norm={grad_norm_val:.6f}")
                    
                    # ChatGPT diagnostic: Check for zero gradients (key problem indicator)
                    zero_grad_count = 0
                    total_params = 0
                    for name, param in self.model.transformer_encoder.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            total_params += 1
                            if param.grad.norm().item() < 1e-8:
                                zero_grad_count += 1
                    
                    if zero_grad_count > 0:
                        logger.warning(f"🚨 Zero gradients detected: {zero_grad_count}/{total_params} transformer params")
                        logger.warning(f"   This indicates frozen learning - check LR, weight decay, and AMP scaling")
                
                # Log learning rate every 5 batches to avoid W&B buffer warnings
                if not skip_step and batch_idx % LOG_LR_EVERY_N_BATCHES == 0:
                    self._log_learning_rate_to_wandb(epoch, batch_idx, len(data_loader))
            except RuntimeError as e:
                logger.error(f"🚨 Gradient error: {e}")
                raise e
            
            total_loss += loss.item()
            num_batches += 1
            
            # ChatGPT early diagnostic: Enhanced logging for first few batches
            if epoch < 3 and batch_idx < 3:
                logger.info(f"🔬 Early diagnostic - Epoch {epoch}, Batch {batch_idx}:")
                logger.info(f"   ├─ Global grad norm: {grad_norm:.6f}")
                logger.info(f"   ├─ Loss: {loss.item():.6f}")
                logger.info(f"   ├─ Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"   ├─ Skip step: {skip_step}")
                logger.info(f"   └─ Reconstruction range: [{outputs['reconstructed'].min().item():.6f}, {outputs['reconstructed'].max().item():.6f}]")
            
            # Calculate enhanced metrics for this batch
            with torch.no_grad():
                batch_metrics = calculate_batch_metrics(
                    self.metrics, outputs, targets, "stage2"
                )
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
            
            # Show batch progress with standardized metrics format (including transformer metrics)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"🏋️ TRAIN | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                       f"RMSE: {loss.item():.4f} | Dice: {batch_metrics.get('dice', 0):.4f} | "
                       f"Contrast: {batch_metrics.get('contrast_ratio', 0):.4f} | "
                       f"Enhancement: {batch_metrics.get('feature_enhancement_ratio', 0):.4f} | "
                       f"Attention: {batch_metrics.get('attention_entropy', 0):.4f} | LR: {current_lr:.2e}")
            
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
        
        # ChatGPT diagnostic: Track attention entropy to detect frozen attention
        attention_entropies = self._calculate_attention_entropy()
        if attention_entropies:
            entropy_str = ", ".join([f"{k}: {v:.3f}" for k, v in attention_entropies.items()])
            logger.info(f"🧠 Attention Entropies: {entropy_str}")
            
            # Key diagnostic: Average entropy should be dropping from ~5.545 (uniform)
            avg_entropy = sum(attention_entropies.values()) / len(attention_entropies)
            uniform_entropy = math.log(256)  # ~5.545 for 256 tokens
            entropy_drop = uniform_entropy - avg_entropy
            
            # ChatGPT's critical diagnostic
            if avg_entropy > 5.5:
                logger.warning(f"⚠️ High attention entropy ({avg_entropy:.3f}) - attention may be frozen/uniform")
                if epoch >= 3:
                    logger.error(f"🚨 After epoch 3, entropy should be dropping! Check LR, weight decay, gradients")
            else:
                logger.info(f"✅ Good attention specialization - entropy drop: {entropy_drop:.3f}")
            
            # Log to W&B if available
            if self.use_wandb and self._wandb_initialized:
                wandb.log({f"attention/{k}": v for k, v in attention_entropies.items()}, step=epoch)
                wandb.log({
                    "attention/avg_entropy": avg_entropy,
                    "attention/uniform_baseline": uniform_entropy,
                    "attention/entropy_drop": entropy_drop,
                    "attention/learning_health": "frozen" if avg_entropy > 5.5 else "learning"
                }, step=epoch)
        
        logger.info(f"📊 TRAIN SUMMARY | RMSE: {avg_loss:.4f} | Dice: {epoch_metrics['dice']:.4f} | "
                   f"Contrast: {epoch_metrics['contrast_ratio']:.4f} | Abs: {epoch_metrics['rmse_absorption']:.4f} | "
                   f"Scat: {epoch_metrics['rmse_scattering']:.4f}")
        
        # ChatGPT diagnostic summary for early epochs
        if epoch < 10:
            # Check for key problems ChatGPT identified
            problems = []
            if attention_entropies:
                avg_entropy = sum(attention_entropies.values()) / len(attention_entropies)
                if avg_entropy > 5.4:
                    problems.append("frozen attention")
            
            # Check if RMSE is improving
            if hasattr(self, '_last_rmse') and avg_loss >= self._last_rmse:
                problems.append("stalled RMSE")
            self._last_rmse = avg_loss
            
            if problems:
                logger.warning(f"🚨 Training issues detected: {', '.join(problems)}")
                logger.warning(f"   Recommended: Check gradients, LR schedule, weight decay groups")
            else:
                logger.info(f"✅ Training health: Good progress detected")
        
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
            'dice': 0.0, 'contrast_ratio': 0.0, 'rmse_overall': 0.0,
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
                # Use same dtype as training for consistency
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with autocast(dtype=dtype):
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
                
                # Show validation batch progress with standardized format including transformer metrics
                logger.info(f"🔍 VALID | Batch {batch_idx + 1:2d}/{len(data_loader):2d} | "
                           f"RMSE: {loss.item():.4f} | Dice: {batch_metrics.get('dice', 0):.4f} | "
                           f"Contrast: {batch_metrics.get('contrast_ratio', 0):.4f} | "
                           f"Enhancement: {batch_metrics.get('feature_enhancement_ratio', 0):.4f} | "
                           f"Attention: {batch_metrics.get('attention_entropy', 0):.4f}")
        
        avg_loss = total_loss / num_batches
        
        # Average metrics across epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        logger.debug(f"✅ Stage 2 validation completed. Average loss: {avg_loss:.6f}")
        logger.info(f"📊 VALID SUMMARY | RMSE: {avg_loss:.4f} | Dice: {epoch_metrics['dice']:.4f} | "
                   f"Contrast: {epoch_metrics['contrast_ratio']:.4f} | Abs: {epoch_metrics['rmse_absorption']:.4f} | "
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
                # Log comprehensive metrics in organized format with clean epoch x-axis
                wandb.log({
                    "epoch": epoch + 1,  # Custom x-axis for epoch metrics (consistent with Stage1)
                    
                    # === PRIMARY METRICS (most important) ===
                    "Metrics/RMSE_Overall_Train": train_loss,
                    "Metrics/RMSE_Overall_Valid": val_loss,
                    "Metrics/Dice_Train": train_metrics['dice'],
                    "Metrics/Dice_Valid": val_metrics['dice'],
                    "Metrics/ContrastRatio_Train": train_metrics['contrast_ratio'],
                    "Metrics/ContrastRatio_Valid": val_metrics['contrast_ratio'],
                    
                    # === DETAILED RMSE BREAKDOWN ===
                    "RMSE_Details/Absorption_Train": train_metrics['rmse_absorption'],
                    "RMSE_Details/Absorption_Valid": val_metrics['rmse_absorption'],
                    "RMSE_Details/Scattering_Train": train_metrics['rmse_scattering'],
                    "RMSE_Details/Scattering_Valid": val_metrics['rmse_scattering'],
                    
                    # === TRANSFORMER SPECIFIC METRICS ===
                    "Transformer/Feature_Enhancement_Train": train_metrics['feature_enhancement_ratio'],
                    "Transformer/Feature_Enhancement_Valid": val_metrics['feature_enhancement_ratio'],
                    "Transformer/Attention_Entropy_Train": train_metrics['attention_entropy'],
                    "Transformer/Attention_Entropy_Valid": val_metrics['attention_entropy'],
                    
                    # === ENHANCED OVERFITTING ANALYSIS ===
                    "Analysis/Overfitting_Ratio": train_loss / val_loss if val_loss > 0 else 1.0,
                    "Analysis/Dice_Gap_Train_Val": train_metrics['dice'] - val_metrics['dice'],
                    "Analysis/Contrast_Gap_Train_Val": train_metrics['contrast_ratio'] - val_metrics['contrast_ratio'],
                    "Analysis/Overfitting_Risk": "High" if (train_loss / val_loss < 0.85 if val_loss > 0 else False) else "Low",
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
            
            # Print epoch summary with clear visual formatting
            if epoch % PROGRESS_LOG_INTERVAL == 0 or epoch == epochs - FINAL_EPOCH_OFFSET:
                logger.info(f"")
                logger.info(f"{'='*80}")
                logger.info(f"🚀 EPOCH {epoch+1:3d}/{epochs} SUMMARY")
                logger.info(f"{'='*80}")
                logger.info(f"📈 Train RMSE: {train_loss:.4f} | Valid RMSE: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"📊 Train Dice: {train_metrics['dice']:.4f} | Valid Dice: {val_metrics['dice']:.4f}")
                logger.info(f"📊 Train Contrast: {train_metrics['contrast_ratio']:.4f} | Valid Contrast: {val_metrics['contrast_ratio']:.4f}")
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
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
            else:
                logger.debug(f"📊 Stage 2 no improvement. Current: {val_loss:.6f}, Best: {best_val_loss:.6f}")
        
        mode = "Enhanced" if self.use_tissue_patches else "Baseline"
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"✅ TRANSFORMER TRAINING COMPLETED ({mode})")
        logger.info(f"🏆 Best RMSE Loss: {best_val_loss:.4f}")
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
