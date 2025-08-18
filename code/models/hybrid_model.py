#!/usr/bin/env python3
"""
Hybrid CNN-Transformer Model for NIR-DOT Reconstruction.

This module implements the complete hybrid architecture that combines 
convolutional neural networks (CNN) with transformer encoders for 
near-infrared diffuse optical tomography (NIR-DOT) volume reconstruction.

The hybrid approach uses a two-stage learning strategy:
1. Stage 1: CNN autoencoder pre-training for spatial feature learning
2. Stage 2: Transformer integration for sequence modeling and context enhancement

The model includes optional tissue context integration for improved
reconstruction accuracy through anatomical constraints, and adaptive
sequence undersampling for massive data augmentation through dynamic masking.

Classes:
    HybridCNNTransformer: Complete hybrid model combining CNN and Transformer components

Features:
    - Two-stage training pipeline support
    - Adaptive sequence undersampling (5-256 variable measurements)
    - Optional tissue context integration
    - Configurable architecture parameters
    - Stage-specific parameter freezing
    - Comprehensive logging and monitoring

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
from typing import Optional, Tuple, Dict, Any

# Third-party imports
import torch
import torch.nn as nn

# Project imports - Clean absolute imports from project root
from code.models.cnn_autoencoder import CNNAutoEncoder
from code.models.transformer_encoder import TransformerEncoder
from .spatially_aware_embedding import SpatiallyAwareEmbedding, TissueFeatureExtractor, SpatiallyAwareEncoderBlock
from code.models.global_pooling_encoder import GlobalPoolingEncoder
from code.utils.logging_config import get_model_logger

# Import configuration constants from component modules
from code.models import cnn_autoencoder as cnn_config
from code.models import transformer_encoder as transformer_config

# =============================================================================
# HYPERPARAMETERS AND CONSTANTS
# =============================================================================

# NIR Measurement Configuration
NIR_INPUT_DIM = 8                       # 8D NIR feature vectors (log_amp, phase, source_xyz, det_xyz)
N_GENERATED_MEASUREMENTS = 1000         # Number of measurements generated per phantom (50 sources × 20 detectors)

# Dynamic Undersampling Configuration (massive data augmentation through variable measurements)
MIN_MEASUREMENTS_TRAINING = 5           # Minimum measurements per batch item
MAX_MEASUREMENTS_TRAINING = 256         # Maximum measurements per batch item
DEFAULT_MEASUREMENTS_INFERENCE = 256    # Default for inference when not training

# Tissue patch configuration (used by NIR processor)
TISSUE_PATCH_SIZE = 16                  # Size of tissue patches (enhanced from 11 for better context)
TISSUE_NUM_PATCHES = 2                  # Number of patches (source + detector)
TISSUE_OUTPUT_DIM = 8                   # Total tissue context dimension (2 patches × 4D each)

# Model Behavior Configuration
USE_TISSUE_PATCHES = True               # Whether to use tissue context by default
TRAINING_STAGE = "stage1"               # Default training stage

# Training Stage Identifiers
STAGE1 = "stage1"                       # CNN autoencoder pre-training
STAGE2 = "stage2"                       # Transformer training stage

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# DYNAMIC SEQUENCE UNDERSAMPLING
# =============================================================================

def fixed_sequence_undersampling(nir_measurements: torch.Tensor, 
                                tissue_patches: Optional[torch.Tensor] = None,
                                n_measurements: int = 256,
                                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Fixed-length sequence undersampling with epoch-consistent indexing.
    
    Each epoch uses the same random subset of 256 measurements for ALL phantoms,
    providing excellent data augmentation while maintaining tensor simplicity.
    Eliminates all attention masking complexity and shape mismatch issues.
    
    Args:
        nir_measurements (torch.Tensor): Full NIR measurements [batch, 1000, 8]
        tissue_patches (torch.Tensor, optional): Full tissue patches [batch, 1000, 2, 2, 16, 16, 16]
        n_measurements (int): Fixed number of measurements to select (default: 256)
        training (bool): Whether in training mode
        
    Returns:
        Tuple containing:
        - Selected NIR measurements [batch, 256, 8]
        - Selected tissue patches [batch, 256, 2, 2, 16, 16, 16] (if provided)
    """
    batch_size, seq_len, feature_dim = nir_measurements.shape
    device = nir_measurements.device
    
    if not training:
        # Validation/Inference: use deterministic subset for consistency  
        # Use a fixed random seed to ensure same subset across validation runs
        torch.manual_seed(42)
        selected_indices = torch.randperm(seq_len, device=device)[:n_measurements]
        selected_indices = selected_indices.sort()[0]  # Sort for consistent ordering
        logger.debug(f"🎯 Fixed undersampling (validation): deterministic subset of {n_measurements} measurements")
    else:
        # Training: random selection of 256 measurements (same for all phantoms in batch)
        selected_indices = torch.randperm(seq_len, device=device)[:n_measurements]
        selected_indices = selected_indices.sort()[0]  # Sort for consistent ordering
        logger.debug(f"🎯 Fixed undersampling (training): random subset of {n_measurements} measurements")
    
    # Select measurements using advanced indexing
    selected_nir = nir_measurements[:, selected_indices, :]  # [batch, 256, 8]
    
    # Select tissue patches if provided
    selected_tissue = None
    if tissue_patches is not None:
        selected_tissue = tissue_patches[:, selected_indices, :, :, :, :, :]  # [batch, 256, 2, 2, 16, 16, 16]
    
    return selected_nir, selected_tissue


# =============================================================================
# HYBRID MODEL ARCHITECTURE
# =============================================================================


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture for advanced NIR-DOT reconstruction.
    
    Combines the spatial feature extraction capabilities of CNNs with the
    sequence modeling power of transformers to achieve superior reconstruction
    quality. The model supports a two-stage training paradigm, adaptive
    sequence undersampling for massive data augmentation, and optional
    tissue context integration.
    
    Architecture Components:
    - CNN Autoencoder: Spatial feature extraction and reconstruction
    - Transformer Encoder: Sequence modeling and context integration  
    - Tissue Context Encoder: Anatomical constraint processing (optional)
    - Adaptive Undersampling: Variable sequence lengths (5-256 measurements)
    
    Training Stages:
    1. Stage 1: CNN autoencoder pre-training for low-level spatial features
    2. Stage 2: Transformer training with frozen decoder and adaptive undersampling
    
    Key Features:
    - Adaptive sequence undersampling: 5-256 variable measurements
    - Massive data augmentation through measurement subset variation
    - Efficient validation mode (256 measurements for consistency)
    - Optional tissue patch integration for enhanced reconstruction
    - Stage-specific component freezing for optimal training
    """
    
    def __init__(self,
                 # CNN autoencoder configuration
                 input_channels: int = cnn_config.INPUT_CHANNELS,  # Both absorption and scattering coefficients
                 output_size: Tuple[int, int, int] = cnn_config.OUTPUT_SIZE,  # Match your data dimensions
                 cnn_base_channels: int = cnn_config.BASE_CHANNELS,
                 
                 # NIR measurement configuration
                 nir_input_dim: int = NIR_INPUT_DIM,  # 8D NIR feature vectors (log_amp, phase, source_xyz, det_xyz)
                 
                 # Tissue context encoder configuration
                 patch_size: int = TISSUE_PATCH_SIZE,  # Tissue patch size matching data format
                 num_patches: int = TISSUE_NUM_PATCHES,  # Source + detector regions
                 tissue_output_dim: int = TISSUE_OUTPUT_DIM,  # 8D total output (2 patches × 4D each)
                 
                 # Transformer encoder configuration
                 transformer_embed_dim: int = transformer_config.EMBED_DIM,
                 transformer_num_layers: int = transformer_config.NUM_LAYERS,
                 transformer_num_heads: int = transformer_config.NUM_HEADS,
                 mlp_ratio: int = transformer_config.MLP_RATIO,
                 dropout: float = transformer_config.DROPOUT,
                 cnn_dropout: float = 0.1,  # CNN autoencoder dropout
                 nir_dropout: float = 0.15,  # NIR processor dropout
                 
                 # Model behavior configuration
                 use_tissue_patches: bool = USE_TISSUE_PATCHES,
                 training_stage: str = TRAINING_STAGE):
        
        super().__init__()
        
        logger.info(f"🏗️  Initializing Hybrid CNN-Transformer: {training_stage} mode, "
                   f"tissue_patches={'enabled' if use_tissue_patches else 'disabled'}")
        
        # Store model configuration
        self.use_tissue_patches = use_tissue_patches
        self.training_stage = training_stage
        self.output_size = output_size
        self.nir_input_dim = nir_input_dim  # Store NIR dimension
        
        # Initialize CNN Autoencoder (used in both stages)
        self.cnn_autoencoder = CNNAutoEncoder(
            input_channels=input_channels,
            output_size=output_size,
            base_channels=cnn_base_channels,
            dropout_rate=cnn_dropout
        )
        
        # Initialize Spatially-Aware Encoder Block (replaces NIR processor)
        self.spatially_aware_encoder = SpatiallyAwareEncoderBlock(
            embed_dim=256,  # Transformer embedding dimension
            dropout=nir_dropout
        )
        
        # Initialize Global Pooling Encoder (post-transformer processing)
        self.global_pooling_encoder = GlobalPoolingEncoder(
            embed_dim=256,  # Match transformer output dimension
            encoded_scan_dim=cnn_config.FEATURE_DIM,  # Match CNN autoencoder feature dimension
            dropout=dropout
        )
        
        # Initialize Optimized Transformer Encoder (stage 2 component)
        self.transformer_encoder = TransformerEncoder(
            cnn_feature_dim=256,  # Now receives 256D tokens from spatially-aware encoder
            tissue_context_dim=0,  # Tissue context now handled in spatially-aware encoder
            embed_dim=256,  # Match transformer embedding dimension
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Range Calibrator - channel-wise affine calibration
        self.range_calibrator = nn.Conv3d(2, 2, kernel_size=1, bias=True)
        # Initialize to identity transformation
        nn.init.eye_(self.range_calibrator.weight.view(2, 2))
        nn.init.zeros_(self.range_calibrator.bias)
        
        # Initialize network weights
        self._init_weights()
        
        # Log detailed model characteristics
        self._log_detailed_parameter_breakdown()
    
    def _log_detailed_parameter_breakdown(self):
        """
        Log detailed parameter breakdown by component and stage-specific usage.
        """
        # Count parameters by component
        cnn_total = sum(p.numel() for p in self.cnn_autoencoder.parameters())
        cnn_encoder = sum(p.numel() for p in self.cnn_autoencoder.encoder.parameters())
        cnn_decoder = sum(p.numel() for p in self.cnn_autoencoder.decoder.parameters())
        
        embedding_total = sum(p.numel() for p in self.spatially_aware_encoder.parameters())
        transformer_total = sum(p.numel() for p in self.transformer_encoder.parameters())
        pooling_total = sum(p.numel() for p in self.global_pooling_encoder.parameters())
        calibrator_total = sum(p.numel() for p in self.range_calibrator.parameters())
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("📊 DETAILED PARAMETER BREAKDOWN:")
        logger.info("   ┌─────────────────────────────────────────────────────────────┐")
        logger.info(f"   │ CNN Autoencoder:        {cnn_total:>8,} params             │")
        logger.info(f"   │   ├─ Encoder:           {cnn_encoder:>8,} params             │")
        logger.info(f"   │   └─ Decoder:           {cnn_decoder:>8,} params             │")
        logger.info(f"   │ Spatially-Aware Embed: {embedding_total:>8,} params             │")
        logger.info(f"   │ Transformer Encoder:   {transformer_total:>8,} params             │")
        logger.info(f"   │ Global Pooling (Attn): {pooling_total:>8,} params             │")
        logger.info(f"   │ Range Calibrator:      {calibrator_total:>8,} params             │")
        logger.info("   ├─────────────────────────────────────────────────────────────┤")
        logger.info(f"   │ TOTAL MODEL:           {total_params:>8,} params             │")
        logger.info(f"   │ TRAINABLE:             {trainable_params:>8,} params             │")
        logger.info("   └─────────────────────────────────────────────────────────────┘")
        
        # Enhanced feature logging
        logger.info("🚀 ENHANCED FEATURES:")
        logger.info(f"   ├─ Attention Pooling: {'✅ Active' if True else '❌ Disabled'}")
        logger.info(f"   ├─ Range Calibrator: {'✅ Active' if True else '❌ Disabled'}")
        logger.info(f"   ├─ Decoder: {'🔒 Frozen (Stage 1 features preserved)' if self.training_stage == 'stage2' else '✅ Active'}")
        logger.info(f"   └─ EMA Training: {'✅ Active' if hasattr(self, 'ema_enabled') else '⚙️  External Config'}")
        
        # Stage-specific parameter usage
        if self.training_stage == 'stage1':
            stage1_active = cnn_total + calibrator_total
            logger.info("🎯 STAGE 1 PARAMETER USAGE:")
            logger.info(f"   ├─ Active: CNN Autoencoder ({cnn_total:,} params)")
            logger.info(f"   ├─ Active: Range Calibrator ({calibrator_total:,} params)")
            logger.info(f"   └─ Unused: Transformer pipeline ({total_params - stage1_active:,} params)")
        
        elif self.training_stage == 'stage2':
            stage2_active = cnn_decoder + embedding_total + transformer_total + pooling_total + calibrator_total
            logger.info("🎯 STAGE 2 PARAMETER USAGE:")
            logger.info(f"   ├─ Active: CNN Decoder ({cnn_decoder:,} params)")
            logger.info(f"   ├─ Active: Embedding ({embedding_total:,} params)")
            logger.info(f"   ├─ Active: Transformer ({transformer_total:,} params)")
            logger.info(f"   ├─ Active: Pooling (Attn) ({pooling_total:,} params)")
            logger.info(f"   ├─ Active: Range Calibrator ({calibrator_total:,} params)")
            logger.info(f"   └─ TOTAL ACTIVE: {stage2_active:,} params")
            logger.info(f"   └─ Discarded: CNN Encoder ({cnn_encoder:,} params)")
    
    def _init_weights(self):
        """
        Initialize weights for transformer and tissue encoder components.
        
        The CNN autoencoder initializes its own weights. This method handles
        additional components that may need specific initialization.
        """
        # Initialize any custom projection layers or additional components
        # CNN autoencoder and transformer/tissue encoders handle their own initialization
    
    def forward(self, dot_measurements: torch.Tensor,
                tissue_patches: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model following the ECBO 2025 architecture.
        
        Architecture Flow:
        Stage 1: ground_truth → CNN autoencoder → reconstruction
        Stage 2: nir_measurements → spatially-aware encoder → transformer → global pooling → CNN decoder
        
        Args:
            dot_measurements (torch.Tensor): 
                Stage 1: Ground truth volumes of shape (batch_size, 2, 64, 64, 64)
                Stage 2: NIR measurements of shape (batch_size, n_measurements, 8)
            tissue_patches (torch.Tensor, optional): Tissue patches of shape
                (batch_size, n_measurements, 2, patch_volume*2). Defaults to None.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model outputs
        """
        batch_size = dot_measurements.shape[0]
        device = dot_measurements.device
        
        outputs = {}
        
        # Determine input type
        is_nir_measurements = (len(dot_measurements.shape) == 3 and 
                             dot_measurements.shape[2] == self.nir_input_dim)
        is_ground_truth = (len(dot_measurements.shape) == 5 or 
                          (len(dot_measurements.shape) == 4 and dot_measurements.shape[1] == 2))
        
        if self.training_stage == STAGE1:
            if not is_ground_truth:
                raise ValueError(
                    f"Stage 1 training requires ground truth volumes, got shape {dot_measurements.shape}. "
                    f"Expected shape: (batch_size, 2, 64, 64, 64)"
                )
            
            reconstructed = self.cnn_autoencoder(dot_measurements)
            # Apply range calibrator
            reconstructed = self.range_calibrator(reconstructed)
            
            outputs.update({
                'reconstructed': reconstructed,
                'stage': STAGE1
            })
            
        elif self.training_stage == STAGE2:
            if not is_nir_measurements:
                raise ValueError(
                    f"Stage 2 training requires NIR measurements, got shape {dot_measurements.shape}. "
                    f"Expected shape: (batch_size, n_measurements, 8)"
                )
            
            # Step 0: Fixed Sequence Undersampling (256 measurements per phantom)
            # Eliminates all attention masking complexity and shape mismatches
            undersampled_nir, undersampled_tissue = fixed_sequence_undersampling(
                nir_measurements=dot_measurements,     # [batch, 1000, 8] 
                tissue_patches=tissue_patches,         # [batch, 1000, 2, 2, 16, 16, 16] or None
                n_measurements=256,                    # Fixed: always 256 measurements
                training=self.training
            )
            
            # Now all tensors have consistent shape: [batch, 256, ...]
            # No attention masking needed - all measurements are "active"
            
            # Step 1: Spatially-Aware Encoder Block
            combined_tokens = self.spatially_aware_encoder(
                nir_measurements=undersampled_nir,     # [batch, 256, 8]
                tissue_patches=undersampled_tissue,    # [batch, 256, 2, 2, 16, 16, 16] or None
                use_tissue_patches=self.use_tissue_patches
            )  # Returns: [batch, 256, embed_dim]
            
            # Step 2: Transformer processing (NO attention masking needed!)
            enhanced_tokens, attention_weights = self.transformer_encoder.forward_sequence(
                measurement_features=combined_tokens,  # [batch, 256, embed_dim]
                attention_mask=None,                   # No masking needed!
                tissue_context=None,
                use_tissue_patches=False
            )  # Returns: [batch, 256, embed_dim], attention_weights
            
            # Step 3: Global pooling (simple averaging - no masking needed)
            encoded_scan = self.global_pooling_encoder(enhanced_tokens)
            
            # Step 4: CNN decoder (using pre-trained weights from Stage 1)
            reconstructed = self.cnn_autoencoder.decode(encoded_scan)  # [batch, 2, 64, 64, 64]
            # Apply range calibrator
            reconstructed = self.range_calibrator(reconstructed)
            
            # Prepare features for metrics (aggregate tokens to single vectors)
            enhanced_features_for_metrics = enhanced_tokens.mean(dim=1)  # [batch, 256, embed_dim] -> [batch, embed_dim]
            cnn_features_for_metrics = combined_tokens.mean(dim=1)       # [batch, 256, embed_dim] -> [batch, embed_dim]
            
            outputs.update({
                'reconstructed': reconstructed,
                'encoded_scan': encoded_scan,
                'enhanced_features': enhanced_features_for_metrics,   # [batch, embed_dim] for metrics
                'cnn_features': cnn_features_for_metrics,             # [batch, embed_dim] for metrics
                'attention_weights': attention_weights,
                'selected_measurements': 256,  # Always 256 measurements
                'original_measurements': dot_measurements.shape[1],  # Original number of measurements (1000)
                'stage': STAGE2
            })
        
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")
        
        return outputs
    
    def set_training_stage(self, stage: str):
        """
        Configure model for specific training stage.
        
        Sets the appropriate training/evaluation modes and parameter freezing
        for the two-stage training approach. Stage 1 trains only the CNN
        autoencoder, while Stage 2 freezes the decoder and trains the transformer.
        
        Args:
            stage (str): Training stage - either "stage1" or "stage2"
            
        Raises:
            ValueError: If stage is not "stage1" or "stage2"
        """
        if stage not in [STAGE1, STAGE2]:
            raise ValueError(f"Invalid stage: {stage}. Must be '{STAGE1}' or '{STAGE2}'")
        
        self.training_stage = stage
        logger.debug(f"🎯 Setting training stage to: {stage}")
        
        if stage == STAGE1:
            # Stage 1: Train CNN autoencoder end-to-end
            self.cnn_autoencoder.train()
            self.transformer_encoder.eval()
            
        elif stage == STAGE2:
            # Stage 2: Freeze CNN decoder, train transformer components
            self.cnn_autoencoder.encoder.eval()
            self.cnn_autoencoder.decoder.eval()
            
            # Freeze CNN decoder parameters
            for param in self.cnn_autoencoder.decoder.parameters():
                param.requires_grad = False
            
            # Enable training for transformer and spatially-aware encoder
            self.transformer_encoder.train()
            self.spatially_aware_encoder.train()
                
            logger.debug("🔒 CNN decoder frozen, transformer components enabled for training")
    
    def toggle_tissue_patches(self, use_tissue_patches: bool):
        """
        Toggle tissue patch usage for experimental control.
        
        Enables or disables tissue context processing for A/B testing and
        ablation studies. When disabled, the model operates in CNN+Transformer
        mode without tissue context.
        
        Args:
            use_tissue_patches (bool): Whether to use tissue context
        """
        self.use_tissue_patches = use_tissue_patches
        logger.info(f"🔄 Tissue patches {'enabled' if use_tissue_patches else 'disabled'}")
    
    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get currently trainable parameters based on training stage.
        
        Returns different parameter sets depending on the current training stage
        to enable stage-specific optimization and parameter counting.
        
        Returns:
            Dict[str, torch.nn.Parameter]: Dictionary of trainable parameters
                organized by component name
        """
        trainable_params = {}
        
        if self.training_stage == STAGE1:
            # Stage 1: Only CNN autoencoder parameters are trainable
            for name, param in self.cnn_autoencoder.named_parameters():
                if param.requires_grad:
                    trainable_params[f"cnn_autoencoder.{name}"] = param
                    
        elif self.training_stage == STAGE2:
            # Stage 2: Transformer and spatially-aware encoder parameters are trainable
            for name, param in self.transformer_encoder.named_parameters():
                if param.requires_grad:
                    trainable_params[f"transformer_encoder.{name}"] = param
            
            # Spatially-aware encoder contains tissue encoding internally
            for name, param in self.spatially_aware_encoder.named_parameters():
                if param.requires_grad:
                    trainable_params[f"spatially_aware_encoder.{name}"] = param
        
        return trainable_params
    
    def load_stage1_weights(self, checkpoint_path: str):
        """
        Load pre-trained CNN autoencoder weights from stage 1.
        
        Loads the CNN autoencoder weights from a stage 1 checkpoint to initialize
        the model for stage 2 training. Essential for the two-stage training approach.
        
        Args:
            checkpoint_path (str): Path to the stage 1 checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"📂 Loading stage 1 weights from: {checkpoint_path}")
        
        # Extract CNN autoencoder state dict from checkpoint
        cnn_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('cnn_autoencoder.'):
                new_key = key[len('cnn_autoencoder.'):]
                cnn_state_dict[new_key] = value
        
        # Load weights into CNN autoencoder
        self.cnn_autoencoder.load_state_dict(cnn_state_dict)
        logger.info("✅ Stage 1 CNN autoencoder weights loaded successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model configuration and statistics.
        
        Provides detailed information about model architecture, parameter counts,
        and current configuration. Useful for logging and model analysis.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information including
                parameter counts, configuration, and component details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        cnn_params = sum(p.numel() for p in self.cnn_autoencoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        spatially_aware_params = sum(p.numel() for p in self.spatially_aware_encoder.parameters())
        
        return {
            'training_stage': self.training_stage,
            'use_tissue_patches': self.use_tissue_patches,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_parameters': cnn_params,
            'transformer_parameters': transformer_params,
            'spatially_aware_parameters': spatially_aware_params,
            'output_size': self.output_size
        }
    
    def encode(self, nir_measurements: torch.Tensor, 
               tissue_patches: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode NIR measurements to latent representation without decoding.
        
        This method performs only the encoding steps of the Stage 2 forward pass:
        spatially-aware encoding, transformer processing, and global pooling.
        It does NOT run the CNN decoder, making it perfect for latent-only training.
        
        Args:
            nir_measurements (torch.Tensor): NIR measurements of shape (batch_size, n_measurements, 8)
            tissue_patches (torch.Tensor, optional): Tissue patches for context
        
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim)
        
        Raises:
            ValueError: If model is not in Stage 2 mode or input shape is invalid
        """
        if self.training_stage != STAGE2:
            raise ValueError(f"encode() method only available in Stage 2, current stage: {self.training_stage}")
        
        # Validate input shape
        if not (len(nir_measurements.shape) == 3 and nir_measurements.shape[2] == self.nir_input_dim):
            raise ValueError(
                f"encode() requires NIR measurements of shape (batch_size, n_measurements, 8), "
                f"got shape {nir_measurements.shape}"
            )
        
        # Step 0: Fixed Sequence Undersampling (256 measurements per phantom)
        undersampled_nir, undersampled_tissue = fixed_sequence_undersampling(
            nir_measurements=nir_measurements,     # [batch, 1000, 8] 
            tissue_patches=tissue_patches,         # [batch, 1000, 2, 2, 16, 16, 16] or None
            n_measurements=256,                    # Fixed: always 256 measurements
            training=self.training
        )
        
        # Step 1: Spatially-Aware Encoder Block
        combined_tokens = self.spatially_aware_encoder(
            nir_measurements=undersampled_nir,     # [batch, 256, 8]
            tissue_patches=undersampled_tissue,    # [batch, 256, 2, 2, 16, 16, 16] or None
            use_tissue_patches=self.use_tissue_patches
        )  # Returns: [batch, 256, embed_dim]
        
        # Step 2: Transformer processing
        enhanced_tokens, _ = self.transformer_encoder.forward_sequence(
            measurement_features=combined_tokens,  # [batch, 256, embed_dim]
            attention_mask=None,                   # No masking needed!
            tissue_context=None,
            use_tissue_patches=False
        )  # Returns: [batch, 256, embed_dim], attention_weights
        
        # Step 3: Global pooling to get latent representation
        encoded_scan = self.global_pooling_encoder(enhanced_tokens)  # [batch, latent_dim]
        
        return encoded_scan
    
    def forward_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass that returns only latent representation for student training.
        
        This method extracts NIR measurements and optional tissue patches from
        the batch and returns the latent representation without decoding.
        
        Args:
            batch: Dictionary containing:
                - 'nir_measurements': [batch_size, n_measurements, 8]
                - 'tissue_patches': [batch_size, n_measurements, 2, 2, 16, 16, 16] (optional)
        
        Returns:
            torch.Tensor: Student latent representation [batch_size, 256]
        """
        nir_measurements = batch['nir_measurements']
        tissue_patches = batch.get('tissue_patches', None)
        
        return self.encode(nir_measurements, tissue_patches)

    def get_calibrator_regularization(self, weight_lambda: float = 1e-5) -> torch.Tensor:
        """
        Compute L2 regularization for the range calibrator.
        
        Regularizes the calibrator to stay close to identity transformation,
        only adjusting when necessary for reconstruction improvement.
        
        Args:
            weight_lambda: Regularization strength
            
        Returns:
            torch.Tensor: L2 regularization loss
        """
        # L2 penalty on deviation from identity
        weight_penalty = torch.norm(self.range_calibrator.weight.view(2, 2) - torch.eye(2, device=self.range_calibrator.weight.device))
        bias_penalty = torch.norm(self.range_calibrator.bias)
        
        return weight_lambda * (weight_penalty + bias_penalty)
