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
reconstruction accuracy through anatomical constraints.

Classes:
    HybridCNNTransformer: Complete hybrid model combining CNN and Transformer components

Features:
    - Two-stage training pipeline support
    - Optional tissue context integration
    - Configurable architecture parameters
    - Stage-specific parameter freezing
    - Comprehensive logging and monitoring

Author: Max Hart
Date: July 2025
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
N_MEASUREMENTS = 256                    # Number of measurements for training (subsampled from 1000 generated)
N_GENERATED_MEASUREMENTS = 1000         # Number of measurements generated per phantom (50 sources × 20 detectors)

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
# HYBRID MODEL ARCHITECTURE
# =============================================================================


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture for advanced NIR-DOT reconstruction.
    
    Combines the spatial feature extraction capabilities of CNNs with the
    sequence modeling power of transformers to achieve superior reconstruction
    quality. The model supports a two-stage training paradigm and optional
    tissue context integration.
    
    Architecture Components:
    - CNN Autoencoder: Spatial feature extraction and reconstruction
    - Transformer Encoder: Sequence modeling and context integration  
    - Tissue Context Encoder: Anatomical constraint processing (optional)
    
    Training Stages:
    1. Stage 1: CNN autoencoder pre-training for low-level spatial features
    2. Stage 2: Transformer training with frozen decoder for high-level modeling
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to 1.
        output_size (Tuple[int, int, int], optional): Target output volume size. 
            Defaults to (64, 64, 64).
        base_channels (int, optional): Base CNN channels. Defaults to 64.
        transformer_layers (int, optional): Number of transformer layers. Defaults to 6.
        transformer_heads (int, optional): Number of attention heads. Defaults to 12.
        embed_dim (int, optional): Transformer embedding dimension. Defaults to 768.
        tissue_patch_size (int, optional): Size of tissue patches. Defaults to 16.
        tissue_num_patches (int, optional): Number of tissue patches. Defaults to 2.
        tissue_output_dim (int, optional): Tissue output dimension. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
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
            embed_dim=256,  # Robin's d_embed dimension
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
            embed_dim=256,  # Match Robin's d_embed dimension
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Initialize network weights
        self._init_weights()
        
        # Log model characteristics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"📊 Hybrid Model initialized: {total_params:,} total params, "
                   f"{trainable_params:,} trainable params")
    
    def _init_weights(self):
        """
        Initialize weights for transformer and tissue encoder components.
        
        The CNN autoencoder initializes its own weights. This method handles
        additional components that may need specific initialization.
        """
        # Initialize any custom projection layers or additional components
        # CNN autoencoder and transformer/tissue encoders handle their own initialization
        logger.debug("🔧 Custom weight initialization completed for hybrid model components")
    
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
        logger.debug(f"🏃 Hybrid Model forward: input shape {dot_measurements.shape}, stage={self.training_stage}")
        
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
            
            logger.debug("📍 Stage 1: CNN autoencoder training")
            reconstructed = self.cnn_autoencoder(dot_measurements)
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
            
            logger.debug("� Stage 2: Robindale transformer enhancement")
            
            # Step 1: Spatially-Aware Encoder Block (spatially-aware embedding + tissue fusion)
            combined_tokens = self.spatially_aware_encoder(
                nir_measurements=dot_measurements,    # [batch, n_measurements, 8]
                tissue_patches=tissue_patches,        # [batch, n_measurements, 2, patch_volume*2] or None
                use_tissue_patches=self.use_tissue_patches
            )  # Returns: [batch, total_tokens, embed_dim]
            
            logger.debug(f"📦 Spatially-aware encoder tokens: {combined_tokens.shape}")
            
            # Step 2: Transformer processing (self-attention + feed-forward)
            enhanced_tokens, attention_weights = self.transformer_encoder.forward_sequence(
                measurement_features=combined_tokens,  # [batch, total_tokens, embed_dim]
                tissue_context=None,  # Tissue context already handled in spatially-aware encoder
                use_tissue_patches=False  # Already integrated in tokens
            )  # Returns: [batch, total_tokens, embed_dim]
            
            logger.debug(f"� Transformer enhanced tokens: {enhanced_tokens.shape}")
            
            # Step 3: Global pooling and encoded scan generation
            encoded_scan = self.global_pooling_encoder(enhanced_tokens)  # [batch, encoded_scan_dim]
            
            logger.debug(f"📦 Encoded scan: {encoded_scan.shape}")
            
            # Step 4: CNN decoder (using pre-trained weights from Stage 1)
            reconstructed = self.cnn_autoencoder.decode(encoded_scan)  # [batch, 2, 64, 64, 64]
            
            logger.debug(f"📦 Final reconstruction: {reconstructed.shape}")
            
            outputs.update({
                'reconstructed': reconstructed,
                'encoded_scan': encoded_scan,
                'enhanced_tokens': enhanced_tokens,
                'combined_tokens': combined_tokens,
                'attention_weights': attention_weights,
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
        logger.info(f"🎯 Setting training stage to: {stage}")
        
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
                
            logger.info("🔒 CNN decoder frozen, transformer components enabled for training")
    
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
        
        logger.debug(f"Found {len(trainable_params)} trainable parameters for {self.training_stage}")
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
