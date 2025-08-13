#!/usr/bin/env python3
"""
Spatially-Aware Embedding Block for NIR-DOT Reconstruction.

This module implements a sophisticated spatially-aware embedding architecture that
processes measurement and position information separately, then combines them with
measurement-specific tissue context for enhanced spatial modeling.

Key Features:
- Spatially-aware embedding for measurement/position processing
- Measurement-specific tissue patch integration (our innovation)
- Learned fusion of measurement and tissue information
- Maintains 1:1 correspondence between measurements and tissue context

Architecture Flow:
1. Spatially-Aware Embedding: measurements + positions → hi_tokens [256D]
2. Tissue Processing: source_patch + detector_patch → tissue_features [256D] 
3. Learned Fusion: hi_tokens + tissue_features → enhanced_tokens [256D]

Classes:
    SpatiallyAwareEmbedding: Measurement/position embedding
    TissueFeatureExtractor: CNN encoder for tissue patches with learned fusion
    SpatiallyAwareEncoderBlock: Complete encoder combining both components

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

from code.utils.logging_config import get_model_logger

# =============================================================================
# CONSTANTS
# =============================================================================

# Input dimensions
MEASUREMENT_DIM = 2                     # [log_amplitude, phase] - xi
POSITION_DIM = 6                        # [src_x, src_y, src_z, det_x, det_y, det_z] - pi
NIR_INPUT_DIM = 8                       # Total: xi + pi

# Embedding dimensions - Standard transformer architecture
EMBED_DIM = 256                         # Target embedding dimension
INTERMEDIATE_DIM = 128                  # Intermediate dimension for initial measurement processing

# Tissue processing
TISSUE_PATCH_SIZE = 16                  # 16x16x16 tissue patches
TISSUE_CHANNELS = 2                     # [μ_a, μ_s]
TISSUE_FEATURE_DIM = 128                # Features from each patch (source/detector)
TISSUE_COMBINED_DIM = 256               # Combined tissue features (128 + 128)

# Fusion configuration
FUSION_HIDDEN_DIM = 384                 # Hidden dimension for learned fusion

# Weight initialization
WEIGHT_INIT_STD = 0.02

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# SPATIALLY-AWARE EMBEDDING BLOCK
# =============================================================================

class SpatiallyAwareEmbedding(nn.Module):
    """
    Spatially-Aware Embedding Block for NIR-DOT measurements.
    
    This implements a proven approach for processing measurement and position data:
    1. Measurement vector Xi → FC layer with d_embed nodes
    2. Output concatenated with position vector Pi  
    3. Concatenated result → Another FC layer with d_embed nodes
    4. Result: Single token of d_embed dimensions
    
    This maintains d_embed throughout, providing consistent embedding dimensions.
    
    Args:
        embed_dim (int): Target embedding dimension
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        
        logger.info(f"🏗️  Initializing SpatiallyAwareEmbedding: embed_dim={embed_dim}")
        
        self.embed_dim = embed_dim
        
        # Step 1: Measurement vector Xi → FC layer with d_embed nodes (first FC)
        self.measurement_embedding = nn.Sequential(
            nn.Linear(MEASUREMENT_DIM, embed_dim),  # 2D → embed_dim
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Step 2: Concatenated [measurement_embed + position] → FC layer with d_embed nodes (second FC)
        concat_dim = embed_dim + POSITION_DIM  # embed_dim + 6D positions
        self.combined_projection = nn.Sequential(
            nn.Linear(concat_dim, embed_dim),       # (embed_dim + 6) → embed_dim
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ SpatiallyAwareEmbedding initialized with {self.count_parameters()} parameters")
    
    def count_parameters(self):
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        """Initialize weights following standard practices."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=WEIGHT_INIT_STD)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, nir_measurements: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatially-aware embedding.
        
        Architecture flow:
        1. Xi (measurements) → embedding layer → hi'
        2. Concatenate [hi', Pi] → combined vector  
        3. Combined vector → final FC layer → hi (d_embed)
        
        Args:
            nir_measurements (torch.Tensor): Shape [batch_size, seq_len, 8]
                Contains [log_amp, phase, src_x, src_y, src_z, det_x, det_y, det_z]
        
        Returns:
            torch.Tensor: Embedded tokens of shape [batch_size, seq_len, embed_dim]
        """
        logger.debug(f"🏃 SpatiallyAwareEmbedding forward: input {nir_measurements.shape}")
        
        batch_size, n_measurements, feature_dim = nir_measurements.shape
        assert feature_dim == NIR_INPUT_DIM, f"Expected {NIR_INPUT_DIM}D input, got {feature_dim}D"
        
        # Split into measurements (xi) and positions (pi)
        measurements = nir_measurements[:, :, :MEASUREMENT_DIM]  # [batch, n_meas, 2] - xi
        positions = nir_measurements[:, :, MEASUREMENT_DIM:]     # [batch, n_meas, 6] - pi
        
        logger.debug(f"📦 Measurements (xi): {measurements.shape}")
        logger.debug(f"📦 Positions (pi): {positions.shape}")
        
                # Step 1: Process measurements through first FC layer
        measurement_features = self.measurement_embedding(measurements)  # [batch, seq, embed_dim]
        logger.debug(f"Measurement embedding output: {measurement_features.shape}")
        
        # Step 2: Concatenate embedded measurements with raw positions
        measurement_position_concat = torch.cat([measurement_features, positions], dim=-1)  # [batch, seq, embed_dim + 6]
        logger.debug(f"Concatenated features: {measurement_position_concat.shape}")
        
        # Step 3: Pass through final FC layer to get d_embed output
        hi_tokens = self.combined_projection(measurement_position_concat)  # [batch, n_meas, embed_dim]
        
        # Apply layer normalization for stability
        hi_tokens = self.layer_norm(hi_tokens)
        
        logger.debug(f"📦 Final hi tokens: {hi_tokens.shape}")
        
        # Validation: Check for NaN values
        if torch.isnan(hi_tokens).any():
            logger.error("🚨 NaN detected in spatially-aware embedding output")
            raise ValueError("NaN detected in spatially-aware embedding")
        
        return hi_tokens


# =============================================================================
# TISSUE FEATURE EXTRACTOR
# =============================================================================

class TissueFeatureExtractor(nn.Module):
    """
    CNN Encoder for measurement-specific tissue patch processing with learned fusion.
    
    This module processes tissue patches from source and detector locations for each
    measurement, then fuses them with measurement tokens using learned projection.
    Each measurement gets its own specific tissue context, maintaining 1:1 correspondence.
    
    Architecture:
    1. Source patch → CNN → source_features [128D]
    2. Detector patch → CNN → detector_features [128D] 
    3. Combined tissue_features = concat([source, detector]) [256D]
    4. Fusion: hi_token [256D] + tissue_features [256D] → enhanced_token [256D]
    
    Args:
        patch_size (int): Size of tissue patches (16x16x16)
        feature_dim (int): Output dimension for each patch (128D)
        embed_dim (int): Embedding dimension for fusion (256D)
        dropout (float): Dropout probability
    """
    
    def __init__(self, patch_size: int = TISSUE_PATCH_SIZE, 
                 feature_dim: int = TISSUE_FEATURE_DIM,
                 embed_dim: int = EMBED_DIM,
                 dropout: float = 0.15):
        super().__init__()
        
        logger.info(f"🏗️  Initializing TissueFeatureExtractor: patch_size={patch_size}, "
                   f"feature_dim={feature_dim}, embed_dim={embed_dim}")
        
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # Shared CNN encoder for both source and detector patches
        # Input: [batch, 2, 16, 16, 16] where 2 = [μ_a, μ_s] channels
        self.patch_cnn = nn.Sequential(
            # Stage 1: 16³ → 8³
            nn.Conv3d(TISSUE_CHANNELS, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 16³ → 8³
            nn.Dropout3d(0.1),
            
            # Stage 2: 8³ → 4³ 
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8³ → 4³
            nn.Dropout3d(0.1),
            
            # Stage 3: 4³ → 2³
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 4³ → 2³
            nn.Dropout3d(0.15),
            
            # Global pooling and projection
            nn.AdaptiveAvgPool3d(1),  # 2³ → 1³
            nn.Flatten(),             # [128]
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, feature_dim)  # → [128D]
        )
        
        # Learned fusion layer for combining hi_tokens with tissue features
        # Input: concat([hi_token, tissue_features]) = [256D + 256D] = [512D]
        # Output: enhanced_token [256D]
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim + TISSUE_COMBINED_DIM, FUSION_HIDDEN_DIM),  # [512D → 384D]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(FUSION_HIDDEN_DIM, embed_dim),  # [384D → 256D]
            nn.LayerNorm(embed_dim)  # Stabilize output
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ TissueFeatureExtractor initialized with {self.count_parameters()} parameters")
    
    def count_parameters(self):
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        """Initialize weights using standard practices."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.normal_(m.weight, std=WEIGHT_INIT_STD)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def process_tissue_patches(self, tissue_patches: torch.Tensor) -> torch.Tensor:
        """
        Process tissue patches to extract features for each measurement.
        
        Args:
            tissue_patches (torch.Tensor): Shape [batch, n_measurements, 2_patches, 2_channels, 16, 16, 16]
                Direct spatial format from optimized data loader
        
        Returns:
            torch.Tensor: Tissue features of shape [batch, n_measurements, 256]
        """
        batch_size, n_measurements, n_patches, n_channels, d, h, w = tissue_patches.shape
        
        # Validate input shape
        assert n_patches == 2, f"Expected 2 patches (source+detector), got {n_patches}"
        assert n_channels == 2, f"Expected 2 channels (μ_a+μ_s), got {n_channels}"
        assert d == h == w == self.patch_size, f"Expected {self.patch_size}³ patches, got {d}×{h}×{w}"
        
        logger.debug(f"🏃 Processing tissue patches: {tissue_patches.shape}")
        
        # Process each measurement's patches separately to maintain correspondence
        measurement_tissue_features = []
        
        for i in range(n_measurements):
            # Get patches for measurement i: [batch, 2_patches, 2_channels, 16, 16, 16]
            measurement_patches = tissue_patches[:, i, :, :, :, :, :]
            
            # Process source patch: [batch, 2_channels, 16, 16, 16]
            source_patch = measurement_patches[:, 0, :, :, :, :]
            source_features = self.patch_cnn(source_patch)  # [batch, 128]
            
            # Process detector patch: [batch, 2_channels, 16, 16, 16]  
            detector_patch = measurement_patches[:, 1, :, :, :, :]
            detector_features = self.patch_cnn(detector_patch)  # [batch, 128]
            
            # Combine source + detector features for measurement i
            combined_features = torch.cat([source_features, detector_features], dim=-1)  # [batch, 256]
            measurement_tissue_features.append(combined_features)
        
        # Stack all measurement features: [batch, n_measurements, 256]
        tissue_features = torch.stack(measurement_tissue_features, dim=1)
        
        logger.debug(f"📦 Tissue features output: {tissue_features.shape}")
        return tissue_features
    
    def fuse_with_measurements(self, hi_tokens: torch.Tensor, 
                              tissue_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse measurement tokens with their corresponding tissue features using learned fusion.
        
        Args:
            hi_tokens (torch.Tensor): Shape [batch, n_measurements, 256]
            tissue_features (torch.Tensor): Shape [batch, n_measurements, 256]
        
        Returns:
            torch.Tensor: Enhanced tokens of shape [batch, n_measurements, 256]
        """
        logger.debug(f"🏃 Fusing hi_tokens {hi_tokens.shape} with tissue_features {tissue_features.shape}")
        
        # Concatenate measurement and tissue features for each measurement
        combined_features = torch.cat([hi_tokens, tissue_features], dim=-1)  # [batch, n_meas, 512]
        
        # Apply learned fusion
        enhanced_tokens = self.fusion_layer(combined_features)  # [batch, n_meas, 256]
        
        logger.debug(f"📦 Enhanced tokens output: {enhanced_tokens.shape}")
        
        # Validation: Check for NaN values
        if torch.isnan(enhanced_tokens).any():
            logger.error("🚨 NaN detected in tissue fusion output")
            raise ValueError("NaN detected in tissue fusion")
        
        return enhanced_tokens


# =============================================================================
# ENHANCED SPATIALLY-AWARE ENCODER BLOCK
# =============================================================================

class SpatiallyAwareEncoderBlock(nn.Module):
    """
    Complete spatially-aware encoder block with measurement-specific tissue fusion.
    
    This is the full embedding block that creates transformer-ready tokens from:
    - NIR measurement data (amplitude, phase) → hi_tokens via spatially-aware approach
    - Spatial coordinates (source/detector positions) → embedded in hi_tokens
    - Tissue patch information → fused with hi_tokens for enhanced mode
    
    The architecture maintains 1:1 correspondence between measurements and tissue patches,
    ensuring each measurement token gets enhanced with its specific tissue context.
    
    Mode behavior:
    - Baseline: Returns hi_tokens [batch, n_measurements, 256] 
    - Enhanced: Returns tissue-enhanced tokens [batch, n_measurements, 256]
    """
    
    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        
        logger.info("🏗️  Initializing SpatiallyAwareEncoderBlock")
        
        # Spatially-aware embedding for measurements and positions
        self.spatially_aware_embedding = SpatiallyAwareEmbedding(embed_dim=embed_dim, dropout=dropout)
        
        # Tissue feature extractor for enhanced mode  
        self.tissue_feature_extractor = TissueFeatureExtractor(
            feature_dim=TISSUE_FEATURE_DIM,
            embed_dim=embed_dim, 
            dropout=dropout
        )
        
        logger.info(f"✅ SpatiallyAwareEncoderBlock initialized")
    
    def forward(self, nir_measurements: torch.Tensor, 
                tissue_patches: Optional[torch.Tensor] = None,
                use_tissue_patches: bool = False) -> torch.Tensor:
        """
        Forward pass through enhanced spatially-aware encoder block.
        
        Args:
            nir_measurements (torch.Tensor): Shape [batch_size, n_measurements, 8]
                Contains [amplitude, phase, source_x, source_y, source_z, detector_x, detector_y, detector_z]
            tissue_patches (torch.Tensor, optional): Shape [batch_size, n_measurements, 2, patch_volume*2]
                Contains tissue properties for source and detector patches
            use_tissue_patches (bool): Whether to use enhanced mode with tissue fusion
        
        Returns:
            torch.Tensor: Spatially-aware tokens for transformer processing
                Baseline: [batch_size, n_measurements, embed_dim] - hi_tokens only
                Enhanced: [batch_size, n_measurements, embed_dim] - tissue-enhanced tokens
        """
        logger.debug(f"🏃 SpatiallyAwareEncoderBlock forward: nir_measurements {nir_measurements.shape}")
        
        # Always create hi tokens from measurements and positions
        hi_tokens = self.spatially_aware_embedding(nir_measurements)  # [batch, n_meas, embed_dim]
        
        if use_tissue_patches and tissue_patches is not None:
            logger.debug("📦 Enhanced mode: fusing tissue features with hi_tokens")
            
            # Extract tissue features for each measurement
            tissue_features = self.tissue_feature_extractor.process_tissue_patches(tissue_patches)
            # tissue_features: [batch, n_measurements, 256]
            
            # Fuse hi_tokens with measurement-specific tissue features
            enhanced_tokens = self.tissue_feature_extractor.fuse_with_measurements(hi_tokens, tissue_features)
            # enhanced_tokens: [batch, n_measurements, 256]
            
            logger.debug(f"📦 Enhanced tokens output: {enhanced_tokens.shape}")
            
            # Validation: Ensure output shape is correct
            assert enhanced_tokens.shape == hi_tokens.shape, \
                f"Enhanced tokens shape {enhanced_tokens.shape} != hi_tokens shape {hi_tokens.shape}"
            
            return enhanced_tokens
        else:
            logger.debug("📦 Baseline mode: hi_tokens only")
            logger.debug(f"📦 Hi tokens output: {hi_tokens.shape}")
            return hi_tokens
