#!/usr/bin/env python3
"""
NIR Measurement Processor with Spatial Attention.

This module implements optimized NIR measurement processing using spatial attention
instead of simple mean pooling. The processor         self.enhanced_projection = nn.Sequential(
            nn.Linear(NIR_INPUT_DIM + TISSUE_CONTEXT_DIM, 48),    # 8 NIR + 8 tissue = 16D input
            nn.ReLU(),
            nn.Linear(48, SPATIAL_EMBED_DIM)
        )cts the spatial relationships
between source-detector pairs and can work with or without tissue context.

Classes:
    SpatialAttentionNIRProcessor: Main NIR processor with spatial attention
    PerMeasurementTissueEncoder: Encoder for individual measurement tissue patches

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
from typing import Optional, Dict, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project imports
from code.utils.logging_config import get_model_logger

# =============================================================================
# CONSTANTS
# =============================================================================

# NIR measurement configuration
NIR_INPUT_DIM = 8                       # [log_amp, phase, src_x, src_y, src_z, det_x, det_y, det_z]
POSITION_DIM = 6                        # Source and detector positions (xyz each)
MEASUREMENT_DIM = 2                     # Log amplitude and phase
N_MEASUREMENTS = 256                    # Number of measurements per phantom

# Output dimensions
CNN_FEATURE_DIM = 256                   # Must match CNN autoencoder feature dimension
TISSUE_CONTEXT_DIM = 8                  # Output from tissue encoder (2 patches × 4D each)

# Spatial attention configuration  
SPATIAL_EMBED_DIM = 256                 # Embedding dimension for spatial attention
NUM_ATTENTION_HEADS = 8                 # Number of attention heads
POSITION_ENCODING_DIM = 64              # Dimension for positional encoding

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# NIR MEASUREMENT PROCESSORS
# =============================================================================

class PerMeasurementTissueEncoder(nn.Module):
    """
    Encoder for tissue patches associated with individual measurements.
    
    Processes tissue patches around source and detector locations for each
    measurement, providing local anatomical context that gets appended to
    the NIR measurement features.
    """
    
    def __init__(self, patch_size: int = 16, output_dim: int = 4):
        super().__init__()
        
        logger.info(f"🏗️  Initializing PerMeasurementTissueEncoder: patch_size={patch_size}, output_dim={output_dim}")
        
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # Advanced CNN for 16³ patches (2 tissue property channels)
        # Deep residual architecture with batch normalization and dropout
        self.patch_encoder = nn.Sequential(
            # Stage 1: 16³ → 8³ (initial feature extraction)
            nn.Conv3d(2, 16, kernel_size=3, padding=1, bias=False),   # 2 → 16 channels
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),  # Residual-style depth
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                   # 16³ → 8³
            nn.Dropout3d(0.1),
            
            # Stage 2: 8³ → 4³ (intermediate features)
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False), # 16 → 32 channels
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False), # Residual-style depth
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                   # 8³ → 4³
            nn.Dropout3d(0.1),
            
            # Stage 3: 4³ → 2³ (high-level features)
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False), # 32 → 64 channels
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False), # Residual-style depth
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                   # 4³ → 2³
            nn.Dropout3d(0.15),
            
            # Global feature aggregation
            nn.AdaptiveAvgPool3d(1),                                 # 2³ → 1³ (global pooling)
            nn.Flatten(),                                            # → 64D feature vector
            
            # Feature projection with residual-style MLP
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim)                                # → output_dim per patch
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ PerMeasurementTissueEncoder initialized with ~{self.count_parameters()} parameters")
    
    def count_parameters(self):
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, tissue_patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through per-measurement tissue encoder.
        
        Args:
            tissue_patches (torch.Tensor): Shape [batch, 2, 16^3*2]
                batch_size measurements × 2 patches (source + detector) × flattened patch data
                where flattened patch data contains interleaved channels (μₐ, μₛ) for 16³ voxels
        
        Returns:
            torch.Tensor: Shape [batch, 8] (4D per source + 4D per detector)
        """
        logger.debug(f"🏃 PerMeasurementTissueEncoder forward: input shape {tissue_patches.shape}")
        
        batch_size, n_patches, flattened_size = tissue_patches.shape
        
        # Reshape flattened patches back to 3D: [batch, 2, 16^3*2] → [batch, 2, 2, 16, 16, 16]
        # The flattened data is interleaved: [ch0_vox0, ch1_vox0, ch0_vox1, ch1_vox1, ...]
        patch_volume = self.patch_size ** 3
        n_channels = 2  # absorption + scattering
        
        # Reshape to separate channels and spatial dimensions
        reshaped_patches = tissue_patches.view(batch_size, n_patches, patch_volume, n_channels)
        # Reorder to [batch, 2, 2, 16*16*16] → [batch, 2, 2, 16, 16, 16]
        reshaped_patches = reshaped_patches.permute(0, 1, 3, 2)  # Move channels before spatial
        reshaped_patches = reshaped_patches.view(batch_size, n_patches, n_channels, 
                                                self.patch_size, self.patch_size, self.patch_size)
        
        logger.debug(f"📦 Reshaped patches: {reshaped_patches.shape}")
        
        # Reshape for batch processing: [batch×2, 2, 16, 16, 16]
        patches = reshaped_patches.view(-1, n_channels, self.patch_size, self.patch_size, self.patch_size)
        logger.debug(f"📦 Patches for CNN: {patches.shape}")
        
        # Encode all patches: [batch×2, output_dim]
        encoded_patches = self.patch_encoder(patches)
        logger.debug(f"📦 Encoded patches: {encoded_patches.shape}")
        
        # Reshape back: [batch, 2, output_dim] → [batch, 2*output_dim]
        encoded_patches = encoded_patches.view(batch_size, n_patches, self.output_dim)
        tissue_contexts = encoded_patches.view(batch_size, n_patches * self.output_dim)
        
        logger.debug(f"📦 PerMeasurementTissueEncoder output: {tissue_contexts.shape}")
        return tissue_contexts


class SpatialAttentionNIRProcessor(nn.Module):
    """
    NIR measurement processor with spatial attention aggregation.
    
    Processes NIR measurements using spatial attention to respect the geometric
    relationships between source-detector pairs. Supports both baseline mode
    (NIR only) and enhanced mode (NIR + tissue context).
    """
    
    def __init__(self):
        super().__init__()
        
        logger.info("🏗️  Initializing SpatialAttentionNIRProcessor")
        
        # Tissue encoder for per-measurement context
        self.tissue_encoder = PerMeasurementTissueEncoder()
        
        # Different projections for baseline vs enhanced mode
        self.baseline_projection = nn.Sequential(
            nn.Linear(NIR_INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, SPATIAL_EMBED_DIM)
        )
        
        self.enhanced_projection = nn.Sequential(
            nn.Linear(NIR_INPUT_DIM + TISSUE_CONTEXT_DIM, 48),  # 8 NIR + 8 tissue = 16 total
            nn.ReLU(),
            nn.Linear(48, SPATIAL_EMBED_DIM)
        )
        
        # Positional encoding for source/detector locations
        self.positional_encoder = nn.Linear(POSITION_DIM, POSITION_ENCODING_DIM)
        
        # Spatial attention mechanism
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=SPATIAL_EMBED_DIM, 
            num_heads=NUM_ATTENTION_HEADS, 
            batch_first=True
        )
        
        # Learnable global query for aggregation
        self.global_query = nn.Parameter(torch.randn(1, 1, SPATIAL_EMBED_DIM))
        
        # Final projection to CNN feature space
        self.final_projection = nn.Linear(SPATIAL_EMBED_DIM, CNN_FEATURE_DIM)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ SpatialAttentionNIRProcessor initialized with ~{self.count_parameters()} parameters")
    
    def count_parameters(self):
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize global query
        nn.init.normal_(self.global_query, std=0.02)
    
    def create_positional_encoding(self, nir_measurements: torch.Tensor) -> torch.Tensor:
        """
        Create positional encoding from source/detector coordinates.
        
        Args:
            nir_measurements (torch.Tensor): Shape [batch, 1, 8] or [batch, 8]
                Last 6 dimensions are [src_x, src_y, src_z, det_x, det_y, det_z]
        
        Returns:
            torch.Tensor: Positional encodings [batch, POSITION_ENCODING_DIM] or [batch, 1, POSITION_ENCODING_DIM]
        """
        # Handle both [batch, 8] and [batch, 1, 8] shapes
        if len(nir_measurements.shape) == 2:
            # Extract positions from NIR measurements: [batch, 8] → [batch, 6]
            positions = nir_measurements[:, 2:]  # [batch, 6]
            squeeze_output = True
        else:
            # Extract positions from NIR measurements: [batch, 1, 8] → [batch, 1, 6]
            positions = nir_measurements[:, :, 2:]  # [batch, 1, 6]
            squeeze_output = False
        
        # Encode positions
        if len(positions.shape) == 2:
            pos_encoding = self.positional_encoder(positions)  # [batch, 64]
        else:
            batch_size, seq_len = positions.shape[:2]
            pos_encoding = self.positional_encoder(positions.view(-1, 6))  # [batch*seq_len, 64]
            pos_encoding = pos_encoding.view(batch_size, seq_len, -1)  # [batch, seq_len, 64]
        
        # Pad to match spatial embedding dimension
        if POSITION_ENCODING_DIM < SPATIAL_EMBED_DIM:
            pad_size = SPATIAL_EMBED_DIM - POSITION_ENCODING_DIM
            if len(pos_encoding.shape) == 2:
                pos_encoding = F.pad(pos_encoding, (0, pad_size))  # [batch, 256]
            else:
                pos_encoding = F.pad(pos_encoding, (0, pad_size))  # [batch, seq_len, 256]
        
        return pos_encoding
    
    def forward(self, nir_measurements: torch.Tensor, 
                tissue_patches: Optional[torch.Tensor] = None, 
                use_tissue_patches: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through spatial attention NIR processor.
        
        Args:
            nir_measurements (torch.Tensor): Shape [batch, 8] (individual measurements)
            tissue_patches (torch.Tensor, optional): Shape [batch, 2, 16^3*2] (individual tissue patches)
            use_tissue_patches (bool): Whether to use tissue context
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'features': Aggregated features [batch, CNN_FEATURE_DIM]
                - 'attention_weights': None (no spatial attention for individual measurements)
                - 'projected_measurements': Projected measurements [batch, SPATIAL_EMBED_DIM]
        """
        logger.debug(f"🏃 SpatialAttentionNIRProcessor forward: nir_measurements {nir_measurements.shape}")
        
        batch_size = nir_measurements.shape[0]
        
        if use_tissue_patches and tissue_patches is not None:
            logger.debug("📦 Enhanced mode: processing tissue context")
            # Enhanced mode: append tissue context to each measurement
            tissue_contexts = self.tissue_encoder(tissue_patches)  # [batch, 16]
            enhanced_measurements = torch.cat([nir_measurements, tissue_contexts], dim=1)  # [batch, 24]
            projected = self.enhanced_projection(enhanced_measurements)  # [batch, 256]
        else:
            logger.debug("📦 Baseline mode: NIR measurements only")
            # Baseline mode: NIR measurements only
            projected = self.baseline_projection(nir_measurements)  # [batch, 256]
        
        logger.debug(f"📦 Projected measurements: {projected.shape}")
        
        # Add positional encoding based on source/detector locations
        pos_encoding = self.create_positional_encoding(nir_measurements.unsqueeze(1)).squeeze(1)  # [batch, 256]
        projected_with_pos = projected + pos_encoding  # [batch, 256]
        logger.debug(f"📦 With positional encoding: {projected_with_pos.shape}")
        
        # For individual measurements, no spatial attention needed - just project to final space
        final_features = self.final_projection(projected_with_pos)  # [batch, CNN_FEATURE_DIM]
        logger.debug(f"📦 Final features: {final_features.shape}")
        
        return {
            'features': final_features,
            'attention_weights': None,  # No spatial attention for individual measurements
            'projected_measurements': projected_with_pos
        }
