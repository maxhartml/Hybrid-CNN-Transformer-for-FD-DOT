#!/usr/bin/env python3
"""
Global Pooling and Encoded Scan Module for NIR-DOT Reconstruction.

This module implements the post-transformer processing that appears in the ECBO 2025 
architecture after the self-attention transformer block:
- Global average pooling across the sequence dimension
- Fully connected layer to create the "Encoded Scan"
- Matches the architecture shown in Figure 1 of the research paper

Classes:
    GlobalPoolingEncoder: Global pooling and FC layer for encoded scan generation

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from code.utils.logging_config import get_model_logger

# =============================================================================
# CONSTANTS
# =============================================================================

# Default dimensions
DEFAULT_EMBED_DIM = 256                 # Input from transformer (Robin's d_embed)
DEFAULT_ENCODED_SCAN_DIM = 256          # Output dimension for CNN decoder

# Weight initialization
WEIGHT_INIT_STD = 0.02

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# GLOBAL POOLING ENCODER
# =============================================================================

class GlobalPoolingEncoder(nn.Module):
    """
    Global pooling and encoded scan generation following the ECBO 2025 architecture.
    
    This module implements the post-transformer processing shown in Figure 1:
    - Takes transformer output tokens
    - Applies global average pooling across sequence dimension
    - Projects through FC layer to create "Encoded Scan"
    - Feeds to pre-trained CNN decoder
    
    This matches the architecture flow:
    Transformer → Global avg → FC → Encoded Scan → Pre-trained CNN decoder
    
    Args:
        embed_dim (int): Input dimension from transformer
        encoded_scan_dim (int): Output dimension for CNN decoder
        dropout (float): Dropout probability
    """
    
    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM, 
                 encoded_scan_dim: int = DEFAULT_ENCODED_SCAN_DIM,
                 dropout: float = 0.1):
        super().__init__()
        
        logger.info(f"🏗️  Initializing GlobalPoolingEncoder: {embed_dim} → {encoded_scan_dim}")
        
        self.embed_dim = embed_dim
        self.encoded_scan_dim = encoded_scan_dim
        
        # Global pooling (average across sequence dimension)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layer to create encoded scan (matching the ECBO 2025 architecture)
        self.encoded_scan_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, encoded_scan_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ GlobalPoolingEncoder initialized with {self.count_parameters()} parameters")
    
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
    
    def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through global pooling encoder.
        
        Args:
            transformer_output (torch.Tensor): Shape [batch_size, seq_len, embed_dim]
                Output from transformer encoder
        
        Returns:
            torch.Tensor: Encoded scan of shape [batch_size, encoded_scan_dim]
                Ready for CNN decoder processing
        """
        logger.debug(f"🏃 GlobalPoolingEncoder forward: input {transformer_output.shape}")
        
        batch_size, seq_len, embed_dim = transformer_output.shape
        assert embed_dim == self.embed_dim, f"Expected {self.embed_dim}D input, got {embed_dim}D"
        
        # Global average pooling across sequence dimension
        # [batch_size, seq_len, embed_dim] → [batch_size, embed_dim, seq_len] → [batch_size, embed_dim, 1]
        pooled = self.global_pool(transformer_output.transpose(1, 2))  # [batch, embed_dim, 1]
        
        # Remove the singleton dimension: [batch_size, embed_dim, 1] → [batch_size, embed_dim]
        pooled = pooled.squeeze(-1)  # [batch, embed_dim]
        
        logger.debug(f"📦 After global pooling: {pooled.shape}")
        
        # Project to encoded scan dimension
        encoded_scan = self.encoded_scan_projection(pooled)  # [batch, encoded_scan_dim]
        
        logger.debug(f"📦 Encoded scan output: {encoded_scan.shape}")
        
        # Validation: Check for NaN values
        if torch.isnan(encoded_scan).any():
            logger.error("🚨 NaN detected in global pooling encoder output")
            raise ValueError("NaN detected in global pooling encoder")
        
        return encoded_scan
