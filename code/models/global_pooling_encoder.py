#!/usr/bin/env python3
"""
Global Pooling and Encoded Scan Module for NIR-DOT Reconstruction.

This module implements the post-transformer processing that appears in the ECBO 2025 
architecture after the self-attention transformer block:
- Learnable attention pooling across the sequence dimension
- Fully connected layer to create the "Encoded Scan"
- Enhanced architecture with attention-based aggregation

Classes:
    GlobalPoolingEncoder: Attention pooling and FC layer for encoded scan generation

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
from code.training.training_config import N_MEASUREMENTS, EMBED_DIM, ENCODED_SCAN_DIM

# =============================================================================
# CONSTANTS
# =============================================================================

# Default dimensions - use centralized config values
DEFAULT_EMBED_DIM = EMBED_DIM                    # Input from transformer
DEFAULT_ENCODED_SCAN_DIM = ENCODED_SCAN_DIM     # Output dimension for CNN decoder

# Weight initialization
WEIGHT_INIT_STD = 0.02

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# GLOBAL POOLING ENCODER
# =============================================================================

class GlobalPoolingEncoder(nn.Module):
    """
    Attention-based pooling and encoded scan generation following enhanced ECBO 2025 architecture.
    
    This module implements learnable attention pooling for post-transformer processing:
    - Takes transformer output tokens
    - Applies learnable attention pooling across sequence dimension
    - Projects through FC layer to create "Encoded Scan"
    - Feeds to pre-trained CNN decoder
    
    Architecture flow:
    Transformer → Attention Pooling → FC → Encoded Scan → Pre-trained CNN decoder
    
    Args:
        embed_dim (int): Input dimension from transformer
        encoded_scan_dim (int): Output dimension for CNN decoder
        dropout (float): Dropout probability
    """
    
    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM, 
                 encoded_scan_dim: int = DEFAULT_ENCODED_SCAN_DIM,
                 dropout: float = 0.1):
        super().__init__()
        
        logger.info(f"🏗️  Initializing GlobalPoolingEncoder with Attention Pooling: {embed_dim} → {encoded_scan_dim}")
        
        self.embed_dim = embed_dim
        self.encoded_scan_dim = encoded_scan_dim
        
        # Learnable attention pooling components
        self.attention_query = nn.Parameter(torch.randn(1, 1, embed_dim) * WEIGHT_INIT_STD)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5  # Scaled dot-product attention
        
        # FC layer to create encoded scan
        self.encoded_scan_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, encoded_scan_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ GlobalPoolingEncoder initialized with attention pooling, {self.count_parameters()} parameters")
    
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
        
        # Initialize attention query
        nn.init.normal_(self.attention_query, std=WEIGHT_INIT_STD)
    
    def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with learnable attention pooling.
        
        Applies learnable attention pooling across all N_MEASUREMENTS tokens, allowing
        the model to dynamically weight token importance for reconstruction.
        
        Args:
            transformer_output (torch.Tensor): Shape [batch_size, N_MEASUREMENTS, embed_dim]
                Output from transformer encoder
        
        Returns:
            torch.Tensor: Encoded scan of shape [batch_size, encoded_scan_dim]
                Ready for CNN decoder processing
        """
        batch_size, seq_len, embed_dim = transformer_output.shape
        assert embed_dim == self.embed_dim, f"Expected {self.embed_dim}D input, got {embed_dim}D"
        assert seq_len == N_MEASUREMENTS, f"Expected exactly {N_MEASUREMENTS} tokens, got {seq_len}"
        
        # Learnable attention pooling
        # Query: [1, 1, embed_dim] → [batch_size, 1, embed_dim]
        query = self.attention_query.expand(batch_size, -1, -1)
        
        # Keys: [batch_size, seq_len, embed_dim]
        keys = self.key_projection(transformer_output)
        
        # Attention weights: [batch_size, 1, seq_len]
        attention_scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum: [batch_size, 1, embed_dim] → [batch_size, embed_dim]
        pooled = torch.bmm(attention_weights, transformer_output).squeeze(1)
        
        # Project to encoded scan dimension
        encoded_scan = self.encoded_scan_projection(pooled)  # [batch, encoded_scan_dim]
        
        return encoded_scan
