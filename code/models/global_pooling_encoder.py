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
from code.training.training_config import N_MEASUREMENTS, EMBED_DIM, ENCODED_SCAN_DIM, GLOBAL_POOLING_QUERIES

# =============================================================================
# CONSTANTS
# =============================================================================

# Default dimensions - use centralized config values
DEFAULT_EMBED_DIM = EMBED_DIM                    # Input from transformer
DEFAULT_ENCODED_SCAN_DIM = ENCODED_SCAN_DIM     # Output dimension for CNN decoder

# Multi-query pooling configuration - centralized from training_config
NUM_POOL_QUERIES = GLOBAL_POOLING_QUERIES       # Number of learnable pooling queries for enhanced representation

# Weight initialization
WEIGHT_INIT_STD = 0.02

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# GLOBAL POOLING ENCODER
# =============================================================================

class GlobalPoolingEncoder(nn.Module):
    """
    Multi-query attention-based pooling and encoded scan generation following enhanced ECBO 2025 architecture.
    
    This module implements learnable multi-query attention pooling for post-transformer processing:
    - Takes transformer output tokens
    - Applies multi-query learnable attention pooling across sequence dimension
    - Projects through FC layer to create "Encoded Scan"
    - Feeds to pre-trained CNN decoder
    
    Architecture flow:
    Transformer → Multi-Query Attention Pooling → FC → Encoded Scan → Pre-trained CNN decoder
    
    Args:
        embed_dim (int): Input dimension from transformer
        encoded_scan_dim (int): Output dimension for CNN decoder  
        num_pool_queries (int): Number of learnable pooling queries
        dropout (float): Dropout probability
    """
    
    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM, 
                 encoded_scan_dim: int = DEFAULT_ENCODED_SCAN_DIM,
                 num_pool_queries: int = NUM_POOL_QUERIES,
                 dropout: float = 0.1):
        super().__init__()
        
        logger.info(f"🏗️  Initializing GlobalPoolingEncoder with Multi-Query Attention Pooling: {embed_dim} → {encoded_scan_dim}, {num_pool_queries} queries")
        
        self.embed_dim = embed_dim
        self.encoded_scan_dim = encoded_scan_dim
        self.num_pool_queries = num_pool_queries
        
        # Multi-query learnable attention pooling components
        self.attention_queries = nn.Parameter(torch.randn(1, num_pool_queries, embed_dim) * WEIGHT_INIT_STD)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5  # Scaled dot-product attention
        
        # Fusion layer for multiple query outputs
        self.query_fusion = nn.Linear(num_pool_queries * embed_dim, embed_dim)
        
        # FC layer to create encoded scan
        self.encoded_scan_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, encoded_scan_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ GlobalPoolingEncoder initialized with {num_pool_queries}-query attention pooling, {self.count_parameters()} parameters")
    
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
        
        # Initialize multi-query attention queries
        nn.init.normal_(self.attention_queries, std=WEIGHT_INIT_STD)
    
    def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with multi-query learnable attention pooling.
        
        Applies multi-query learnable attention pooling across all N_MEASUREMENTS tokens, 
        allowing the model to dynamically extract multiple complementary representations
        for enhanced reconstruction capability.
        
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
        
        # Multi-query learnable attention pooling
        # Queries: [1, num_queries, embed_dim] → [batch_size, num_queries, embed_dim]
        queries = self.attention_queries.expand(batch_size, -1, -1)
        
        # Keys and Values: [batch_size, seq_len, embed_dim]
        keys = self.key_projection(transformer_output)
        values = self.value_projection(transformer_output)
        
        # Multi-head attention computation
        # Attention scores: [batch_size, num_queries, seq_len]
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum for each query: [batch_size, num_queries, embed_dim]
        multi_pooled = torch.bmm(attention_weights, values)
        
        # Flatten and fuse multiple query representations: [batch_size, num_queries * embed_dim]
        flattened = multi_pooled.view(batch_size, -1)
        fused = self.query_fusion(flattened)  # [batch_size, embed_dim]
        
        # Project to encoded scan dimension
        encoded_scan = self.encoded_scan_projection(fused)  # [batch, encoded_scan_dim]
        
        return encoded_scan
