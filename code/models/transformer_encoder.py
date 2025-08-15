#!/usr/bin/env python3
"""
Transformer Encoder for NIR-DOT Sequence Modeling.

This module implements a transformer-based encoder for processing sequential data
in near-infrared diffuse optical tomography (NIR-DOT) applications. The transformer
uses multi-head self-attention and positional encoding to capture long-range 
dependencies and temporal patterns in the data.

The encoder is designed for stage 2 training in a two-stage hybrid approach,
focusing on sequence modeling and contextual understanding.

Classes:
    PositionalEncoding: Sinusoidal positional encoding for sequences
    MultiHeadAttention: Multi-head self-attention mechanism
    TransformerLayer: Single transformer encoder layer
    TransformerEncoder: Complete transformer encoder stack

Features:
    - Multi-head self-attention with configurable heads
    - Sinusoidal positional encoding
    - Layer normalization and residual connections
    - Configurable MLP expansion ratio
    - Support for attention mask and visualization

Author: Max Hart
Date: July 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import sys
import warnings
from typing import Optional, Tuple, Dict, List, Union
import math

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project imports  
from code.utils.logging_config import get_model_logger

# =============================================================================
# HYPERPARAMETERS AND CONSTANTS
# =============================================================================

# Model Architecture Parameters (OPTIMIZED FOR CHECKPOINT COMPATIBILITY)
EMBED_DIM = 256                         # Transformer embedding dimension
NUM_LAYERS = 6                          # Number of transformer layers (keep compatible with Stage 1 checkpoint)
NUM_HEADS = 4                           # Number of attention heads (optimal head_dim=64 for low-SNR physics)
MLP_RATIO = 3                           # MLP expansion ratio (keep compatible with Stage 1 checkpoint)
DROPOUT = 0.05                          # Dropout probability (reduced since not overfitting)
MAX_SEQ_LEN = 512                       # Maximum sequence length (reduced from 1000 for memory efficiency)
                                        # Robin's approach: pad when dscan < dseq, truncate when dscan > dseq

# Positional Encoding Configuration
POSITIONAL_ENCODING_MAX_LEN = 5000      # Maximum sequence length for positional encoding
POSITIONAL_ENCODING_BASE = 10000.0      # Base for positional encoding calculation

# Attention Mechanism Configuration
ATTENTION_SCALE_BASE = 1.0              # Base for attention scaling (divided by sqrt(head_dim))

# Weight Initialization Configuration
WEIGHT_INIT_STD = 0.02                  # Standard deviation for weight initialization
POSITIONAL_EMBEDDING_INIT_STD = 0.02    # Standard deviation for positional embedding

# Token Type Configuration
NUM_TOKEN_TYPES = 2                     # Number of token types (CNN, tissue)
CNN_TOKEN_TYPE = 0                      # Token type ID for CNN features
TISSUE_TOKEN_TYPE = 1                   # Token type ID for tissue features

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# TRANSFORMER COMPONENTS
# =============================================================================


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer sequences.
    
    Adds positional information to input embeddings using sine and cosine functions
    of different frequencies. This allows the transformer to understand the relative
    positions of elements in the sequence without using positional parameters.
    
    Args:
        embed_dim (int): Embedding dimension (must be even)
        max_len (int, optional): Maximum sequence length. Defaults to 5000.
    """
    
    def __init__(self, embed_dim: int, max_len: int = POSITIONAL_ENCODING_MAX_LEN):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(POSITIONAL_ENCODING_BASE) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (seq_len, batch_size, embed_dim)
            
        Returns:
            torch.Tensor: Position-encoded embeddings of same shape
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism for transformers.
    
    Implements the core attention mechanism that allows the model to focus on
    different parts of the input sequence simultaneously. Multiple attention heads
    enable the model to capture various types of relationships and dependencies.
    
    Args:
        embed_dim (int): Embedding dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = DROPOUT):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = ATTENTION_SCALE_BASE / math.sqrt(self.head_dim)  # Scaling factor for dot-product attention
        
        # Store attention weights for entropy calculation (ChatGPT diagnostic)
        self.last_attention_weights = None
        
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
            
        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided (use smaller value for half-precision compatibility)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Compute attention probabilities and apply dropout
        attention_weights = F.softmax(scores, dim=-1)
        
        # Store attention weights for entropy calculation (ChatGPT diagnostic)
        self.last_attention_weights = attention_weights.detach()
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape and project to output dimensions
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attended)
        
        return output, attention_weights


# =============================================================================
# TRANSFORMER LAYERS
# =============================================================================


class TransformerLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward network.
    
    Implements a standard transformer encoder layer consisting of multi-head
    self-attention followed by a position-wise feed-forward network. Both
    sub-layers are wrapped with residual connections and layer normalization.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads  
        mlp_ratio (int, optional): Ratio for MLP hidden dimension. Defaults to 4.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = MLP_RATIO, 
                 dropout: float = DROPOUT):
        super().__init__()
        # Multi-head self-attention mechanism
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization for pre-norm architecture
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Position-wise feed-forward network (MLP)
        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x, mask=None):
        """
        Forward pass through transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Output tensor of shape (batch_size, seq_len, embed_dim)
                - Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Self-attention with residual connection and pre-norm
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward network with residual connection and pre-norm
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x, attn_weights


# ============= TRANSFORMER ENCODER ================

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequence modeling in NIR-DOT reconstruction.
    
    Multi-layer transformer encoder that processes CNN-extracted features with
    optional tissue context information. Uses self-attention to capture long-range
    dependencies and contextual relationships in the data sequence.
    
    Designed for stage 2 training in a multi-stage learning approach, focusing
    on sequence modeling and high-level feature integration.
    
    Args:
        cnn_feature_dim (int, optional): Dimension of CNN features. Defaults to 512.
        tissue_context_dim (int, optional): Dimension of tissue context. Defaults to 256.
        embed_dim (int, optional): Transformer embedding dimension. Defaults to 768.
        num_layers (int, optional): Number of transformer layers. Defaults to 6.
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_ratio (int, optional): MLP expansion ratio. Defaults to 4.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 1000.
    """
    
    def __init__(self, 
                 cnn_feature_dim: int,  # Must be provided by caller (from CNN module)
                 tissue_context_dim: int = 0,  # Default 0 means no tissue context
                 embed_dim: int = EMBED_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_heads: int = NUM_HEADS,
                 mlp_ratio: int = MLP_RATIO,
                 dropout: float = DROPOUT,
                 max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        
        logger.info(f"🏗️  Initializing Transformer Encoder: {num_layers} layers, "
                   f"{num_heads} heads, embed_dim={embed_dim}")
        
        self.cnn_feature_dim = cnn_feature_dim
        self.tissue_context_dim = tissue_context_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len  # Store for sequence length management
        
        # Memory estimation for attention matrices
        attention_memory_mb = (64 * num_heads * max_seq_len * max_seq_len * 4) / (1024 * 1024)
        logger.info(f"📊 Estimated attention memory usage: {attention_memory_mb:.1f} MB per batch")
        
        # Linear projections to map input features to embedding space
        self.cnn_projection = nn.Linear(cnn_feature_dim, embed_dim)
        
        # Conditional tissue projection - only create if tissue_context_dim > 0
        if tissue_context_dim > 0:
            self.tissue_projection = nn.Linear(tissue_context_dim, embed_dim)
        else:
            self.tissue_projection = None
        
        # Token type embeddings to distinguish different input modalities
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, embed_dim)  # 0: CNN, 1: tissue
        
        # Note: No positional encoding needed - spatial relationships handled by NIR processor
        
        # Stack of transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection to reconstruct feature space
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, cnn_feature_dim),  # Project back to CNN feature space
            nn.ReLU()
        )
        
        # Initialize network weights
        self._init_weights()
        
        # Log model characteristics with architectural details
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"📊 Transformer Encoder initialized:")
        logger.info(f"   ├─ Architecture: {num_layers} layers × {num_heads} heads")
        logger.info(f"   ├─ Embedding: {embed_dim}D (head_dim: {embed_dim//num_heads})")
        logger.info(f"   ├─ Total params: {total_params:,}")
        logger.info(f"   └─ All trainable: {trainable_params:,}")
    
    def _init_weights(self):
        """
        Initialize network weights using transformer-optimized initialization.
        
        Uses scaled initialization for large transformers to prevent gradient explosion
        and ensure stable initial training. Based on GPT-2 and modern transformer practices.
        """
        linear_count = 0
        norm_count = 0
        embed_count = 0
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization for output projections to prevent gradient explosion
                if 'out_proj' in name or 'output_projection' in name:
                    # Output projections: smaller std for stability
                    nn.init.normal_(m.weight, 0, 0.02 / (2 * self.num_layers) ** 0.5)
                else:
                    # Regular linear layers: standard transformer initialization
                    nn.init.normal_(m.weight, 0, 0.02)
                linear_count += 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                norm_count += 1
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, WEIGHT_INIT_STD)
                embed_count += 1
        
        logger.debug(f"🔧 Initialized transformer weights: {linear_count} linear, {norm_count} norm, {embed_count} embed layers")
        
    
    def forward(self, cnn_features: torch.Tensor, 
                tissue_context: Optional[torch.Tensor] = None,
                use_tissue_patches: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer encoder.
        
        Processes CNN features with optional tissue context through the transformer
        architecture. The model can operate in two modes: CNN-only or CNN+tissue context.
        
        Args:
            cnn_features (torch.Tensor): CNN encoded features of shape (batch_size, cnn_feature_dim)
            tissue_context (torch.Tensor, optional): Tissue context features of shape 
                (batch_size, tissue_context_dim). Defaults to None.
            use_tissue_patches (bool, optional): Toggle for including tissue context. 
                Defaults to False.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Enhanced features of shape (batch_size, cnn_feature_dim)
                - Attention weights from the last layer (optional)
        """
        batch_size = cnn_features.shape[0]
        device = cnn_features.device
        
        # Project CNN features to transformer embedding space
        cnn_embedded = self.cnn_projection(cnn_features).unsqueeze(1)  # [B, 1, embed_dim]
        
        # Build input token sequence
        if use_tissue_patches and tissue_context is not None and self.tissue_projection is not None:
            # Project tissue context to embedding dimension
            tissue_embedded = self.tissue_projection(tissue_context).unsqueeze(1)  # [B, 1, embed_dim]
            
            # Concatenate CNN and tissue features into sequence
            token_sequence = torch.cat([cnn_embedded, tissue_embedded], dim=1)  # [B, 2, embed_dim]
            
            # Add token type embeddings to distinguish modalities
            token_types = torch.tensor([0, 1], device=device).unsqueeze(0).expand(batch_size, -1)
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
            
        else:
            # CNN features only mode
            token_sequence = cnn_embedded  # [B, 1, embed_dim]
            
            # Add token type embedding for CNN-only mode
            token_types = torch.tensor([0], device=device).unsqueeze(0).expand(batch_size, -1)
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
        
        # Process through transformer layers
        attention_weights_list = []
        x = token_sequence
        
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)
        
        # Apply final layer normalization
        x = self.layer_norm(x)
        
        # Extract enhanced CNN features (always the first token in sequence)
        enhanced_cnn_features = x[:, 0, :]  # [B, embed_dim]
        
        # Project enhanced features back to original CNN feature space
        enhanced_features = self.output_projection(enhanced_cnn_features)  # [B, cnn_feature_dim]
        
        # Combine attention weights from all layers for analysis
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=1)  # [B, L, H, S, S]
        else:
            attention_weights = None
        
        return enhanced_features, attention_weights
    
    def forward_sequence(self, measurement_features: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None,
                        tissue_context: Optional[torch.Tensor] = None,
                        use_tissue_patches: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        SIMPLIFIED forward pass for fixed 256-measurement sequences (Option C).
        
        Processes exactly 256 NIR measurement features as tokens, eliminating all
        padding/truncation complexity and attention masking overhead.
        
        This simplified approach:
        1. Expects exactly 256 measurements (no padding/truncation needed)
        2. Projects all measurements to embedding space in one batch operation
        3. Adds positional embeddings for all 256 positions
        4. Processes through transformer layers without attention masking
        5. Returns raw transformer output for global pooling
        
        Args:
            measurement_features (torch.Tensor): Measurement features of shape 
                (batch_size, 256, cnn_feature_dim) - exactly 256 measurements
            attention_mask (torch.Tensor, optional): Ignored in this implementation
            tissue_context (torch.Tensor, optional): Tissue context features. Defaults to None.
            use_tissue_patches (bool, optional): Toggle for tissue context inclusion. 
                Defaults to False.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Raw transformer output of shape (batch_size, 256, embed_dim)
                - Attention weights from all layers (optional)
        """
        batch_size, n_measurements, feature_dim = measurement_features.shape
        device = measurement_features.device
        
        # Verify we have exactly 256 measurements (as expected from fixed undersampling)
        assert n_measurements == 256, f"Expected exactly 256 measurements, got {n_measurements}"
        
        # PROJECT ALL MEASUREMENTS: [B, 256, feature_dim] -> [B, 256, embed_dim]
        token_sequence = self.cnn_projection(measurement_features)
        
        # Add tissue context if available
        if use_tissue_patches and tissue_context is not None and self.tissue_projection is not None:
            # Project tissue context to embedding dimension
            tissue_embedded = self.tissue_projection(tissue_context).unsqueeze(1)  # [B, 1, embed_dim]
            
            # Concatenate measurements and tissue features
            token_sequence = torch.cat([token_sequence, tissue_embedded], dim=1)  # [B, 257, embed_dim]
            
            # Add token type embeddings
            token_types = torch.cat([
                torch.zeros(256, device=device, dtype=torch.long),  # Measurement tokens
                torch.ones(1, device=device, dtype=torch.long)     # Tissue token
            ]).unsqueeze(0).expand(batch_size, -1)
            
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
        else:
            # Add token type embeddings for measurement-only mode (all zeros for measurements)
            token_types = torch.zeros(256, device=device, dtype=torch.long)
            token_types = token_types.unsqueeze(0).expand(batch_size, -1)
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
        
        # PROCESS THROUGH TRANSFORMER LAYERS (no attention masking needed)
        attention_weights_list = []
        x = token_sequence
        
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, mask=None)  # No masking for fixed sequences
            attention_weights_list.append(attn_weights)
        
        # Apply final layer normalization
        x = self.layer_norm(x)
        
        # Combine attention weights from all layers for analysis
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=1)  # [B, L, H, S, S]
        else:
            attention_weights = None
        
        return x, attention_weights
    
    def get_attention_maps(self, cnn_features: torch.Tensor,
                          tissue_context: Optional[torch.Tensor] = None,
                          use_tissue_patches: bool = False) -> Optional[torch.Tensor]:
        """
        Extract attention maps for visualization and analysis.
        
        Provides access to the attention weights from all transformer layers,
        useful for understanding what the model is focusing on during processing.
        
        Args:
            cnn_features (torch.Tensor): CNN encoded features of shape (batch_size, cnn_feature_dim)
            tissue_context (torch.Tensor, optional): Tissue context features. Defaults to None.
            use_tissue_patches (bool, optional): Toggle for tissue context inclusion. 
                Defaults to False.
        
        Returns:
            torch.Tensor: Attention weights from the last layer, or None if unavailable
        """
        _, attention_weights = self.forward(cnn_features, tissue_context, use_tissue_patches)
        return attention_weights
