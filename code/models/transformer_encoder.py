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

# Model Architecture Parameters (OPTIMIZED)
EMBED_DIM = 256                         # Transformer embedding dimension (reduced from 768)
NUM_LAYERS = 4                          # Number of transformer layers (reduced from 6)
NUM_HEADS = 8                           # Number of attention heads (reduced from 12)
MLP_RATIO = 3                           # MLP expansion ratio (reduced from 4)
DROPOUT = 0.1                           # Dropout probability
MAX_SEQ_LEN = 1000                      # Maximum sequence length

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
        
        logger.debug(f"🔧 MultiHeadAttention initialized: embed_dim={embed_dim}, num_heads={num_heads}, head_dim={self.head_dim}")
    
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
        logger.debug(f"🏃 MultiHeadAttention forward: query {query.shape}, key {key.shape}, value {value.shape}")
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        logger.debug(f"📦 After projection and reshaping: q {q.shape}, k {k.shape}, v {v.shape}")
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logger.debug(f"📦 Attention scores shape: {scores.shape}")
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            logger.debug("🎭 Applied attention mask")
        
        # Compute attention probabilities and apply dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        logger.debug(f"📦 Attention weights shape: {attention_weights.shape}")
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        logger.debug(f"📦 Attended values shape: {attended.shape}")
        
        # Reshape and project to output dimensions
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
        logger.debug(f"📦 Reshaped attended shape: {attended.shape}")
        
        output = self.out_proj(attended)
        logger.debug(f"📦 MultiHeadAttention output shape: {output.shape}")
        
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
        
        logger.debug(f"🔧 TransformerLayer initialized: embed_dim={embed_dim}, num_heads={num_heads}, mlp_dim={mlp_dim}")
    
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
        logger.debug(f"🏃 TransformerLayer forward: input shape {x.shape}")
        
        # Self-attention with residual connection and pre-norm
        attn_out, attn_weights = self.attention(x, x, x, mask)
        logger.debug(f"📦 After attention: {attn_out.shape}")
        
        x = self.norm1(x + attn_out)
        logger.debug(f"📦 After norm1 + residual: {x.shape}")
        
        # Feed-forward network with residual connection and pre-norm
        mlp_out = self.mlp(x)
        logger.debug(f"📦 After MLP: {mlp_out.shape}")
        
        x = self.norm2(x + mlp_out)
        logger.debug(f"📦 TransformerLayer output: {x.shape}")
        
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
        
        # Log model characteristics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"📊 Transformer Encoder initialized: {total_params:,} total params, "
                   f"{trainable_params:,} trainable params")
    
    def _init_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        Applies Xavier uniform initialization to linear layers and standard
        initialization to layer normalization and embedding layers.
        """
        logger.debug("🔧 Initializing Transformer Encoder weights...")
        linear_count = 0
        norm_count = 0
        embed_count = 0
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                logger.debug(f"🔧 Xavier uniform init: {name}.weight {m.weight.shape}")
                linear_count += 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    logger.debug(f"🔧 Zero bias init: {name}.bias {m.bias.shape}")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                logger.debug(f"🔧 LayerNorm init: {name} (weight=1, bias=0)")
                norm_count += 1
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, WEIGHT_INIT_STD)
                logger.debug(f"🔧 Normal init: {name}.weight {m.weight.shape}, std={WEIGHT_INIT_STD}")
                embed_count += 1
        
        logger.debug(f"✅ Weight initialization completed: {linear_count} linear layers, {norm_count} norm layers, {embed_count} embeddings")
    
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
        logger.debug(f"🏃 Transformer Encoder forward: cnn_features {cnn_features.shape}, use_tissue_patches={use_tissue_patches}")
        
        # Project CNN features to transformer embedding space
        cnn_embedded = self.cnn_projection(cnn_features).unsqueeze(1)  # [B, 1, embed_dim]
        logger.debug(f"📦 CNN embedded: {cnn_embedded.shape}")
        
        # Build input token sequence
        if use_tissue_patches and tissue_context is not None and self.tissue_projection is not None:
            logger.debug(f"🧬 Including tissue context: {tissue_context.shape}")
            
            # Project tissue context to embedding dimension
            tissue_embedded = self.tissue_projection(tissue_context).unsqueeze(1)  # [B, 1, embed_dim]
            logger.debug(f"📦 Tissue embedded: {tissue_embedded.shape}")
            
            # Concatenate CNN and tissue features into sequence
            token_sequence = torch.cat([cnn_embedded, tissue_embedded], dim=1)  # [B, 2, embed_dim]
            logger.debug(f"📦 Token sequence with tissue: {token_sequence.shape}")
            
            # Add token type embeddings to distinguish modalities
            token_types = torch.tensor([0, 1], device=device).unsqueeze(0).expand(batch_size, -1)
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
            logger.debug(f"📦 After token type embedding: {token_sequence.shape}")
            
        else:
            logger.debug("🔧 CNN features only mode")
            # CNN features only mode
            token_sequence = cnn_embedded  # [B, 1, embed_dim]
            
            # Add token type embedding for CNN-only mode
            token_types = torch.tensor([0], device=device).unsqueeze(0).expand(batch_size, -1)
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
            logger.debug(f"📦 CNN-only token sequence: {token_sequence.shape}")
        
        # Skip positional encoding - spatial relationships already encoded by NIR processor
        logger.debug("⏭️  Skipping positional encoding (spatial info handled by NIR processor)")
        
        # Process through transformer layers
        logger.debug(f"🔄 Processing through {len(self.layers)} transformer layers...")
        attention_weights_list = []
        x = token_sequence
        
        for i, layer in enumerate(self.layers):
            logger.debug(f"🔄 Processing layer {i+1}/{len(self.layers)}...")
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)
            logger.debug(f"📦 Layer {i+1} output: {x.shape}")
        
        # Apply final layer normalization
        logger.debug("🧼 Applying final layer normalization...")
        x = self.layer_norm(x)
        logger.debug(f"📦 After final layer norm: {x.shape}")
        
        # Extract enhanced CNN features (always the first token in sequence)
        enhanced_cnn_features = x[:, 0, :]  # [B, embed_dim]
        logger.debug(f"📦 Enhanced CNN features: {enhanced_cnn_features.shape}")
        
        # Project enhanced features back to original CNN feature space
        enhanced_features = self.output_projection(enhanced_cnn_features)  # [B, cnn_feature_dim]
        logger.debug(f"📦 Final transformer output: {enhanced_features.shape}")
        
        # Combine attention weights from all layers for analysis
        attention_weights = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None
        logger.debug(f"✅ Transformer Encoder forward pass completed")
        
        return enhanced_features, attention_weights
    
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
        logger.debug("🔍 Extracting attention maps from transformer encoder...")
        _, attention_weights = self.forward(cnn_features, tissue_context, use_tissue_patches)
        logger.debug(f"📦 Attention weights extracted: {attention_weights.shape if attention_weights is not None else 'None'}")
        return attention_weights
