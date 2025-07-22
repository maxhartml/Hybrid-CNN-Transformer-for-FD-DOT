"""
Transformer Encoder for stage 2 training following Robin Dale's approach.
Processes CNN features with optional tissue context for enhanced reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attended)
        
        return output, attention_weights


class TransformerLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for stage 2 training following Robin Dale's approach.
    Processes CNN features with optional tissue context for enhanced reconstruction.
    """
    
    def __init__(self, 
                 cnn_feature_dim: int = 512,
                 tissue_context_dim: int = 256,
                 embed_dim: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 max_seq_len: int = 1000):
        super().__init__()
        
        self.cnn_feature_dim = cnn_feature_dim
        self.tissue_context_dim = tissue_context_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Feature projection layers
        self.cnn_projection = nn.Linear(cnn_feature_dim, embed_dim)
        self.tissue_projection = nn.Linear(tissue_context_dim, embed_dim)
        
        # Token type embeddings to distinguish CNN vs tissue features
        self.token_type_embedding = nn.Embedding(2, embed_dim)  # 0: CNN, 1: tissue
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection for reconstruction features
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, cnn_feature_dim),  # Back to CNN feature space
            nn.ReLU()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, cnn_features: torch.Tensor, 
                tissue_context: Optional[torch.Tensor] = None,
                use_tissue_patches: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer encoder.
        
        Args:
            cnn_features: CNN encoded features [batch_size, cnn_feature_dim]
            tissue_context: Tissue context features [batch_size, tissue_context_dim] or None
            use_tissue_patches: Boolean toggle for tissue patch usage
        
        Returns:
            Tuple of (enhanced_features, attention_weights)
        """
        batch_size = cnn_features.shape[0]
        device = cnn_features.device
        
        # Project CNN features to embedding dimension
        cnn_embedded = self.cnn_projection(cnn_features).unsqueeze(1)  # [B, 1, embed_dim]
        
        # Create token sequence
        if use_tissue_patches and tissue_context is not None:
            # Project tissue context to embedding dimension
            tissue_embedded = self.tissue_projection(tissue_context).unsqueeze(1)  # [B, 1, embed_dim]
            
            # Combine CNN and tissue features
            token_sequence = torch.cat([cnn_embedded, tissue_embedded], dim=1)  # [B, 2, embed_dim]
            
            # Add token type embeddings
            token_types = torch.tensor([0, 1], device=device).unsqueeze(0).expand(batch_size, -1)
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
            
        else:
            # Only CNN features
            token_sequence = cnn_embedded  # [B, 1, embed_dim]
            
            # Add token type embedding for CNN only
            token_types = torch.tensor([0], device=device).unsqueeze(0).expand(batch_size, -1)
            token_type_emb = self.token_type_embedding(token_types)
            token_sequence = token_sequence + token_type_emb
        
        # Add positional encoding
        token_sequence = self.positional_encoding(token_sequence.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer layers
        attention_weights_list = []
        x = token_sequence
        
        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)
        
        # Apply final layer normalization
        x = self.layer_norm(x)
        
        # Extract enhanced CNN features (always first token)
        enhanced_cnn_features = x[:, 0, :]  # [B, embed_dim]
        
        # Project back to CNN feature space
        enhanced_features = self.output_projection(enhanced_cnn_features)  # [B, cnn_feature_dim]
        
        # Stack attention weights for analysis
        attention_weights = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None
        
        return enhanced_features, attention_weights
    
    def get_attention_maps(self, cnn_features: torch.Tensor,
                          tissue_context: Optional[torch.Tensor] = None,
                          use_tissue_patches: bool = False) -> Optional[torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            cnn_features: CNN encoded features
            tissue_context: Tissue context features or None
            use_tissue_patches: Boolean toggle
        
        Returns:
            Attention weights from all layers
        """
        _, attention_weights = self.forward(cnn_features, tissue_context, use_tissue_patches)
        return attention_weights
