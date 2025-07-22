"""
Tissue Context Encoder for handling tissue patches with toggle functionality.
Encodes tissue context information when use_tissue_patches=True.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TissueContextEncoder(nn.Module):
    """
    Encoder for tissue context patches with toggle functionality.
    Processes tissue patches to provide contextual information for reconstruction.
    """
    
    def __init__(self, 
                 patch_size: int = 7,  # Changed from 16 to match actual data
                 num_patches: int = 2,  # Changed from 8 to match actual data (source + detector)
                 embed_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # Patch embedding - convert flattened patches to embedding vectors
        # Each patch is already flattened to patch_size^3 * 2
        patch_volume = (patch_size ** 3) * 2  # 2 channels (mu_a, mu_s)
        self.patch_embedding = nn.Linear(patch_volume, embed_dim)
        
        # Positional embeddings for patches
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer layers for inter-patch attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection to create global tissue context
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim * num_patches, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
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
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
    
    def forward(self, tissue_patches: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Forward pass through tissue context encoder.
        
        Args:
            tissue_patches: Tensor of shape [batch_size, num_patches, patch_size^3 * 2]
                           or None if use_tissue_patches=False
        
        Returns:
            Tissue context tensor of shape [batch_size, embed_dim] or None
        """
        if tissue_patches is None:
            return None
        
        batch_size = tissue_patches.shape[0]
        
        # tissue_patches should already be [B, num_patches, patch_size^3 * 2]
        # No need to reshape if it's already correctly formatted
        if len(tissue_patches.shape) != 3:
            raise ValueError(f"Expected tissue_patches shape [B, N, D], got {tissue_patches.shape}")
        
        # Embed patches
        patch_embeddings = self.patch_embedding(tissue_patches)  # [B, N, embed_dim]
        
        # Add positional embeddings
        patch_embeddings = patch_embeddings + self.position_embedding
        
        # Apply transformer layers for inter-patch attention
        transformed = self.transformer(patch_embeddings)  # [B, N, embed_dim]
        
        # Apply layer normalization
        transformed = self.layer_norm(transformed)
        
        # Create global context by flattening and projecting
        global_context = transformed.view(batch_size, -1)  # [B, N * embed_dim]
        
        # Add skip connection and projection to desired output dimension
        tissue_context = self.context_projection(global_context)  # [B, embed_dim]
        
        return tissue_context
    
    def get_attention_weights(self, tissue_patches: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization purposes.
        
        Args:
            tissue_patches: Tensor of shape [batch_size, num_patches, patch_size^3]
        
        Returns:
            Attention weights from the last transformer layer
        """
        if tissue_patches is None:
            return None
        
        batch_size = tissue_patches.shape[0]
        
        # Flatten patches if needed
        if len(tissue_patches.shape) == 5:
            tissue_patches = tissue_patches.view(batch_size, self.num_patches, -1)
        
        # Embed patches
        patch_embeddings = self.patch_embedding(tissue_patches)
        patch_embeddings = patch_embeddings + self.position_embedding
        
        # Get attention weights from the transformer
        # Note: This requires modification to get attention weights
        # For now, return dummy weights
        attention_weights = torch.ones(batch_size, self.num_patches, self.num_patches)
        return attention_weights


class TissueContextToggle:
    """
    Utility class for managing tissue context toggle functionality.
    Provides clean interface for enabling/disabling tissue patches.
    """
    
    @staticmethod
    def process_tissue_patches(tissue_patches: Optional[torch.Tensor], 
                              use_tissue_patches: bool) -> Optional[torch.Tensor]:
        """
        Process tissue patches based on toggle setting.
        
        Args:
            tissue_patches: Raw tissue patches tensor or None
            use_tissue_patches: Boolean toggle for tissue patch usage
        
        Returns:
            Processed tissue patches or None based on toggle
        """
        if not use_tissue_patches:
            return None
        return tissue_patches
    
    @staticmethod
    def create_dummy_context(batch_size: int, embed_dim: int, 
                           device: torch.device) -> torch.Tensor:
        """
        Create dummy tissue context when tissue patches are disabled.
        
        Args:
            batch_size: Batch size
            embed_dim: Embedding dimension
            device: Device to create tensor on
        
        Returns:
            Zero tensor of appropriate shape
        """
        return torch.zeros(batch_size, embed_dim, device=device)
    
    @staticmethod
    def merge_contexts(cnn_features: torch.Tensor, 
                      tissue_context: Optional[torch.Tensor],
                      use_tissue_patches: bool) -> torch.Tensor:
        """
        Merge CNN features with tissue context based on toggle.
        
        Args:
            cnn_features: CNN encoded features
            tissue_context: Tissue context features or None
            use_tissue_patches: Boolean toggle
        
        Returns:
            Merged features
        """
        if not use_tissue_patches or tissue_context is None:
            return cnn_features
        
        # Concatenate CNN features with tissue context
        return torch.cat([cnn_features, tissue_context], dim=1)
