"""
Tissue Context Encoder for NIR-DOT reconstruction enhancement.

This module implements a tissue context encoder that processes anatomical
information to enhance near-infrared diffuse optical tomography (NIR-DOT) 
reconstruction. The encoder generates contextual embeddings from tissue
properties and anatomical constraints.

The tissue context encoder provides auxiliary information to improve
reconstruction accuracy in the hybrid learning approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..utils.logging_config import get_model_logger

# Initialize logger for this module
logger = get_model_logger(__name__)


class TissueContextEncoder(nn.Module):
    """
    Tissue context encoder for anatomical information processing.
    
    Processes tissue property patches to generate contextual embeddings that
    enhance NIR-DOT reconstruction. The encoder handles variable numbers of
    tissue patches (e.g., source and detector regions) and generates compact
    representations of anatomical constraints.
    
    The encoder uses a simplified transformer-like architecture to process
    spatial tissue information and generate context vectors for integration
    with the main reconstruction pipeline.
    
    Args:
        patch_size (int, optional): Size of each tissue patch. Defaults to 7.
        num_patches (int, optional): Number of tissue patches. Defaults to 2.
        embed_dim (int, optional): Embedding dimension. Defaults to 256.
        num_layers (int, optional): Number of encoding layers. Defaults to 3.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        mlp_ratio (int, optional): MLP expansion ratio. Defaults to 4.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self, 
                 patch_size = 7,  # Size matches actual tissue patch data - can be int or tuple
                 num_patches: int = 2,  # Source + detector patches
                 embed_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_tissue_types: int = None):  # Optional backward compatibility
        super().__init__()
        
        # Handle patch_size as either int or tuple
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) == 3:
                patch_volume_base = patch_size[0] * patch_size[1] * patch_size[2]
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple must have 3 dimensions, got {len(patch_size)}")
        else:
            patch_volume_base = patch_size ** 3
            self.patch_size = patch_size
        
        logger.info(f"🏗️  Initializing Tissue Context Encoder: {num_patches} patches, "
                   f"patch_size={patch_size}, embed_dim={embed_dim}")
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # Patch embedding layer - converts flattened tissue patches to embeddings
        # Each patch contains tissue properties (mu_a, mu_s) across patch volume
        patch_volume = patch_volume_base * 2  # 2 tissue property channels
        self.patch_embedding = nn.Linear(patch_volume, embed_dim)
        
        # Learnable positional embeddings for spatial patch relationships
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer encoder for inter-patch attention and context modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Context aggregation and projection network
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim * num_patches, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )
        
        # Initialize network weights
        self._init_weights()
        
        # Log model characteristics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"📊 Tissue Context Encoder initialized: {total_params:,} total params, "
                   f"{trainable_params:,} trainable params")
    
    def _init_weights(self):
        """
        Initialize network weights using appropriate strategies.
        
        Uses Xavier uniform initialization for linear layers, standard
        initialization for layer normalization, and truncated normal
        for positional embeddings.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize positional embeddings with small random values
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
    
    def forward(self, tissue_patches: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Forward pass through tissue context encoder.
        
        Processes tissue property patches through embedding, transformer attention,
        and context aggregation to generate a compact tissue context representation.
        
        Args:
            tissue_patches (torch.Tensor, optional): Tissue patches of shape 
                (batch_size, num_patches, patch_size^3 * 2) where the last dimension
                contains flattened tissue properties (mu_a, mu_s). Can be None.
        
        Returns:
            torch.Tensor: Tissue context embedding of shape (batch_size, embed_dim),
                or None if tissue_patches is None
        """
        if tissue_patches is None:
            return None
        
        batch_size = tissue_patches.shape[0]
        
        # Validate input tensor shape
        if len(tissue_patches.shape) != 3:
            raise ValueError(f"Expected tissue_patches shape [B, N, D], got {tissue_patches.shape}")
        
        # Embed tissue patches to transformer embedding space
        patch_embeddings = self.patch_embedding(tissue_patches)  # [B, N, embed_dim]
        
        # Add learnable positional information
        patch_embeddings = patch_embeddings + self.position_embedding
        
        # Process through transformer encoder for inter-patch relationships
        transformed = self.transformer(patch_embeddings)  # [B, N, embed_dim]
        
        # Apply final layer normalization
        transformed = self.layer_norm(transformed)
        
        # Aggregate patch embeddings into global tissue context
        global_context = transformed.view(batch_size, -1)  # [B, N * embed_dim]
        
        # Project to final context embedding dimension
        tissue_context = self.context_projection(global_context)  # [B, embed_dim]
        
        return tissue_context
    
    def get_attention_weights(self, tissue_patches: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for analysis and visualization.
        
        Provides access to the attention patterns learned by the transformer
        encoder, useful for understanding inter-patch relationships and
        tissue importance weighting.
        
        Args:
            tissue_patches (torch.Tensor): Tissue patches of shape 
                (batch_size, num_patches, patch_size^3 * 2)
        
        Returns:
            torch.Tensor: Attention weights from transformer layers, or None if
                tissue_patches is None
        """
        if tissue_patches is None:
            return None
        
        batch_size = tissue_patches.shape[0]
        
        # Ensure proper tensor shape for processing
        if len(tissue_patches.shape) == 5:
            tissue_patches = tissue_patches.view(batch_size, self.num_patches, -1)
        
        # Embed patches and add positional information
        patch_embeddings = self.patch_embedding(tissue_patches)
        patch_embeddings = patch_embeddings + self.position_embedding
        
        # Note: Extracting attention weights from nn.TransformerEncoder requires
        # custom implementation. For now, return placeholder attention pattern.
        # In practice, you would need a custom transformer implementation
        # to access intermediate attention weights.
        attention_weights = torch.ones(batch_size, self.num_patches, self.num_patches,
                                     device=tissue_patches.device)
        
        logger.debug("Attention weight extraction requested (placeholder returned)")
        return attention_weights


class TissueContextToggle:
    """
    Utility class for managing tissue context toggle functionality.
    
    Provides a clean interface for enabling and disabling tissue patch
    processing in the reconstruction pipeline. This allows for easy
    experimentation with and without tissue context information.
    """
    
    @staticmethod
    def process_tissue_patches(tissue_patches: Optional[torch.Tensor], 
                              use_tissue_patches: bool) -> Optional[torch.Tensor]:
        """
        Process tissue patches based on toggle setting.
        
        Conditionally processes tissue patches depending on the toggle state.
        When disabled, returns None to skip tissue context processing entirely.
        
        Args:
            tissue_patches (torch.Tensor, optional): Raw tissue patches tensor
            use_tissue_patches (bool): Boolean toggle for tissue patch usage
        
        Returns:
            torch.Tensor: Processed tissue patches or None based on toggle
        """
        if not use_tissue_patches:
            logger.debug("Tissue patches disabled by toggle")
            return None
        return tissue_patches
    
    @staticmethod
    def create_dummy_context(batch_size: int, embed_dim: int, 
                           device: torch.device) -> torch.Tensor:
        """
        Create dummy tissue context when tissue patches are disabled.
        
        Generates a zero tensor with appropriate dimensions to maintain
        computational compatibility when tissue context is not used.
        
        Args:
            batch_size (int): Batch size for the dummy context
            embed_dim (int): Embedding dimension to match expected size
            device (torch.device): Device to create tensor on
        
        Returns:
            torch.Tensor: Zero tensor of shape (batch_size, embed_dim)
        """
        return torch.zeros(batch_size, embed_dim, device=device)
    
    @staticmethod
    def merge_contexts(cnn_features: torch.Tensor, 
                      tissue_context: Optional[torch.Tensor],
                      use_tissue_patches: bool) -> torch.Tensor:
        """
        Merge CNN features with tissue context based on toggle setting.
        
        Combines CNN-extracted features with tissue context information
        when available, or uses CNN features alone when tissue context
        is disabled.
        
        Args:
            cnn_features (torch.Tensor): CNN encoded features
            tissue_context (torch.Tensor, optional): Tissue context features or None
            use_tissue_patches (bool): Boolean toggle for tissue usage
        
        Returns:
            torch.Tensor: Merged feature representation
        """
        if not use_tissue_patches or tissue_context is None:
            logger.debug("Using CNN features only (tissue context disabled or None)")
            return cnn_features
        
        # Concatenate CNN features with tissue context for enhanced representation
        merged = torch.cat([cnn_features, tissue_context], dim=1)
        logger.debug(f"Merged CNN features {cnn_features.shape} with tissue context {tissue_context.shape}")
        return merged
