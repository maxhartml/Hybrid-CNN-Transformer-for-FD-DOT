"""
Hybrid CNN-Transformer model implementing Robin Dale's two-stage training approach.
Integrates all components with clean toggle functionality for tissue patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .cnn_autoencoder import CNNAutoEncoder
from .tissue_context_encoder import TissueContextEncoder, TissueContextToggle
from .transformer_encoder import TransformerEncoder


class HybridCNNTransformer(nn.Module):
    """
    Complete hybrid CNN-Transformer model following Robin Dale's approach.
    
    Two-stage training:
    1. CNN autoencoder pre-training
    2. Transformer training with frozen decoder
    
    Features:
    - Toggle functionality for tissue patches
    - Clean experimental design for A/B testing
    - Seamless integration with existing data pipeline
    """
    
    def __init__(self,
                 # CNN autoencoder parameters
                 input_channels: int = 1,
                 output_size: Tuple[int, int, int] = (64, 64, 64),
                 cnn_base_channels: int = 64,
                 
                 # Tissue context encoder parameters
                 patch_size: int = 7,  # Match actual data
                 num_patches: int = 2,  # Match actual data (source + detector)
                 tissue_embed_dim: int = 256,
                 tissue_num_layers: int = 3,
                 tissue_num_heads: int = 8,
                 
                 # Transformer encoder parameters
                 transformer_embed_dim: int = 768,
                 transformer_num_layers: int = 6,
                 transformer_num_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 
                 # Training configuration
                 use_tissue_patches: bool = True,
                 training_stage: str = "stage1"):  # "stage1" or "stage2"
        
        super().__init__()
        
        # Store configuration
        self.use_tissue_patches = use_tissue_patches
        self.training_stage = training_stage
        self.output_size = output_size
        
        # CNN Autoencoder (stage 1)
        self.cnn_autoencoder = CNNAutoEncoder(
            input_channels=input_channels,
            output_size=output_size,
            base_channels=cnn_base_channels
        )
        
        # Tissue Context Encoder (stage 2)
        self.tissue_encoder = TissueContextEncoder(
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=tissue_embed_dim,
            num_layers=tissue_num_layers,
            num_heads=tissue_num_heads,
            dropout=dropout
        ) if use_tissue_patches else None
        
        # Transformer Encoder (stage 2)
        self.transformer_encoder = TransformerEncoder(
            cnn_feature_dim=self.cnn_autoencoder.encoder.feature_dim,
            tissue_context_dim=tissue_embed_dim if use_tissue_patches else 0,
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Tissue context toggle utility
        self.toggle_utils = TissueContextToggle()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for newly added components"""
        # CNN autoencoder and other components have their own initialization
        pass
    
    def forward(self, dot_measurements: torch.Tensor,
                tissue_patches: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model.
        
        Args:
            dot_measurements: DOT measurement volumes [batch_size, channels, H, W, D]
            tissue_patches: Tissue patches [batch_size, num_patches, patch_size^3] or None
        
        Returns:
            Dictionary containing model outputs
        """
        batch_size = dot_measurements.shape[0]
        device = dot_measurements.device
        
        outputs = {}
        
        if self.training_stage == "stage1":
            # Stage 1: CNN autoencoder only
            reconstructed = self.cnn_autoencoder(dot_measurements)
            outputs.update({
                'reconstructed': reconstructed,
                'stage': 'stage1'
            })
            
        elif self.training_stage == "stage2":
            # Stage 2: Transformer with frozen decoder
            
            # Encode with CNN (frozen in stage 2)
            with torch.no_grad() if self.training else torch.enable_grad():
                cnn_features = self.cnn_autoencoder.encode(dot_measurements)
            
            # Process tissue patches based on toggle
            processed_tissue_patches = self.toggle_utils.process_tissue_patches(
                tissue_patches, self.use_tissue_patches
            )
            
            # Encode tissue context if available
            tissue_context = None
            if self.use_tissue_patches and self.tissue_encoder is not None:
                tissue_context = self.tissue_encoder(processed_tissue_patches)
            
            # Enhance features with transformer
            enhanced_features, attention_weights = self.transformer_encoder(
                cnn_features, tissue_context, self.use_tissue_patches
            )
            
            # Decode with frozen CNN decoder
            with torch.no_grad():
                reconstructed = self.cnn_autoencoder.decode(enhanced_features)
            
            outputs.update({
                'reconstructed': reconstructed,
                'cnn_features': cnn_features,
                'enhanced_features': enhanced_features,
                'tissue_context': tissue_context,
                'attention_weights': attention_weights,
                'stage': 'stage2'
            })
        
        else:
            raise ValueError(f"Invalid training stage: {self.training_stage}")
        
        return outputs
    
    def set_training_stage(self, stage: str):
        """Set the training stage and configure model accordingly"""
        if stage not in ["stage1", "stage2"]:
            raise ValueError(f"Invalid stage: {stage}. Must be 'stage1' or 'stage2'")
        
        self.training_stage = stage
        
        if stage == "stage1":
            # Stage 1: Train CNN autoencoder only
            self.cnn_autoencoder.train()
            if self.tissue_encoder is not None:
                self.tissue_encoder.eval()
            self.transformer_encoder.eval()
            
        elif stage == "stage2":
            # Stage 2: Freeze CNN decoder, train transformer and tissue encoder
            self.cnn_autoencoder.encoder.eval()
            self.cnn_autoencoder.decoder.eval()
            
            # Freeze CNN decoder parameters
            for param in self.cnn_autoencoder.decoder.parameters():
                param.requires_grad = False
            
            # Enable training for transformer and tissue encoder
            self.transformer_encoder.train()
            if self.tissue_encoder is not None:
                self.tissue_encoder.train()
    
    def toggle_tissue_patches(self, use_tissue_patches: bool):
        """Toggle tissue patch usage for A/B testing"""
        self.use_tissue_patches = use_tissue_patches
        
        if not use_tissue_patches and self.tissue_encoder is not None:
            self.tissue_encoder.eval()
    
    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Get trainable parameters based on current stage"""
        trainable_params = {}
        
        if self.training_stage == "stage1":
            # Stage 1: Only CNN autoencoder parameters
            for name, param in self.cnn_autoencoder.named_parameters():
                if param.requires_grad:
                    trainable_params[f"cnn_autoencoder.{name}"] = param
                    
        elif self.training_stage == "stage2":
            # Stage 2: Transformer and tissue encoder parameters
            for name, param in self.transformer_encoder.named_parameters():
                if param.requires_grad:
                    trainable_params[f"transformer_encoder.{name}"] = param
            
            if self.use_tissue_patches and self.tissue_encoder is not None:
                for name, param in self.tissue_encoder.named_parameters():
                    if param.requires_grad:
                        trainable_params[f"tissue_encoder.{name}"] = param
        
        return trainable_params
    
    def load_stage1_weights(self, checkpoint_path: str):
        """Load pre-trained CNN autoencoder weights for stage 2"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract CNN autoencoder state dict
        cnn_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('cnn_autoencoder.'):
                new_key = key[len('cnn_autoencoder.'):]
                cnn_state_dict[new_key] = value
        
        # Load weights
        self.cnn_autoencoder.load_state_dict(cnn_state_dict)
        print(f"Loaded stage 1 weights from {checkpoint_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        cnn_params = sum(p.numel() for p in self.cnn_autoencoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        tissue_params = sum(p.numel() for p in self.tissue_encoder.parameters()) if self.tissue_encoder else 0
        
        return {
            'training_stage': self.training_stage,
            'use_tissue_patches': self.use_tissue_patches,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_parameters': cnn_params,
            'transformer_parameters': transformer_params,
            'tissue_encoder_parameters': tissue_params,
            'output_size': self.output_size
        }
