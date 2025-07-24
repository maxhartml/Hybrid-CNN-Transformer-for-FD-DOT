"""
Hybrid CNN-Transformer Model for NIR-DOT reconstruction.

This module implements the complete hybrid architecture that combines 
convolutional neural networks (CNN) with transformer encoders for 
near-infrared diffuse optical tomography (NIR-DOT) volume reconstruction.

The hybrid approach uses a two-stage learning strategy:
1. Stage 1: CNN autoencoder pre-training for spatial feature learning
2. Stage 2: Transformer integration for sequence modeling and context enhancement

The model includes optional tissue context integration for improved
reconstruction accuracy through anatomical constraints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import sys
import os
import math

# Add parent directories to path for logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    # Package imports
    from .cnn_autoencoder import CNNAutoEncoder
    from .tissue_context_encoder import TissueContextEncoder, TissueContextToggle
    from .transformer_encoder import TransformerEncoder
    from ..utils.logging_config import get_model_logger
except ImportError:
    # Standalone imports - add current directory to path
    current_dir = os.path.dirname(__file__)
    sys.path.insert(0, current_dir)
    from cnn_autoencoder import CNNAutoEncoder
    from tissue_context_encoder import TissueContextEncoder, TissueContextToggle
    from transformer_encoder import TransformerEncoder
    from utils.logging_config import get_model_logger

# =============================================================================
# HYPERPARAMETERS AND CONSTANTS
# =============================================================================

# Model Architecture Parameters
DEFAULT_INPUT_CHANNELS = 2              # Absorption and scattering coefficients
DEFAULT_OUTPUT_SIZE = (60, 60, 60)      # Target volume dimensions
DEFAULT_CNN_BASE_CHANNELS = 64          # Base CNN channels

# NIR Measurement Configuration
DEFAULT_NIR_INPUT_DIM = 8               # 8D NIR feature vectors (log_amp, phase, source_xyz, det_xyz)

# Tissue Context Encoder Configuration
DEFAULT_PATCH_SIZE = 7                  # Tissue patch size
DEFAULT_NUM_PATCHES = 2                 # Source + detector regions
DEFAULT_TISSUE_EMBED_DIM = 256          # Tissue embedding dimension
DEFAULT_TISSUE_NUM_LAYERS = 3           # Number of tissue encoder layers
DEFAULT_TISSUE_NUM_HEADS = 8            # Number of tissue attention heads

# Transformer Encoder Configuration
DEFAULT_TRANSFORMER_EMBED_DIM = 768     # Transformer embedding dimension
DEFAULT_TRANSFORMER_NUM_LAYERS = 6      # Number of transformer layers
DEFAULT_TRANSFORMER_NUM_HEADS = 12      # Number of transformer attention heads
DEFAULT_MLP_RATIO = 4                   # MLP expansion ratio
DEFAULT_DROPOUT = 0.1                   # Dropout probability

# Model Behavior Configuration
DEFAULT_USE_TISSUE_PATCHES = True       # Whether to use tissue context
DEFAULT_TRAINING_STAGE = "stage1"       # Default training stage

# Training Stages
STAGE1 = "stage1"                       # CNN autoencoder pre-training
STAGE2 = "stage2"                       # Transformer training stage

# Initialize logger for this module
logger = get_model_logger(__name__)


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture for advanced NIR-DOT reconstruction.
    
    Combines the spatial feature extraction capabilities of CNNs with the
    sequence modeling power of transformers to achieve superior reconstruction
    quality. The model supports a two-stage training paradigm and optional
    tissue context integration.
    
    Architecture Components:
    - CNN Autoencoder: Spatial feature extraction and reconstruction
    - Transformer Encoder: Sequence modeling and context integration  
    - Tissue Context Encoder: Anatomical constraint processing (optional)
    
    Training Stages:
    1. Stage 1: CNN autoencoder pre-training for low-level spatial features
    2. Stage 2: Transformer training with frozen decoder for high-level modeling
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to 1.
        output_size (Tuple[int, int, int], optional): Target output volume size. 
            Defaults to (64, 64, 64).
        base_channels (int, optional): Base CNN channels. Defaults to 64.
        transformer_layers (int, optional): Number of transformer layers. Defaults to 6.
        transformer_heads (int, optional): Number of attention heads. Defaults to 12.
        embed_dim (int, optional): Transformer embedding dimension. Defaults to 768.
        tissue_patch_size (int, optional): Size of tissue patches. Defaults to 7.
        tissue_num_patches (int, optional): Number of tissue patches. Defaults to 2.
        tissue_embed_dim (int, optional): Tissue embedding dimension. Defaults to 256.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self,
                 # CNN autoencoder configuration
                 input_channels: int = DEFAULT_INPUT_CHANNELS,  # Both absorption and scattering coefficients
                 output_size: Tuple[int, int, int] = DEFAULT_OUTPUT_SIZE,  # Match your data dimensions
                 cnn_base_channels: int = DEFAULT_CNN_BASE_CHANNELS,
                 
                 # NIR measurement configuration
                 nir_input_dim: int = DEFAULT_NIR_INPUT_DIM,  # 8D NIR feature vectors (log_amp, phase, source_xyz, det_xyz)
                 
                 # Tissue context encoder configuration
                 patch_size: int = DEFAULT_PATCH_SIZE,  # Tissue patch size matching data format
                 num_patches: int = DEFAULT_NUM_PATCHES,  # Source + detector regions
                 tissue_embed_dim: int = DEFAULT_TISSUE_EMBED_DIM,
                 tissue_num_layers: int = DEFAULT_TISSUE_NUM_LAYERS,
                 tissue_num_heads: int = DEFAULT_TISSUE_NUM_HEADS,
                 
                 # Transformer encoder configuration
                 transformer_embed_dim: int = DEFAULT_TRANSFORMER_EMBED_DIM,
                 transformer_num_layers: int = DEFAULT_TRANSFORMER_NUM_LAYERS,
                 transformer_num_heads: int = DEFAULT_TRANSFORMER_NUM_HEADS,
                 mlp_ratio: int = DEFAULT_MLP_RATIO,
                 dropout: float = DEFAULT_DROPOUT,
                 
                 # Model behavior configuration
                 use_tissue_patches: bool = DEFAULT_USE_TISSUE_PATCHES,
                 training_stage: str = DEFAULT_TRAINING_STAGE):
        
        super().__init__()
        
        logger.info(f"🏗️  Initializing Hybrid CNN-Transformer: {training_stage} mode, "
                   f"tissue_patches={'enabled' if use_tissue_patches else 'disabled'}")
        
        # Store model configuration
        self.use_tissue_patches = use_tissue_patches
        self.training_stage = training_stage
        self.output_size = output_size
        self.nir_input_dim = nir_input_dim  # Store NIR dimension
        
        # Initialize CNN Autoencoder (used in both stages)
        self.cnn_autoencoder = CNNAutoEncoder(
            input_channels=input_channels,
            output_size=output_size,
            base_channels=cnn_base_channels
        )
        
        # Initialize Tissue Context Encoder (conditional)
        self.tissue_encoder = TissueContextEncoder(
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=tissue_embed_dim,
            num_layers=tissue_num_layers,
            num_heads=tissue_num_heads,
            dropout=dropout
        ) if use_tissue_patches else None
        
        # Initialize Transformer Encoder (stage 2 component)
        self.transformer_encoder = TransformerEncoder(
            cnn_feature_dim=self.cnn_autoencoder.encoder.feature_dim,
            tissue_context_dim=tissue_embed_dim if use_tissue_patches else 0,
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # NIR projection network for Stage 2 (NIR measurements → CNN feature space)
        self.nir_projection = nn.Sequential(
            nn.Linear(self.nir_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256), 
            nn.ReLU(),
            nn.Linear(256, 512)    # Project to CNN feature dimension
        )
        
        # Utility for tissue context toggle functionality
        self.toggle_utils = TissueContextToggle()
        
        # Initialize network weights
        self._init_weights()
        
        # Log model characteristics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"📊 Hybrid Model initialized: {total_params:,} total params, "
                   f"{trainable_params:,} trainable params")
    
    def _init_weights(self):
        """
        Initialize weights for transformer and tissue encoder components.
        
        The CNN autoencoder initializes its own weights. This method handles
        additional components that may need specific initialization.
        """
        # Initialize any custom projection layers or additional components
        # CNN autoencoder and transformer/tissue encoders handle their own initialization
        logger.debug("🔧 Custom weight initialization completed for hybrid model components")
    
    def forward(self, dot_measurements: torch.Tensor,
                tissue_patches: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model.
        
        Processes input measurements through the appropriate pipeline based on
        the current training stage. Stage 1 uses only the CNN autoencoder, while
        Stage 2 integrates transformer and optional tissue context.
        
        Args:
            dot_measurements (torch.Tensor): DOT measurements of shape 
                (batch_size, channels, D, H, W)
            tissue_patches (torch.Tensor, optional): Tissue property patches of shape
                (batch_size, num_patches, patch_size^3 * 2). Defaults to None.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'reconstruction': Reconstructed volume
                - 'cnn_features': CNN-extracted features (stage 2 only)
                - 'enhanced_features': Transformer-enhanced features (stage 2 only)
                - 'attention_weights': Attention weights (stage 2 only, if available)
        """
        batch_size = dot_measurements.shape[0]
        device = dot_measurements.device
        
        outputs = {}
        
        if self.training_stage == STAGE1:
            # Stage 1: CNN autoencoder only
            reconstructed = self.cnn_autoencoder(dot_measurements)
            outputs.update({
                'reconstructed': reconstructed,
                'stage': STAGE1
            })
            
        elif self.training_stage == STAGE2:
            # Stage 2: Handle NIR measurements → Transformer → CNN decoder
            
            # Check input type: NIR measurements vs ground truth volumes
            # NEW FORMAT: NIR measurements are (batch_size, 1500, 8) for complete phantoms
            # OLD FORMAT: Individual measurements were (batch_size, 8)
            if len(dot_measurements.shape) == 3 and dot_measurements.shape[2] == self.nir_input_dim:
                # Complete phantom NIR measurements: (batch_size, 1500, nir_input_dim)
                batch_size, n_measurements, n_features = dot_measurements.shape
                
                # Reshape to process all measurements in batch
                nir_features = dot_measurements.view(-1, n_features)  # (batch_size * 1500, nir_input_dim)
                
                # Project all measurements to CNN feature space  
                projected_features = self.nir_projection(nir_features)  # (batch_size * 1500, 512)
                
                # Reshape back to batch format and aggregate measurements per phantom
                projected_features = projected_features.view(batch_size, n_measurements, 512)  # (batch_size, 1500, 512)
                
                # Aggregate measurements to single feature vector per phantom
                # Using mean pooling across all measurements - could also use attention pooling
                cnn_features = projected_features.mean(dim=1)  # (batch_size, 512)
                
            elif len(dot_measurements.shape) == 2 and dot_measurements.shape[1] == self.nir_input_dim:
                # Individual NIR measurements: (batch_size, nir_input_dim) - for compatibility
                nir_features = dot_measurements  # Shape: (batch_size, nir_input_dim)
                
                # Project NIR measurements to CNN feature space  
                cnn_features = self.nir_projection(nir_features)  # Shape: (batch_size, 512)
                
            else:
                # Ground truth volumes input (for compatibility): extract CNN features
                with torch.no_grad() if self.training else torch.enable_grad():
                    cnn_features = self.cnn_autoencoder.encode(dot_measurements)  # (batch, 512)
            
            # Process tissue patches based on toggle
            processed_tissue_patches = self.toggle_utils.process_tissue_patches(
                tissue_patches, self.use_tissue_patches
            )
            
            # Encode tissue context if available
            tissue_context = None
            if self.use_tissue_patches and self.tissue_encoder is not None:
                if processed_tissue_patches is not None:
                    tissue_context = self.tissue_encoder(processed_tissue_patches)
            
            # Transform features using transformer (expecting 512D CNN features)
            # Ensure CNN features require gradients for backpropagation
            if not cnn_features.requires_grad:
                cnn_features = cnn_features.detach().requires_grad_(True)
            
            enhanced_features, attention_weights = self.transformer_encoder(
                cnn_features=cnn_features,
                tissue_context=tissue_context,
                use_tissue_patches=self.use_tissue_patches
            )
            
            # Decode using frozen CNN decoder (gradients flow through, but params frozen)
            reconstructed = self.cnn_autoencoder.decode(enhanced_features)
            
            outputs.update({
                'reconstructed': reconstructed,
                'cnn_features': cnn_features,
                'enhanced_features': enhanced_features,
                'stage': 'stage2'
            })
            
            # Add attention weights if available
            if attention_weights is not None:
                outputs['attention_weights'] = attention_weights
        
        else:
            raise ValueError(f"Invalid training stage: {self.training_stage}. "
                           f"Expected 'stage1' or 'stage2'")
        
        return outputs
    
    def set_training_stage(self, stage: str):
        """
        Configure model for specific training stage.
        
        Sets the appropriate training/evaluation modes and parameter freezing
        for the two-stage training approach. Stage 1 trains only the CNN
        autoencoder, while Stage 2 freezes the decoder and trains the transformer.
        
        Args:
            stage (str): Training stage - either "stage1" or "stage2"
            
        Raises:
            ValueError: If stage is not "stage1" or "stage2"
        """
        if stage not in [STAGE1, STAGE2]:
            raise ValueError(f"Invalid stage: {stage}. Must be '{STAGE1}' or '{STAGE2}'")
        
        self.training_stage = stage
        logger.info(f"🎯 Setting training stage to: {stage}")
        
        if stage == STAGE1:
            # Stage 1: Train CNN autoencoder end-to-end
            self.cnn_autoencoder.train()
            if self.tissue_encoder is not None:
                self.tissue_encoder.eval()
            self.transformer_encoder.eval()
            
        elif stage == STAGE2:
            # Stage 2: Freeze CNN decoder, train transformer components
            self.cnn_autoencoder.encoder.eval()
            self.cnn_autoencoder.decoder.eval()
            
            # Freeze CNN decoder parameters
            for param in self.cnn_autoencoder.decoder.parameters():
                param.requires_grad = False
            
            # Enable training for transformer and tissue encoder
            self.transformer_encoder.train()
            if self.tissue_encoder is not None:
                self.tissue_encoder.train()
                
            logger.info("🔒 CNN decoder frozen, transformer components enabled for training")
    
    def toggle_tissue_patches(self, use_tissue_patches: bool):
        """
        Toggle tissue patch usage for experimental control.
        
        Enables or disables tissue context processing for A/B testing and
        ablation studies. When disabled, the model operates in CNN+Transformer
        mode without tissue context.
        
        Args:
            use_tissue_patches (bool): Whether to use tissue context
        """
        self.use_tissue_patches = use_tissue_patches
        logger.info(f"🔄 Tissue patches {'enabled' if use_tissue_patches else 'disabled'}")
        
        if not use_tissue_patches and self.tissue_encoder is not None:
            self.tissue_encoder.eval()
    
    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get currently trainable parameters based on training stage.
        
        Returns different parameter sets depending on the current training stage
        to enable stage-specific optimization and parameter counting.
        
        Returns:
            Dict[str, torch.nn.Parameter]: Dictionary of trainable parameters
                organized by component name
        """
        trainable_params = {}
        
        if self.training_stage == STAGE1:
            # Stage 1: Only CNN autoencoder parameters are trainable
            for name, param in self.cnn_autoencoder.named_parameters():
                if param.requires_grad:
                    trainable_params[f"cnn_autoencoder.{name}"] = param
                    
        elif self.training_stage == STAGE2:
            # Stage 2: Transformer and tissue encoder parameters are trainable
            for name, param in self.transformer_encoder.named_parameters():
                if param.requires_grad:
                    trainable_params[f"transformer_encoder.{name}"] = param
            
            if self.use_tissue_patches and self.tissue_encoder is not None:
                for name, param in self.tissue_encoder.named_parameters():
                    if param.requires_grad:
                        trainable_params[f"tissue_encoder.{name}"] = param
        
        logger.debug(f"Found {len(trainable_params)} trainable parameters for {self.training_stage}")
        return trainable_params
    
    def load_stage1_weights(self, checkpoint_path: str):
        """
        Load pre-trained CNN autoencoder weights from stage 1.
        
        Loads the CNN autoencoder weights from a stage 1 checkpoint to initialize
        the model for stage 2 training. Essential for the two-stage training approach.
        
        Args:
            checkpoint_path (str): Path to the stage 1 checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"📂 Loading stage 1 weights from: {checkpoint_path}")
        
        # Extract CNN autoencoder state dict from checkpoint
        cnn_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('cnn_autoencoder.'):
                new_key = key[len('cnn_autoencoder.'):]
                cnn_state_dict[new_key] = value
        
        # Load weights into CNN autoencoder
        self.cnn_autoencoder.load_state_dict(cnn_state_dict)
        logger.info("✅ Stage 1 CNN autoencoder weights loaded successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model configuration and statistics.
        
        Provides detailed information about model architecture, parameter counts,
        and current configuration. Useful for logging and model analysis.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information including
                parameter counts, configuration, and component details
        """
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
