"""
Model components for hybrid CNN-Transformer NIR-DOT reconstruction.

This package contains the neural network architectures for near-infrared 
diffuse optical tomography (NIR-DOT) volume reconstruction using a hybrid
approach that combines CNNs and transformers.

Two-stage training approach:
1. CNN autoencoder pre-training for spatial feature learning
2. Transformer training with frozen decoder for sequence modeling
"""

from .cnn_autoencoder import CNNAutoEncoder
from .tissue_context_encoder import TissueContextEncoder  
from .transformer_encoder import TransformerEncoder
from .hybrid_model import HybridCNNTransformer

__all__ = [
    'CNNAutoEncoder',
    'TissueContextEncoder', 
    'TransformerEncoder',
    'HybridCNNTransformer'
]
