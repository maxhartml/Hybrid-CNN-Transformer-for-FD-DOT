"""
Stage 1 CNN Autoencoder Teacher for Latent-Only Stage 2 Training

This module provides a wrapper around a pre-trained Stage 1 CNN autoencoder model
to serve as an online teacher for Stage 2 latent-only training. The teacher
generates ground truth latent representations for the student transformer to match.

The teacher loads ONLY the CNN autoencoder weights from the Stage 1 checkpoint,
ignoring any transformer-related parameters to avoid size mismatch issues.
"""

import torch
import torch.nn as nn
from pathlib import Path
import os
import sys

# Add code directory to path for imports
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from models.hybrid_model import HybridCNNTransformer
from models.cnn_autoencoder import CNNAutoEncoder
from training.training_config import LATENT_DIM, TRAINING_STAGE1, CUDA_DEVICE, N_MEASUREMENTS
from utils.logging_config import get_training_logger

# Initialize module logger
logger = get_training_logger(__name__)

# Default device - can be overridden in constructor
DEFAULT_DEVICE = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")


class TeacherStage1:
    """
    Wrapper for pre-trained Stage 1 CNN autoencoder to generate latent targets.
    
    This class loads a pre-trained Stage 1 CNN autoencoder (not the full hybrid model)
    and uses it to generate latent representations that serve as ground truth targets 
    for Stage 2 latent-only training.
    """
    
    def __init__(self, checkpoint_path: str, device: torch.device = DEFAULT_DEVICE):
        """
        Initialize the teacher model.
        
        Args:
            checkpoint_path: Path to the pre-trained Stage 1 checkpoint
            device: Device to load the model on
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.checkpoint_path = checkpoint_path
        
        # Load the pre-trained Stage 1 CNN autoencoder
        self.model = self._load_stage1_model(checkpoint_path)
        self.model.eval()  # Always in eval mode
        
        # Freeze all parameters - teacher is fixed
        for param in self.model.parameters():
            param.requires_grad = False
            
        logger.info(f"✅ Loaded Stage 1 CNN autoencoder teacher from: {checkpoint_path}")
        logger.info(f"🔒 Teacher model parameters frozen: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_stage1_model(self, checkpoint_path: str) -> CNNAutoEncoder:
        """Load only the CNN autoencoder weights from Stage 1 checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            full_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            full_state_dict = checkpoint['state_dict']
        else:
            full_state_dict = checkpoint
        
        # Handle compiled model state dict (remove _orig_mod. prefix)
        if any(key.startswith('_orig_mod.') for key in full_state_dict.keys()):
            full_state_dict = {k.replace('_orig_mod.', ''): v for k, v in full_state_dict.items()}
        
        # Filter only CNN autoencoder keys (ignore transformer-related keys)
        cnn_state_dict = {}
        cnn_prefix = 'cnn_autoencoder.'
        
        for key, value in full_state_dict.items():
            if key.startswith(cnn_prefix):
                # Remove the 'cnn_autoencoder.' prefix to match CNNAutoEncoder structure
                new_key = key[len(cnn_prefix):]
                cnn_state_dict[new_key] = value
        
        logger.info(f"🔍 Extracted {len(cnn_state_dict)} CNN autoencoder parameters from checkpoint")
        logger.info(f"🚫 Ignored {len(full_state_dict) - len(cnn_state_dict)} non-CNN parameters")
        
        # Initialize only the CNN autoencoder (not the full hybrid model)
        model = CNNAutoEncoder().to(self.device)
        
        # Load the filtered CNN weights
        model.load_state_dict(cnn_state_dict, strict=False)  # Allow partial loading
        
        logger.info("✅ Loaded CNN autoencoder weights from Stage 1 checkpoint (ignoring transformer parameters)")
        
        return model
    
    @torch.no_grad()
    def encode_from_gt_std(self, gt_std: torch.Tensor) -> torch.Tensor:
        """
        Encode standardized ground truth using Stage 1 CNN encoder.
        
        Args:
            gt_std: Standardized ground truth [batch_size, 2, 64, 64, 64]
            
        Returns:
            latent: Encoded latent representation [batch_size, 256]
        """
        self.model.eval()
        
        # Use the CNN autoencoder encoder directly
        with torch.no_grad():
            latent = self.model.encoder(gt_std)
            return latent
    
    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation using Stage 1 CNN decoder.
        
        Args:
            z: Latent representation [batch_size, 256]
            
        Returns:
            pred_std: Reconstructed volume in standardized space [batch_size, 2, 64, 64, 64]
        """
        self.model.eval()
        
        # Use the CNN autoencoder decoder directly
        with torch.no_grad():
            pred_std = self.model.decoder(z)
            return pred_std

    @torch.no_grad()
    def get_latent_representations(self, nir_input: torch.Tensor) -> torch.Tensor:
        """
        DEPRECATED: This method causes stage flipping issues.
        Use encode_from_gt_std() for proper teacher-student training.
        
        Args:
            nir_input: Input NIR data tensor [batch_size, n_measurements, 8]
            
        Returns:
            latent: Latent representations [batch_size, latent_dim]
        """
        # This is a temporary fallback - should not be used in production
        batch_size = nir_input.shape[0]
        device = nir_input.device
        
        # Return random latent for now - caller should use encode_from_gt_std instead
        return torch.randn(batch_size, LATENT_DIM, device=device, dtype=nir_input.dtype)
    
    def get_full_reconstruction(self, gt_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get both latent representation and full reconstruction from teacher.
        
        This method uses the proper Stage 1 encoder-decoder path without stage flipping.
        
        Args:
            gt_std: Standardized ground truth [batch_size, 2, 64, 64, 64]
            
        Returns:
            latent: Latent representations [batch_size, latent_dim]
            reconstruction: Reconstructed DOT data [batch_size, 2, 64, 64, 64]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Use proper Stage 1 encoder-decoder path
            latent = self.encode_from_gt_std(gt_std)
            reconstruction = self.decode_from_latent(latent)
        
        return latent, reconstruction
    
    def __call__(self, gt_std: torch.Tensor) -> torch.Tensor:
        """Convenience method for getting latent representations."""
        return self.encode_from_gt_std(gt_std)


def load_teacher_stage1(checkpoint_path: str = None, device: torch.device = DEFAULT_DEVICE) -> TeacherStage1:
    """
    Convenience function to load Stage 1 CNN autoencoder teacher model.
    
    Args:
        checkpoint_path: Path to Stage 1 checkpoint. If None, uses default location.
        device: Device to load on
        
    Returns:
        TeacherStage1 instance with CNN autoencoder loaded
    """
    if checkpoint_path is None:
        # Automatically find the best Stage 1 checkpoint
        from .training_utils import find_best_checkpoint
        from .training_config import CHECKPOINT_BASE_DIR
        
        logger.info("🔍 No checkpoint path provided - searching for best Stage 1 checkpoint...")
        checkpoint_path = find_best_checkpoint(CHECKPOINT_BASE_DIR, "stage1")
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No valid Stage 1 checkpoints found in {CHECKPOINT_BASE_DIR}. "
                f"Please run Stage 1 training first or provide a specific checkpoint path."
            )
    
    return TeacherStage1(checkpoint_path, device)
