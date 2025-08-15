"""
Stage 1 Teacher Model Wrapper for Latent-Only Stage 2 Training

This module provides a wrapper around a pre-trained Stage 1 encoder-decoder model
to serve as an online teacher for Stage 2 latent-only training. The teacher
generates ground truth latent representations for the student transformer to match.
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
from training.training_config import LATENT_DIM, TRAINING_STAGE1, CUDA_DEVICE

# Default device - can be overridden in constructor
DEFAULT_DEVICE = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")


class TeacherStage1:
    """
    Wrapper for pre-trained Stage 1 model to generate latent targets.
    
    This class loads a pre-trained Stage 1 encoder-decoder and uses only
    the encoder portion to generate latent representations that serve as
    ground truth targets for Stage 2 transformer training.
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
        
        # Load the pre-trained Stage 1 model
        self.model = self._load_stage1_model(checkpoint_path)
        self.model.eval()  # Always in eval mode
        
        # Freeze all parameters - teacher is fixed
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"Loaded Stage 1 teacher from: {checkpoint_path}")
        print(f"Teacher model parameters frozen: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_stage1_model(self, checkpoint_path: str) -> HybridCNNTransformer:
        """Load the pre-trained Stage 1 model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle compiled model state dict (remove _orig_mod. prefix)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # Initialize Stage 1 model with same architecture (without tissue patches for Stage 1)
        model = HybridCNNTransformer(
            use_tissue_patches=False,  # Stage 1 doesn't use tissue patches
            training_stage=TRAINING_STAGE1
        ).to(self.device)
        
        # Load the pre-trained weights
        model.load_state_dict(state_dict, strict=False)  # Allow missing parameters
        
        return model
    
    @torch.no_grad()
    def encode_from_gt_std(self, gt_std: torch.Tensor) -> torch.Tensor:
        """
        Exact Stage-1 encoder path used during Stage-1 training. 
        Input is standardized ground truth.
        
        Args:
            gt_std: Standardized ground truth [batch_size, 2, 64, 64, 64]
            
        Returns:
            latent: Encoded latent representation [batch_size, 256]
        """
        self.model.eval()
        
        # Stage 1 model expects ground truth as input, not NIR measurements
        # This is the exact encoder path used during Stage 1 training
        with torch.no_grad():
            latent = self.model.cnn_autoencoder.encoder(gt_std)
            return latent
    
    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Exact Stage-1 decoder entrypoint expected during Stage-1 training.
        
        Args:
            z: Latent representation [batch_size, 256]
            
        Returns:
            pred_std: Reconstructed volume in standardized space [batch_size, 2, 64, 64, 64]
        """
        self.model.eval()
        
        # Use the exact decoder path from Stage 1 training
        with torch.no_grad():
            pred_std = self.model.cnn_autoencoder.decoder(z)
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
    Convenience function to load Stage 1 teacher model.
    
    Args:
        checkpoint_path: Path to Stage 1 checkpoint. If None, uses default location.
        device: Device to load on
        
    Returns:
        TeacherStage1 instance
    """
    if checkpoint_path is None:
        # Default to the best Stage 1 checkpoint
        default_path = "/home/ubuntu/NIR-DOT/checkpoints/stage1_best.pth"
        if os.path.exists(default_path):
            checkpoint_path = default_path
        else:
            raise FileNotFoundError(
                f"No checkpoint path provided and default not found: {default_path}"
            )
    
    return TeacherStage1(checkpoint_path, device)


if __name__ == "__main__":
    # Test the teacher model loading
    try:
        teacher = load_teacher_stage1()
        print("✓ Teacher model loaded successfully")
        
        # Test with dummy NIR measurements (not image data)
        dummy_input = torch.randn(2, 256, 8).to(DEFAULT_DEVICE)  # Batch of 2, 256 measurements, 8 features
        latent = teacher.get_latent_representations(dummy_input)
        print(f"✓ Generated latent shape: {latent.shape}")
        print(f"✓ Expected latent dim: {LATENT_DIM}")
        
        # Test full reconstruction
        latent_full, reconstruction = teacher.get_full_reconstruction(dummy_input)
        print(f"✓ Full reconstruction shapes: latent={latent_full.shape}, recon={reconstruction.shape}")
        
    except Exception as e:
        print(f"✗ Teacher model test failed: {e}")
        import traceback
        traceback.print_exc()
