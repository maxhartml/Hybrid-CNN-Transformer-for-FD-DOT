#!/usr/bin/env python3
"""
🎯 UNIVERSAL PHYSICS-BASED VISUALIZATION UTILITIES 🎯

Revolutionary visualization approach for NIR-DOT training that preserves physical meaning:

BREAKTHROUGH FEATURES:
✅ Universal Physics Normalization: Colors have absolute meaning across ALL phantoms
✅ Cross-Phantom Comparability: Same optical properties → Same colors
✅ Scientific Accuracy: No misleading visualizations, physics preserved
✅ Model Debugging: Easily spot reconstruction errors
✅ Training Insights: See if model learns physics correctly

NORMALIZATION RANGES:
• Absorption: 0 (air) → 0.0245 mm⁻¹ (strongest tumor: 0.007 × 3.5)
• Scattering: 0 (air) → 2.95 mm⁻¹ (strongest tumor: 1.18 × 2.5)

BENEFITS OVER OLD APPROACH:
❌ Old: Per-image normalization → Physics meaning lost
✅ New: Universal normalization → Physics preserved
❌ Old: Can't compare phantoms → Misleading
✅ New: Direct phantom comparison → Scientifically accurate

Functions:
    log_reconstruction_images_to_wandb: Universal physics-aware reconstruction visualization
    universal_physics_normalization: Core normalization preserving physics relationships

Author: Max Hart
Date: August 2025 - Universal Physics Update
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import torch
import wandb
from typing import Optional

# Project imports
from code.utils.logging_config import get_training_logger

# Initialize logger
logger = get_training_logger(__name__)

# =============================================================================
# SHARED VISUALIZATION FUNCTIONS
# =============================================================================

def log_reconstruction_images_to_wandb(predictions: torch.Tensor, 
                                     targets: torch.Tensor, 
                                     epoch: int, 
                                     prefix: str = "Reconstructions",
                                     step: Optional[int] = None,
                                     phantom_ids: Optional[np.ndarray] = None) -> None:
    """
    Enhanced function to log 3D reconstruction slices to W&B for visualization.
    
    This function extracts middle slices from 3D volumes in all three dimensions
    (XY, XZ, YZ) for the first TWO phantoms in the batch, and logs them to 
    Weights & Biases with consistent global normalization to preserve relative 
    intensity relationships. Each image includes the actual phantom ID for 
    easy cross-reference with original data files.
    
    Args:
        predictions (torch.Tensor): Predicted volumes [batch, 2, D, H, W]
        targets (torch.Tensor): Target volumes [batch, 2, D, H, W]
        epoch (int): Current epoch number
        prefix (str): Prefix for W&B logging keys. Default: "Reconstructions"
        step (Optional[int]): Optional step number for logging
        phantom_ids (Optional[np.ndarray]): Array of phantom IDs from validation batch
    """
    try:
        # Extract data for first TWO phantoms in batch for variety
        num_phantoms_to_show = min(2, predictions.shape[0])
        
        logger.debug(f"Logging images for {num_phantoms_to_show} phantoms - Pred shape: {predictions.shape}, Target shape: {targets.shape}")
        
        # Extract slices from different dimensions for BOTH channels and BOTH phantoms
        absorption_channel = 0  # μₐ (absorption coefficient)
        scattering_channel = 1  # μ′s (reduced scattering coefficient)
        
        # We'll log images for each phantom separately
        for phantom_idx in range(num_phantoms_to_show):
            pred_phantom = predictions[phantom_idx].cpu().numpy()  # Shape: [2, D, H, W]
            target_phantom = targets[phantom_idx].cpu().numpy()
            
            # Get actual phantom ID if available, otherwise use generic numbering
            if phantom_ids is not None and phantom_idx < len(phantom_ids):
                actual_phantom_id = int(phantom_ids[phantom_idx])
                phantom_label = f"Phantom_{actual_phantom_id:03d}"  # e.g., "Phantom_001", "Phantom_452"
            else:
                phantom_label = f"Phantom_{phantom_idx + 1}"  # Fallback: "Phantom_1", "Phantom_2"
            
            # Extract all slices for this phantom (all 3 orientations × 2 channels)
            # XY plane (Z=32) - middle slice in Z dimension  
            pred_xy_abs = pred_phantom[absorption_channel, :, :, pred_phantom.shape[-1]//2]
            target_xy_abs = target_phantom[absorption_channel, :, :, target_phantom.shape[-1]//2]
            pred_xy_scat = pred_phantom[scattering_channel, :, :, pred_phantom.shape[-1]//2]
            target_xy_scat = target_phantom[scattering_channel, :, :, target_phantom.shape[-1]//2]
            
            # XZ plane (Y=32) - middle slice in Y dimension  
            pred_xz_abs = pred_phantom[absorption_channel, :, pred_phantom.shape[-2]//2, :]
            target_xz_abs = target_phantom[absorption_channel, :, target_phantom.shape[-2]//2, :]
            pred_xz_scat = pred_phantom[scattering_channel, :, pred_phantom.shape[-2]//2, :]
            target_xz_scat = target_phantom[scattering_channel, :, target_phantom.shape[-2]//2, :]
            
            # YZ plane (X=32) - middle slice in X dimension
            pred_yz_abs = pred_phantom[absorption_channel, pred_phantom.shape[-3]//2, :, :]
            target_yz_abs = target_phantom[absorption_channel, pred_phantom.shape[-3]//2, :, :]
            pred_yz_scat = pred_phantom[scattering_channel, pred_phantom.shape[-3]//2, :, :]
            target_yz_scat = target_phantom[scattering_channel, pred_phantom.shape[-3]//2, :, :]
            
            # UNIVERSAL PHYSICS-BASED NORMALIZATION - PRESERVES CROSS-PHANTOM COMPARABILITY
            def universal_physics_normalization(data, channel_name):
                """
                🎯 UNIVERSAL PHYSICS-BASED NORMALIZATION 🎯
                
                Revolutionary approach that preserves physical meaning across ALL phantoms:
                • Air (0.0) → Black (0) universally
                • Physics values → Consistent colors across phantoms
                • Same optical properties → Same visualization colors
                • Cross-phantom comparison becomes meaningful
                • Scientific accuracy preserved
                
                BENEFITS:
                ✅ Physics-preserving: Colors have absolute meaning
                ✅ Cross-phantom comparable: Compare different phantoms directly  
                ✅ Scientifically accurate: No misleading visualizations
                ✅ Model debugging: Spot reconstruction errors easily
                ✅ Training insights: See if model learns physics correctly
                
                Args:
                    data: 2D slice of optical properties (any phantom)
                    channel_name: "absorption" or "scattering" to determine physical ranges
                
                Returns:
                    Normalized intensities [0-255] preserving universal physics relationships
                """
                data_np = data.cpu().numpy() if hasattr(data, 'cpu') else data
                
                # UNIVERSAL PHYSICS RANGES: Air (0) to Maximum Possible Values
                if "absorption" in channel_name.lower():
                    # Full absorption range: 0 (air) to 0.0245 (strongest tumor: 0.007 × 3.5)
                    max_value = 0.0245
                else:
                    # Full scattering range: 0 (air) to 2.95 (strongest tumor: 1.18 × 2.5)  
                    max_value = 2.95
                
                # Universal linear mapping: physics_value → grayscale_intensity
                # Air (0) → Black (0), Max_possible → White (255)
                normalized = np.clip((data_np / max_value) * 255, 0, 255).astype(np.uint8)
                
                return normalized
            
            # Apply UNIVERSAL physics-based normalization to all slices
            # Both targets and predictions now use the same universal normalization approach
            # TARGETS: Ground truth with discrete values - same universal normalization
            target_xy_abs_norm = universal_physics_normalization(target_xy_abs, "absorption")
            target_xy_scat_norm = universal_physics_normalization(target_xy_scat, "scattering")
            target_xz_abs_norm = universal_physics_normalization(target_xz_abs, "absorption")
            target_xz_scat_norm = universal_physics_normalization(target_xz_scat, "scattering")
            target_yz_abs_norm = universal_physics_normalization(target_yz_abs, "absorption")
            target_yz_scat_norm = universal_physics_normalization(target_yz_scat, "scattering")
            
            # PREDICTIONS: Continuous values - same universal normalization (consistency!)
            pred_xy_abs_norm = universal_physics_normalization(pred_xy_abs, "absorption")
            pred_xy_scat_norm = universal_physics_normalization(pred_xy_scat, "scattering")
            pred_xz_abs_norm = universal_physics_normalization(pred_xz_abs, "absorption")
            pred_xz_scat_norm = universal_physics_normalization(pred_xz_scat, "scattering")
            pred_yz_abs_norm = universal_physics_normalization(pred_yz_abs, "absorption")
            pred_yz_scat_norm = universal_physics_normalization(pred_yz_scat, "scattering")
            
            # Calculate value ranges for informative captions
            pred_xy_abs_range = f"[{pred_xy_abs.min():.5f}, {pred_xy_abs.max():.5f}]"
            target_xy_abs_range = f"[{target_xy_abs.min():.5f}, {target_xy_abs.max():.5f}]"
            pred_xy_scat_range = f"[{pred_xy_scat.min():.5f}, {pred_xy_scat.max():.5f}]"
            target_xy_scat_range = f"[{target_xy_scat.min():.5f}, {target_xy_scat.max():.5f}]"
            
            # Log to W&B with phantom-specific naming and universal physics normalization info
            phantom_logs = {
                # Absorption channel (μₐ) - All 3 slices for this phantom
                f"{prefix}/Absorption/{phantom_label}/predicted_xy_slice": wandb.Image(pred_xy_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ XY (z=32) | Range: {pred_xy_abs_range} | 🎯 Universal Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/target_xy_slice": wandb.Image(target_xy_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ XY (z=32) | Range: {target_xy_abs_range} | 🎯 Universal Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/predicted_xz_slice": wandb.Image(pred_xz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ XZ (y=32) | Range: [{pred_xz_abs.min():.5f}, {pred_xz_abs.max():.5f}] | 🎯 Universal Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/target_xz_slice": wandb.Image(target_xz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ XZ (y=32) | Range: [{target_xz_abs.min():.5f}, {target_xz_abs.max():.5f}] | 🎯 Universal Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/predicted_yz_slice": wandb.Image(pred_yz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ YZ (x=32) | Range: [{pred_yz_abs.min():.5f}, {pred_yz_abs.max():.5f}] | 🎯 Universal Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/target_yz_slice": wandb.Image(target_yz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ YZ (x=32) | Range: [{target_yz_abs.min():.5f}, {target_yz_abs.max():.5f}] | 🎯 Universal Physics Norm [0→0.0245]"),
                
                # Scattering channel (μ′s) - All 3 slices for this phantom
                f"{prefix}/Scattering/{phantom_label}/predicted_xy_slice": wandb.Image(pred_xy_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′s XY (z=32) | Range: {pred_xy_scat_range} | 🎯 Universal Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/target_xy_slice": wandb.Image(target_xy_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′s XY (z=32) | Range: {target_xy_scat_range} | 🎯 Universal Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/predicted_xz_slice": wandb.Image(pred_xz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′s XZ (y=32) | Range: [{pred_xz_scat.min():.5f}, {pred_xz_scat.max():.5f}] | 🎯 Universal Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/target_xz_slice": wandb.Image(target_xz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′s XZ (y=32) | Range: [{target_xz_scat.min():.5f}, {target_xz_scat.max():.5f}] | 🎯 Universal Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/predicted_yz_slice": wandb.Image(pred_yz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′s YZ (x=32) | Range: [{pred_yz_scat.min():.5f}, {pred_yz_scat.max():.5f}] | 🎯 Universal Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/target_yz_slice": wandb.Image(target_yz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′s YZ (x=32) | Range: [{target_yz_scat.min():.5f}, {target_yz_scat.max():.5f}] | 🎯 Universal Physics Norm [0→2.95]"),
            }
            
            # Log this phantom's images without step parameter (consistent with main metrics)
            # Include epoch info in the logged data for proper x-axis alignment
            phantom_logs["epoch"] = epoch + 1
            wandb.log(phantom_logs)
        
        logger.debug(f"✅ Successfully logged reconstruction images for {num_phantoms_to_show} phantoms at epoch {epoch}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to log reconstruction images: {e}")
        logger.debug(f"Error details: {str(e)}")
