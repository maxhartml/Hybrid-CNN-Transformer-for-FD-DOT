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
                                     phantom_ids: Optional[np.ndarray] = None,
                                     gt_standardizer=None,   # NEW: Stage-1 ground truth standardizer
                                     add_autocontrast_preview: bool = True) -> None:  # NEW: Debug autocontrast
    """
    🎯 PHYSICS-AWARE RECONSTRUCTION VISUALIZATION WITH INVERSE-STANDARDIZATION 🎯
    
    Enhanced function to log 3D reconstruction slices to W&B for visualization.
    This function guarantees that images show RAW PHYSICAL UNITS by applying
    inverse standardization before rendering, ensuring W&B visualizations match
    improving validation metrics.
    
    CRITICAL FIXES:
    ✅ Inverse-standardization: Convert from z-scored to raw physics values
    ✅ Correct slice indexing: [B, 2, D, H, W] → proper XY/XZ/YZ slices  
    ✅ Physics range clamping: Prevent negative/out-of-range values
    ✅ Sanity metrics: Per-channel min/max/percentiles logged to W&B
    ✅ Autocontrast preview: Debug fallback for tiny physics values
    
    Args:
        predictions (torch.Tensor): Predicted volumes [batch, 2, D, H, W]
        targets (torch.Tensor): Target volumes [batch, 2, D, H, W]
        epoch (int): Current epoch number
        prefix (str): Prefix for W&B logging keys. Default: "Reconstructions"
        step (Optional[int]): Optional step number for logging
        phantom_ids (Optional[np.ndarray]): Array of phantom IDs from validation batch
        gt_standardizer: Stage-1 ground truth standardizer for inverse transform
        add_autocontrast_preview (bool): Add debug autocontrast images
    """
    try:
        # 0) BASIC VALIDATION
        assert predictions.ndim == 5 and targets.ndim == 5, f"Expected [B, 2, D, H, W], got pred: {predictions.shape}, tgt: {targets.shape}"
        assert predictions.shape[1] == 2 and targets.shape[1] == 2, "Expected 2 channels (mu_a, mu_s')"
        
        # 1) INVERSE-STANDARDIZE TO RAW PHYSICS UNITS (CRITICAL FIX!)
        # This ensures W&B visualizations show actual physics values, not z-scored garbage
        if gt_standardizer is not None and hasattr(gt_standardizer, "inverse_transform"):
            logger.debug("🔧 Applying inverse standardization to convert to raw physics units")
            with torch.no_grad():
                predictions = gt_standardizer.inverse_transform(predictions)
                targets = gt_standardizer.inverse_transform(targets)
        elif gt_standardizer is not None and hasattr(gt_standardizer, "inverse_transform_ground_truth"):
            logger.debug("🔧 Applying inverse_transform_ground_truth to convert to raw physics units")
            with torch.no_grad():
                predictions = gt_standardizer.inverse_transform_ground_truth(predictions)
                targets = gt_standardizer.inverse_transform_ground_truth(targets)
        else:
            logger.warning("⚠️ No gt_standardizer provided - assuming inputs are already in raw physics units")
        
        # 2) CLAMP TO VALID PHYSICS RANGES (CRITICAL FOR VISUALIZATION!)
        # Prevent negative/out-of-range optical properties that break visualization
        mu_a_max = 0.0245   # Maximum absorption: 0.007 (tumor baseline) × 3.5 (max enhancement)
        mu_s_max = 2.95     # Maximum scattering: 1.18 (tumor baseline) × 2.5 (max enhancement)
        
        predictions = predictions.clone()
        targets = targets.clone()
        predictions[:, 0].clamp_(0.0, mu_a_max)   # μₐ channel
        predictions[:, 1].clamp_(0.0, mu_s_max)   # μ′ₛ channel
        targets[:, 0].clamp_(0.0, mu_a_max)       # μₐ channel
        targets[:, 1].clamp_(0.0, mu_s_max)       # μ′ₛ channel
        
        # 3) COMPUTE SANITY METRICS FOR W&B LOGGING
        # These scalars help debug if inverse-standardization worked correctly
        with torch.no_grad():
            # Per-channel statistics for predictions (raw physics)
            pred_mu_a = predictions[:, 0]  # All μₐ values in batch
            pred_mu_s = predictions[:, 1]  # All μ′ₛ values in batch
            tgt_mu_a = targets[:, 0]
            tgt_mu_s = targets[:, 1]
            
            # Compute percentiles for robust statistics
            sanity_metrics = {
                "viz_stats/pred_mu_a_min": pred_mu_a.min().item(),
                "viz_stats/pred_mu_a_max": pred_mu_a.max().item(),
                "viz_stats/pred_mu_a_p1": torch.quantile(pred_mu_a, 0.01).item(),
                "viz_stats/pred_mu_a_p50": torch.quantile(pred_mu_a, 0.50).item(),
                "viz_stats/pred_mu_a_p99": torch.quantile(pred_mu_a, 0.99).item(),
                
                "viz_stats/pred_mu_s_min": pred_mu_s.min().item(),
                "viz_stats/pred_mu_s_max": pred_mu_s.max().item(),
                "viz_stats/pred_mu_s_p1": torch.quantile(pred_mu_s, 0.01).item(),
                "viz_stats/pred_mu_s_p50": torch.quantile(pred_mu_s, 0.50).item(),
                "viz_stats/pred_mu_s_p99": torch.quantile(pred_mu_s, 0.99).item(),
                
                "viz_stats/tgt_mu_a_min": tgt_mu_a.min().item(),
                "viz_stats/tgt_mu_a_max": tgt_mu_a.max().item(),
                "viz_stats/tgt_mu_a_p1": torch.quantile(tgt_mu_a, 0.01).item(),
                "viz_stats/tgt_mu_a_p50": torch.quantile(tgt_mu_a, 0.50).item(),
                "viz_stats/tgt_mu_a_p99": torch.quantile(tgt_mu_a, 0.99).item(),
                
                "viz_stats/tgt_mu_s_min": tgt_mu_s.min().item(),
                "viz_stats/tgt_mu_s_max": tgt_mu_s.max().item(),
                "viz_stats/tgt_mu_s_p1": torch.quantile(tgt_mu_s, 0.01).item(),
                "viz_stats/tgt_mu_s_p50": torch.quantile(tgt_mu_s, 0.50).item(),
                "viz_stats/tgt_mu_s_p99": torch.quantile(tgt_mu_s, 0.99).item(),
            }
            
            # SANITY CHECK: Warn if physics values are suspiciously small
            if sanity_metrics["viz_stats/pred_mu_a_p99"] < 1e-5:
                logger.warning("⚠️ Suspiciously small μₐ values - check inverse standardization!")
            if sanity_metrics["viz_stats/pred_mu_s_p99"] < 0.05:
                logger.warning("⚠️ Suspiciously small μ′ₛ values - check inverse standardization!")
            
            # Log sanity metrics to W&B
            wandb.log(sanity_metrics, commit=False)
        
        # 4) EXTRACT SLICES FOR VISUALIZATION (FIXED INDEXING!)
        # Extract data for first TWO phantoms in batch for variety
        num_phantoms_to_show = min(2, predictions.shape[0])
        
        logger.debug(f"Logging images for {num_phantoms_to_show} phantoms - Pred shape: {predictions.shape}, Target shape: {targets.shape}")
        
        # We'll log images for each phantom separately
        for phantom_idx in range(num_phantoms_to_show):
            pred_phantom = predictions[phantom_idx].detach().cpu().numpy()  # Shape: [2, D, H, W]
            target_phantom = targets[phantom_idx].detach().cpu().numpy()
            
            # Get actual phantom ID if available, otherwise use generic numbering
            if phantom_ids is not None and phantom_idx < len(phantom_ids):
                actual_phantom_id = int(phantom_ids[phantom_idx])
                phantom_label = f"Phantom_{actual_phantom_id:03d}"  # e.g., "Phantom_001", "Phantom_452"
            else:
                phantom_label = f"Phantom_{phantom_idx + 1}"  # Fallback: "Phantom_1", "Phantom_2"
            
            # FIXED SLICE INDEXING FOR [2, D, H, W] VOLUMES
            C, D, H, W = pred_phantom.shape
            z = D // 2; y = H // 2; x = W // 2  # Middle slice indices
            
            # μₐ channel = 0, μ′ₛ channel = 1 
            # XY slices (slice along D dimension): vol[c, z, :, :]
            pred_xy_abs = pred_phantom[0, z, :, :]    # [H, W]
            tgt_xy_abs = target_phantom[0, z, :, :]
            pred_xy_scat = pred_phantom[1, z, :, :]
            tgt_xy_scat = target_phantom[1, z, :, :]
            
            # XZ slices (slice along H dimension): vol[c, :, y, :]  
            pred_xz_abs = pred_phantom[0, :, y, :]    # [D, W]
            tgt_xz_abs = target_phantom[0, :, y, :]
            pred_xz_scat = pred_phantom[1, :, y, :]
            tgt_xz_scat = target_phantom[1, :, y, :]
            
            # YZ slices (slice along W dimension): vol[c, :, :, x]
            pred_yz_abs = pred_phantom[0, :, :, x]    # [D, H]
            tgt_yz_abs = target_phantom[0, :, :, x]
            pred_yz_scat = pred_phantom[1, :, :, x]
            tgt_yz_scat = target_phantom[1, :, :, x]
            
            # 5) PHYSICS-BASED NORMALIZATION (PRESERVES SCIENTIFIC MEANING)
            def physics_norm(img2d: np.ndarray, channel: str) -> np.ndarray:
                """
                🎯 UNIVERSAL PHYSICS-BASED NORMALIZATION 🎯
                
                Revolutionary approach that preserves physical meaning across ALL phantoms:
                • Air (0.0) → Black (0) universally
                • Physics values → Consistent colors across phantoms
                • Same optical properties → Same visualization colors
                • Cross-phantom comparison becomes meaningful
                • Scientific accuracy preserved
                
                Args:
                    img2d: 2D slice of optical properties (any phantom)
                    channel: "absorption" or "scattering" to determine physical ranges
                
                Returns:
                    Normalized intensities [0-255] preserving universal physics relationships
                """
                if "absorption" in channel.lower():
                    # Full absorption range: 0 (air) to 0.0245 (strongest tumor: 0.007 × 3.5)
                    vmax = 0.0245
                else:
                    # Full scattering range: 0 (air) to 2.95 (strongest tumor: 1.18 × 2.5)  
                    vmax = 2.95
                
                # Universal linear mapping: physics_value → grayscale_intensity
                # Air (0) → Black (0), Max_possible → White (255)
                normalized = np.clip((img2d / vmax) * 255, 0, 255).astype(np.uint8)
                return normalized
            
            # 6) AUTOCONTRAST PREVIEW (DEBUG FAILSAFE)
            def autocontrast(img2d: np.ndarray) -> np.ndarray:
                """Autocontrast normalization for debugging tiny physics values."""
                p1, p99 = np.percentile(img2d, [1, 99])
                scaled = (img2d - p1) / (max(p99 - p1, 1e-8))
                return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
            
            # Apply PHYSICS-BASED normalization to all slices
            # TARGETS: Ground truth with discrete values
            tgt_xy_abs_norm = physics_norm(tgt_xy_abs, "absorption")
            tgt_xy_scat_norm = physics_norm(tgt_xy_scat, "scattering")
            tgt_xz_abs_norm = physics_norm(tgt_xz_abs, "absorption")
            tgt_xz_scat_norm = physics_norm(tgt_xz_scat, "scattering")
            tgt_yz_abs_norm = physics_norm(tgt_yz_abs, "absorption")
            tgt_yz_scat_norm = physics_norm(tgt_yz_scat, "scattering")
            
            # PREDICTIONS: Continuous values - same physics normalization
            pred_xy_abs_norm = physics_norm(pred_xy_abs, "absorption")
            pred_xy_scat_norm = physics_norm(pred_xy_scat, "scattering")
            pred_xz_abs_norm = physics_norm(pred_xz_abs, "absorption")
            pred_xz_scat_norm = physics_norm(pred_xz_scat, "scattering")
            pred_yz_abs_norm = physics_norm(pred_yz_abs, "absorption")
            pred_yz_scat_norm = physics_norm(pred_yz_scat, "scattering")
            
            # AUTOCONTRAST PREVIEWS (for debugging tiny values)
            if add_autocontrast_preview:
                # Generate autocontrast versions
                pred_xy_abs_auto = autocontrast(pred_xy_abs)
                tgt_xy_abs_auto = autocontrast(tgt_xy_abs)
                pred_xy_scat_auto = autocontrast(pred_xy_scat)
                tgt_xy_scat_auto = autocontrast(tgt_xy_scat)
            
            # Calculate value ranges for informative captions
            pred_xy_abs_range = f"[{pred_xy_abs.min():.5f}, {pred_xy_abs.max():.5f}]"
            tgt_xy_abs_range = f"[{tgt_xy_abs.min():.5f}, {tgt_xy_abs.max():.5f}]"
            pred_xy_scat_range = f"[{pred_xy_scat.min():.5f}, {pred_xy_scat.max():.5f}]"
            tgt_xy_scat_range = f"[{tgt_xy_scat.min():.5f}, {tgt_xy_scat.max():.5f}]"
            
                        
            # 7) LOG TO W&B WITH BOTH PHYSICS AND AUTOCONTRAST VERSIONS
            phantom_logs = {
                # ===== ABSORPTION CHANNEL (μₐ) - PHYSICS NORMALIZATION =====
                f"{prefix}/Absorption/{phantom_label}/predicted_xy_physics": wandb.Image(pred_xy_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ XY (z={z}) | Range: {pred_xy_abs_range} | 🎯 Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/target_xy_physics": wandb.Image(tgt_xy_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ XY (z={z}) | Range: {tgt_xy_abs_range} | 🎯 Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/predicted_xz_physics": wandb.Image(pred_xz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ XZ (y={y}) | Range: [{pred_xz_abs.min():.5f}, {pred_xz_abs.max():.5f}] | 🎯 Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/target_xz_physics": wandb.Image(tgt_xz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ XZ (y={y}) | Range: [{tgt_xz_abs.min():.5f}, {tgt_xz_abs.max():.5f}] | 🎯 Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/predicted_yz_physics": wandb.Image(pred_yz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ YZ (x={x}) | Range: [{pred_yz_abs.min():.5f}, {pred_yz_abs.max():.5f}] | 🎯 Physics Norm [0→0.0245]"),
                f"{prefix}/Absorption/{phantom_label}/target_yz_physics": wandb.Image(tgt_yz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ YZ (x={x}) | Range: [{tgt_yz_abs.min():.5f}, {tgt_yz_abs.max():.5f}] | 🎯 Physics Norm [0→0.0245]"),
                
                # ===== SCATTERING CHANNEL (μ′ₛ) - PHYSICS NORMALIZATION =====
                f"{prefix}/Scattering/{phantom_label}/predicted_xy_physics": wandb.Image(pred_xy_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′ₛ XY (z={z}) | Range: {pred_xy_scat_range} | 🎯 Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/target_xy_physics": wandb.Image(tgt_xy_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′ₛ XY (z={z}) | Range: {tgt_xy_scat_range} | 🎯 Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/predicted_xz_physics": wandb.Image(pred_xz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′ₛ XZ (y={y}) | Range: [{pred_xz_scat.min():.5f}, {pred_xz_scat.max():.5f}] | 🎯 Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/target_xz_physics": wandb.Image(tgt_xz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′ₛ XZ (y={y}) | Range: [{tgt_xz_scat.min():.5f}, {tgt_xz_scat.max():.5f}] | 🎯 Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/predicted_yz_physics": wandb.Image(pred_yz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′ₛ YZ (x={x}) | Range: [{pred_yz_scat.min():.5f}, {pred_yz_scat.max():.5f}] | 🎯 Physics Norm [0→2.95]"),
                f"{prefix}/Scattering/{phantom_label}/target_yz_physics": wandb.Image(tgt_yz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′ₛ YZ (x={x}) | Range: [{tgt_yz_scat.min():.5f}, {tgt_yz_scat.max():.5f}] | 🎯 Physics Norm [0→2.95]"),
            }
            
            # ===== AUTOCONTRAST PREVIEWS (DEBUG ONLY) =====
            if add_autocontrast_preview:
                phantom_logs.update({
                    f"{prefix}/Debug_Autocontrast/{phantom_label}/predicted_xy_autocontrast": wandb.Image(pred_xy_abs_auto, 
                        caption=f"Epoch {epoch + 1} - {phantom_label} DEBUG: Predicted μₐ XY (Autocontrast) | Range: {pred_xy_abs_range}"),
                    f"{prefix}/Debug_Autocontrast/{phantom_label}/target_xy_autocontrast": wandb.Image(tgt_xy_abs_auto, 
                        caption=f"Epoch {epoch + 1} - {phantom_label} DEBUG: Ground Truth μₐ XY (Autocontrast) | Range: {tgt_xy_abs_range}"),
                    f"{prefix}/Debug_Autocontrast/{phantom_label}/predicted_xy_scat_autocontrast": wandb.Image(pred_xy_scat_auto, 
                        caption=f"Epoch {epoch + 1} - {phantom_label} DEBUG: Predicted μ′ₛ XY (Autocontrast) | Range: {pred_xy_scat_range}"),
                    f"{prefix}/Debug_Autocontrast/{phantom_label}/target_xy_scat_autocontrast": wandb.Image(tgt_xy_scat_auto, 
                        caption=f"Epoch {epoch + 1} - {phantom_label} DEBUG: Ground Truth μ′ₛ XY (Autocontrast) | Range: {tgt_xy_scat_range}"),
                })
            
            # Log this phantom's images - use commit=False to batch with other metrics
            wandb.log(phantom_logs, commit=False)
        
        logger.debug(f"✅ Successfully logged reconstruction images for {num_phantoms_to_show} phantoms at epoch {epoch}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to log reconstruction images: {e}")
        logger.debug(f"Error details: {str(e)}")
