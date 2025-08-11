#!/usr/bin/env python3
"""
Shared Visualization Utilities for NIR-DOT Training.

This module provides shared visualization functions used across both Stage 1 and Stage 2
training to avoid code duplication and ensure consistent visualization behavior.

Functions:
    log_reconstruction_images_to_wandb: Shared function for logging 3D reconstruction slices

Author: Max Hart
Date: August 2025
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
            
            # Enhanced normalization with better contrast preservation
            def normalize_with_tissue_tumor_contrast(data, is_prediction=False):
                """
                Enhanced normalization specifically designed for NIR-DOT optical properties.
                
                Key insight: We have 3 discrete classes (air=0, tissue, tumor) with known ranges.
                For predictions, we want to enhance the tissue-tumor contrast while preserving
                the physical meaning of optical property values.
                
                Args:
                    data: 2D slice of optical properties
                    is_prediction: If True, enhance contrast for continuous predictions
                                 If False, preserve discrete ground truth structure
                """
                data_np = data.cpu().numpy() if hasattr(data, 'cpu') else data
                
                if is_prediction:
                    # For predictions: Use sophisticated contrast enhancement
                    # that emphasizes tissue-tumor boundaries
                    
                    # Separate air (≈0) from tissue regions
                    air_mask = data_np <= 0.001
                    tissue_mask = data_np > 0.001
                    
                    if not tissue_mask.any():
                        # All air - just return zeros
                        return np.zeros_like(data_np, dtype=np.uint8)
                    
                    # Focus on tissue region enhancement
                    tissue_values = data_np[tissue_mask]
                    
                    if len(tissue_values) == 0:
                        return np.zeros_like(data_np, dtype=np.uint8)
                    
                    # Use robust statistics for tissue region
                    tissue_min = np.percentile(tissue_values, 5)   # Bottom 5% of tissue
                    tissue_max = np.percentile(tissue_values, 95)  # Top 5% of tissue
                    
                    # Create enhanced contrast mapping
                    normalized = np.zeros_like(data_np)
                    
                    # Air stays black (0)
                    normalized[air_mask] = 0
                    
                    # Tissue region gets enhanced contrast mapping
                    if tissue_max > tissue_min:
                        # Non-linear mapping to emphasize tumor regions
                        tissue_normalized = (data_np[tissue_mask] - tissue_min) / (tissue_max - tissue_min)
                        tissue_normalized = np.clip(tissue_normalized, 0, 1)
                        
                        # Apply gamma correction to enhance tumor contrast
                        # Lower gamma (0.7) brightens mid-tones (potential tumors)
                        tissue_normalized = np.power(tissue_normalized, 0.7)
                        
                        # Map to 50-255 range (keep 0-49 for air)
                        normalized[tissue_mask] = (tissue_normalized * 205 + 50).astype(np.uint8)
                    else:
                        # Uniform tissue - use medium gray
                        normalized[tissue_mask] = 128
                        
                else:
                    # For ground truth: Map optical property strength to brightness
                    # CRITICAL FIX: Sort by optical property values, not detection order!
                    normalized = np.zeros_like(data_np, dtype=np.uint8)
                    
                    # Air = Black (0)
                    air_mask = data_np == 0.0
                    normalized[air_mask] = 0
                    
                    # Find tissue and tumor values - SORT BY OPTICAL PROPERTY VALUE
                    unique_vals = np.unique(data_np[data_np > 0])
                    unique_vals = np.sort(unique_vals)  # CRITICAL: Sort by strength, not detection order
                    
                    if len(unique_vals) >= 1:
                        # Lowest value = Healthy tissue = Medium gray (128)
                        tissue_val = unique_vals[0]  # Weakest absorption = healthy tissue
                        tissue_mask = np.abs(data_np - tissue_val) < 1e-6
                        normalized[tissue_mask] = 128
                    
                    # Map remaining values (tumors) based on optical property strength
                    tumor_intensities = [255, 215, 175, 135, 95]  # Brightest to darkest
                    for i, tumor_val in enumerate(unique_vals[1:]):  # Skip tissue (index 0)
                        tumor_mask = np.abs(data_np - tumor_val) < 1e-6
                        # Higher optical property value = Higher intensity (brighter)
                        intensity_idx = min(i, len(tumor_intensities) - 1)
                        
                        # Map tumors by strength: Strongest tumor gets brightest intensity
                        # Since unique_vals is sorted ascending, strongest tumor is at the end
                        tumor_rank = len(unique_vals[1:]) - 1 - i  # Reverse index for strongest first
                        actual_intensity = tumor_intensities[min(tumor_rank, len(tumor_intensities) - 1)]
                        normalized[tumor_mask] = actual_intensity
                
                return normalized
            
            # Apply physics-aware normalization to all slices
            # Ground truth slices (discrete values: air=0, tissue, tumor)
            target_xy_abs_norm = normalize_with_tissue_tumor_contrast(target_xy_abs, is_prediction=False)
            target_xy_scat_norm = normalize_with_tissue_tumor_contrast(target_xy_scat, is_prediction=False)
            target_xz_abs_norm = normalize_with_tissue_tumor_contrast(target_xz_abs, is_prediction=False)
            target_xz_scat_norm = normalize_with_tissue_tumor_contrast(target_xz_scat, is_prediction=False)
            target_yz_abs_norm = normalize_with_tissue_tumor_contrast(target_yz_abs, is_prediction=False)
            target_yz_scat_norm = normalize_with_tissue_tumor_contrast(target_yz_scat, is_prediction=False)
            
            # Prediction slices (continuous values: enhanced contrast for tumor detection)
            pred_xy_abs_norm = normalize_with_tissue_tumor_contrast(pred_xy_abs, is_prediction=True)
            pred_xy_scat_norm = normalize_with_tissue_tumor_contrast(pred_xy_scat, is_prediction=True)
            pred_xz_abs_norm = normalize_with_tissue_tumor_contrast(pred_xz_abs, is_prediction=True)
            pred_xz_scat_norm = normalize_with_tissue_tumor_contrast(pred_xz_scat, is_prediction=True)
            pred_yz_abs_norm = normalize_with_tissue_tumor_contrast(pred_yz_abs, is_prediction=True)
            pred_yz_scat_norm = normalize_with_tissue_tumor_contrast(pred_yz_scat, is_prediction=True)
            
            # Calculate value ranges for informative captions
            pred_xy_abs_range = f"[{pred_xy_abs.min():.5f}, {pred_xy_abs.max():.5f}]"
            target_xy_abs_range = f"[{target_xy_abs.min():.5f}, {target_xy_abs.max():.5f}]"
            pred_xy_scat_range = f"[{pred_xy_scat.min():.5f}, {pred_xy_scat.max():.5f}]"
            target_xy_scat_range = f"[{target_xy_scat.min():.5f}, {target_xy_scat.max():.5f}]"
            
            # Log to W&B with phantom-specific naming and value range information
            phantom_logs = {
                # Absorption channel (μₐ) - All 3 slices for this phantom
                f"{prefix}/Absorption/{phantom_label}/predicted_xy_slice": wandb.Image(pred_xy_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ XY (z=32) | Range: {pred_xy_abs_range}"),
                f"{prefix}/Absorption/{phantom_label}/target_xy_slice": wandb.Image(target_xy_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ XY (z=32) | Range: {target_xy_abs_range}"),
                f"{prefix}/Absorption/{phantom_label}/predicted_xz_slice": wandb.Image(pred_xz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ XZ (y=32) | Range: [{pred_xz_abs.min():.5f}, {pred_xz_abs.max():.5f}]"),
                f"{prefix}/Absorption/{phantom_label}/target_xz_slice": wandb.Image(target_xz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ XZ (y=32) | Range: [{target_xz_abs.min():.5f}, {target_xz_abs.max():.5f}]"),
                f"{prefix}/Absorption/{phantom_label}/predicted_yz_slice": wandb.Image(pred_yz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μₐ YZ (x=32) | Range: [{pred_yz_abs.min():.5f}, {pred_yz_abs.max():.5f}]"),
                f"{prefix}/Absorption/{phantom_label}/target_yz_slice": wandb.Image(target_yz_abs_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μₐ YZ (x=32) | Range: [{target_yz_abs.min():.5f}, {target_yz_abs.max():.5f}]"),
                
                # Scattering channel (μ′s) - All 3 slices for this phantom
                f"{prefix}/Scattering/{phantom_label}/predicted_xy_slice": wandb.Image(pred_xy_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′s XY (z=32) | Range: {pred_xy_scat_range}"),
                f"{prefix}/Scattering/{phantom_label}/target_xy_slice": wandb.Image(target_xy_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′s XY (z=32) | Range: {target_xy_scat_range}"),
                f"{prefix}/Scattering/{phantom_label}/predicted_xz_slice": wandb.Image(pred_xz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′s XZ (y=32) | Range: [{pred_xz_scat.min():.5f}, {pred_xz_scat.max():.5f}]"),
                f"{prefix}/Scattering/{phantom_label}/target_xz_slice": wandb.Image(target_xz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′s XZ (y=32) | Range: [{target_xz_scat.min():.5f}, {target_xz_scat.max():.5f}]"),
                f"{prefix}/Scattering/{phantom_label}/predicted_yz_slice": wandb.Image(pred_yz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Predicted μ′s YZ (x=32) | Range: [{pred_yz_scat.min():.5f}, {pred_yz_scat.max():.5f}]"),
                f"{prefix}/Scattering/{phantom_label}/target_yz_slice": wandb.Image(target_yz_scat_norm, 
                    caption=f"Epoch {epoch + 1} - {phantom_label} Ground Truth μ′s YZ (x=32) | Range: [{target_yz_scat.min():.5f}, {target_yz_scat.max():.5f}]"),
            }
            
            # Log this phantom's images with step=epoch for correct slider behavior
            wandb.log(phantom_logs, step=epoch)
        
        # Log epoch information once with matching step
        wandb.log({"epoch": epoch + 1}, step=epoch)
        
        logger.debug(f"✅ Successfully logged reconstruction images for {num_phantoms_to_show} phantoms at epoch {epoch}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to log reconstruction images: {e}")
        logger.debug(f"Error details: {str(e)}")
