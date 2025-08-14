#!/usr/bin/env python3
"""
Standardization Utilities for NIR-DOT Ground Truth Normalization.

This module provides standardization utilities for normalizing ground truth
volumes in NIR-DOT reconstruction. The per-channel z-score normalization
ensures stable training by standardizing μₐ and μ′ₛ channels independently.

The standardizer is designed to:
- Fit normalization parameters only on training data
- Transform ground truth during training and validation
- Inverse-transform for human-interpretable metrics
- Persist normalization statistics for Stage 2 reuse

Classes:
    PerChannelZScore: Per-channel z-score normalization for 3D volumes

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
from typing import Dict, Tuple, Optional
import pickle

# Third-party imports
import torch
import numpy as np

# Project imports
from code.utils.logging_config import get_training_logger

# =============================================================================
# CONSTANTS
# =============================================================================

# Normalization constants
EPSILON = 1e-8                          # Small epsilon for numerical stability
NUM_CHANNELS = 2                        # μₐ and μ′ₛ channels

# Channel names for logging
ABSORPTION_CHANNEL = "mu_a"             # μₐ (absorption coefficient)
SCATTERING_CHANNEL = "mu_s_prime"       # μ′ₛ (reduced scattering coefficient)
CHANNEL_NAMES = [ABSORPTION_CHANNEL, SCATTERING_CHANNEL]

# Initialize module logger
logger = get_training_logger(__name__)

# =============================================================================
# PER-CHANNEL Z-SCORE STANDARDIZER
# =============================================================================

class PerChannelZScore:
    """
    Per-channel z-score normalization for 3D volumetric data.
    
    This class implements channel-wise standardization for NIR-DOT ground truth
    volumes, where each optical property (μₐ, μ′ₛ) is normalized independently
    using z-score standardization: (x - mean) / std.
    
    The standardizer:
    - Computes mean and std per channel from training data only
    - Applies standardization consistently across train/val splits
    - Provides inverse transformation for metric computation
    - Persists normalization statistics for later reuse
    
    Expected input shape: [batch_size, num_channels, depth, height, width]
    where num_channels = 2 (μₐ, μ′ₛ)
    
    Attributes:
        means (torch.Tensor): Per-channel means [num_channels]
        stds (torch.Tensor): Per-channel standard deviations [num_channels]
        fitted (bool): Whether the standardizer has been fitted to data
        device (torch.device): Device for tensor operations
    
    Example:
        >>> standardizer = PerChannelZScore()
        >>> standardizer.fit(train_targets)  # Shape: [N, 2, 64, 64, 64]
        >>> 
        >>> # During training
        >>> std_targets = standardizer.transform(targets)
        >>> predictions = model(inputs)  # Predict in standardized space
        >>> loss = criterion(predictions, std_targets)
        >>> 
        >>> # During validation  
        >>> raw_predictions = standardizer.inverse_transform(predictions)
        >>> raw_targets = targets  # Keep original for metrics
        >>> metrics = compute_metrics(raw_predictions, raw_targets)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the per-channel z-score standardizer.
        
        Args:
            device (Optional[torch.device]): Device for tensor operations.
                If None, uses CPU by default.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.means = None
        self.stds = None
        self.fitted = False
        
        logger.debug(f"🔧 Initialized PerChannelZScore standardizer on device: {self.device}")
    
    def fit(self, volumes: torch.Tensor) -> 'PerChannelZScore':
        """
        Fit the standardizer to training data by computing per-channel statistics.
        
        This method computes the mean and standard deviation for each channel
        across all spatial dimensions and batch samples. Statistics are computed
        only once and should be fitted on training data only.
        
        Args:
            volumes (torch.Tensor): Training volumes of shape [N, 2, D, H, W]
                where N is the number of samples, 2 is number of channels (μₐ, μ′ₛ),
                and D, H, W are spatial dimensions.
        
        Returns:
            PerChannelZScore: Self for method chaining
        
        Raises:
            ValueError: If input tensor has incorrect shape or contains invalid values
        """
        if len(volumes.shape) != 5:
            raise ValueError(f"Expected 5D tensor [N, C, D, H, W], got shape {volumes.shape}")
        
        if volumes.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected {NUM_CHANNELS} channels, got {volumes.shape[1]}")
        
        # Move to correct device
        volumes = volumes.to(self.device)
        
        # Compute statistics per channel across all spatial dimensions and samples
        # Shape: [N, 2, D, H, W] -> compute over dimensions [0, 2, 3, 4] -> [2]
        self.means = torch.mean(volumes, dim=[0, 2, 3, 4])  # Shape: [2]
        self.stds = torch.std(volumes, dim=[0, 2, 3, 4])    # Shape: [2]
        
        # Ensure std is not zero (add small epsilon if needed)
        self.stds = torch.clamp(self.stds, min=EPSILON)
        
        self.fitted = True
        
        # Log normalization statistics
        logger.info("📊 Fitted per-channel standardization statistics:")
        for i, (channel_name, mean_val, std_val) in enumerate(zip(CHANNEL_NAMES, self.means, self.stds)):
            logger.info(f"   Channel {i} ({channel_name}): mean={mean_val:.6f}, std={std_val:.6f}")
        
        return self
    
    def transform(self, volumes: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score standardization to volumes.
        
        Transforms input volumes using the fitted statistics:
        standardized = (volumes - mean) / std
        
        Args:
            volumes (torch.Tensor): Volumes to standardize [N, 2, D, H, W]
        
        Returns:
            torch.Tensor: Standardized volumes with same shape as input
        
        Raises:
            RuntimeError: If standardizer has not been fitted
            ValueError: If input tensor has incorrect shape
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before transforming data")
        
        if len(volumes.shape) != 5 or volumes.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected shape [N, {NUM_CHANNELS}, D, H, W], got {volumes.shape}")
        
        # Move to correct device
        volumes = volumes.to(self.device)
        
        # Apply per-channel standardization
        # Broadcasting: [N, 2, D, H, W] - [2] -> [N, 2, D, H, W]
        means = self.means.view(1, NUM_CHANNELS, 1, 1, 1)  # Shape: [1, 2, 1, 1, 1]
        stds = self.stds.view(1, NUM_CHANNELS, 1, 1, 1)    # Shape: [1, 2, 1, 1, 1]
        
        standardized = (volumes - means) / stds
        
        return standardized
    
    def inverse_transform(self, standardized_volumes: torch.Tensor) -> torch.Tensor:
        """
        Inverse z-score standardization to recover original scale.
        
        Transforms standardized volumes back to original scale:
        original = standardized * std + mean
        
        Args:
            standardized_volumes (torch.Tensor): Standardized volumes [N, 2, D, H, W]
        
        Returns:
            torch.Tensor: Volumes in original scale with same shape as input
        
        Raises:
            RuntimeError: If standardizer has not been fitted
            ValueError: If input tensor has incorrect shape
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before inverse transforming data")
        
        if len(standardized_volumes.shape) != 5 or standardized_volumes.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected shape [N, {NUM_CHANNELS}, D, H, W], got {standardized_volumes.shape}")
        
        # Move to correct device
        standardized_volumes = standardized_volumes.to(self.device)
        
        # Apply inverse per-channel standardization
        # Broadcasting: [N, 2, D, H, W] * [2] + [2] -> [N, 2, D, H, W]
        means = self.means.view(1, NUM_CHANNELS, 1, 1, 1)  # Shape: [1, 2, 1, 1, 1]
        stds = self.stds.view(1, NUM_CHANNELS, 1, 1, 1)    # Shape: [1, 2, 1, 1, 1]
        
        original = standardized_volumes * stds + means
        
        return original
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return state dictionary for saving standardizer parameters.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing normalization statistics
        
        Raises:
            RuntimeError: If standardizer has not been fitted
        """
        if not self.fitted:
            raise RuntimeError("Cannot save state of unfitted standardizer")
        
        return {
            'means': self.means,
            'stds': self.stds,
            'fitted': torch.tensor(self.fitted)
        }
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> 'PerChannelZScore':
        """
        Load state dictionary to restore standardizer parameters.
        
        Args:
            state_dict (Dict[str, torch.Tensor]): Dictionary containing normalization statistics
        
        Returns:
            PerChannelZScore: Self for method chaining
        
        Raises:
            KeyError: If required keys are missing from state_dict
        """
        required_keys = ['means', 'stds', 'fitted']
        for key in required_keys:
            if key not in state_dict:
                raise KeyError(f"Missing required key '{key}' in state_dict")
        
        self.means = state_dict['means'].to(self.device)
        self.stds = state_dict['stds'].to(self.device)
        self.fitted = bool(state_dict['fitted'].item())
        
        logger.info("✅ Loaded standardizer state:")
        for i, (channel_name, mean_val, std_val) in enumerate(zip(CHANNEL_NAMES, self.means, self.stds)):
            logger.info(f"   Channel {i} ({channel_name}): mean={mean_val:.6f}, std={std_val:.6f}")
        
        return self
    
    def save(self, path: str) -> None:
        """
        Save standardizer to disk.
        
        Args:
            path (str): File path to save standardizer
        
        Raises:
            RuntimeError: If standardizer has not been fitted
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted standardizer")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        
        logger.info(f"💾 Saved standardizer to: {path}")
    
    def load(self, path: str) -> 'PerChannelZScore':
        """
        Load standardizer from disk.
        
        Args:
            path (str): File path to load standardizer from
        
        Returns:
            PerChannelZScore: Self for method chaining
        
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Standardizer file not found: {path}")
        
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        
        logger.info(f"📂 Loaded standardizer from: {path}")
        
        return self
    
    def to(self, device: torch.device) -> 'PerChannelZScore':
        """
        Move standardizer to specified device.
        
        Args:
            device (torch.device): Target device
        
        Returns:
            PerChannelZScore: Self for method chaining
        """
        self.device = device
        if self.fitted:
            self.means = self.means.to(device)
            self.stds = self.stds.to(device)
        
        return self
    
    def __repr__(self) -> str:
        """String representation of the standardizer."""
        if self.fitted:
            return (f"PerChannelZScore(fitted=True, device={self.device}, "
                   f"means={self.means.tolist()}, stds={self.stds.tolist()})")
        else:
            return f"PerChannelZScore(fitted=False, device={self.device})"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_standardizer(device: Optional[torch.device] = None) -> PerChannelZScore:
    """
    Create a new per-channel z-score standardizer.
    
    Args:
        device (Optional[torch.device]): Device for tensor operations
    
    Returns:
        PerChannelZScore: New standardizer instance
    """
    return PerChannelZScore(device=device)


def fit_standardizer_on_dataloader(dataloader, device: Optional[torch.device] = None) -> PerChannelZScore:
    """
    Fit a standardizer on all ground truth volumes in a dataloader.
    
    This utility function creates a standardizer and fits it on all ground truth
    volumes from a training dataloader. It accumulates all volumes in memory
    before fitting, so use with care for large datasets.
    
    Args:
        dataloader: PyTorch DataLoader containing ground truth volumes
        device (Optional[torch.device]): Device for tensor operations
    
    Returns:
        PerChannelZScore: Fitted standardizer
    
    Note:
        This function assumes dataloader returns batches with 'ground_truth' key
        or tuples/lists where ground truth is at a specific index.
    """
    logger.info("🔧 Fitting standardizer on training dataloader...")
    
    standardizer = PerChannelZScore(device=device)
    all_volumes = []
    
    # Collect all volumes from dataloader
    for batch_idx, batch in enumerate(dataloader):
        # Handle different batch formats
        if isinstance(batch, dict):
            if 'ground_truth' in batch:
                volumes = batch['ground_truth']
            elif 'targets' in batch:
                volumes = batch['targets']
            else:
                raise KeyError("Could not find ground truth in batch dict. Expected 'ground_truth' or 'targets' key.")
        elif isinstance(batch, (list, tuple)):
            # Assume ground truth is second element (inputs, targets)
            volumes = batch[1] if len(batch) > 1 else batch[0]
        else:
            # Assume batch is directly the volumes
            volumes = batch
        
        all_volumes.append(volumes.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"   Processed {batch_idx + 1} batches...")
    
    # Concatenate all volumes and fit
    all_volumes = torch.cat(all_volumes, dim=0)
    logger.info(f"📊 Fitting standardizer on {all_volumes.shape[0]} volumes...")
    
    standardizer.fit(all_volumes)
    
    logger.info("✅ Standardizer fitting complete!")
    return standardizer


if __name__ == "__main__":
    # Simple test
    logger.info("🧪 Testing PerChannelZScore standardizer...")
    
    # Create synthetic test data
    torch.manual_seed(42)
    test_data = torch.randn(10, 2, 8, 8, 8) * 2 + 1  # [N, C, D, H, W]
    
    # Test standardizer
    standardizer = PerChannelZScore()
    standardizer.fit(test_data)
    
    # Transform and inverse transform
    standardized = standardizer.transform(test_data)
    recovered = standardizer.inverse_transform(standardized)
    
    # Check recovery accuracy
    recovery_error = torch.mean(torch.abs(test_data - recovered))
    logger.info(f"Recovery error: {recovery_error:.8f} (should be ~0)")
    
    # Check standardization properties
    std_means = torch.mean(standardized, dim=[0, 2, 3, 4])
    std_stds = torch.std(standardized, dim=[0, 2, 3, 4])
    logger.info(f"Standardized means: {std_means} (should be ~0)")
    logger.info(f"Standardized stds: {std_stds} (should be ~1)")
    
    logger.info("✅ Standardizer test completed!")
