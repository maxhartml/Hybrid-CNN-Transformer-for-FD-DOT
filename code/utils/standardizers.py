#!/usr/bin/env python3
"""
Standardization Utilities for NIR-DOT Ground Truth and Input Normalization.

This module provides comprehensive standardization utilities for NIR-DOT reconstruction,
including ground truth volume normalization, NIR measurement standardization, 
position coordinate scaling, and tissue patch normalization.

The standardizers are designed to:
- Fit normalization parameters only on training data
- Transform inputs during training and validation consistently  
- Inverse-transform for human-interpretable metrics
- Persist normalization statistics for Stage 2 reuse
- Maintain consistency between Stage 1 and Stage 2 pipelines

Classes:
    PerChannelZScore: Per-channel z-score normalization for 3D volumes
    MeasurementStandardizer: Per-feature z-score normalization for NIR measurements
    PositionScaler: Min-max scaling for spatial coordinates to [-1, 1]
    TissuePatchStandardizer: Uses ground truth stats for tissue patch normalization

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
NIR_FEATURE_DIM = 2                     # [log_amplitude, phase] measurement features
POSITION_DIM = 6                        # [src_x, src_y, src_z, det_x, det_y, det_z]
NIR_INPUT_DIM = 8                       # Total NIR input dimension

# Standardization bounds
Z_SCORE_CLAMP_RANGE = (-5.0, 5.0)      # Clamp z-scored values to prevent extreme outliers

# Channel names for logging
ABSORPTION_CHANNEL = "mu_a"             # μₐ (absorption coefficient)
SCATTERING_CHANNEL = "mu_s_prime"       # μ′ₛ (reduced scattering coefficient)
CHANNEL_NAMES = [ABSORPTION_CHANNEL, SCATTERING_CHANNEL]

# NIR measurement feature names
MEASUREMENT_FEATURES = ["log_amplitude", "phase"]

# Position coordinate names  
POSITION_COORDINATES = ["src_x", "src_y", "src_z", "det_x", "det_y", "det_z"]

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
    
    def set_stats(self, mean, std):
        """
        Set pre-computed statistics for the standardizer.
        
        Args:
            mean (torch.Tensor): Per-channel means [2]
            std (torch.Tensor): Per-channel standard deviations [2]
        """
        self.means = mean.clone().to(self.device)
        self.stds = std.clone().to(self.device)
        self.fitted = True
    
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
# NIR MEASUREMENT STANDARDIZER
# =============================================================================

class MeasurementStandardizer:
    """
    Per-feature z-score normalization for NIR measurements.
    
    This class implements feature-wise standardization for NIR measurement data,
    where each measurement feature (log_amplitude, phase) is normalized independently
    using z-score standardization: (x - mean) / std.
    
    The standardizer:
    - Computes mean and std per feature from training data only
    - Applies standardization consistently across train/val/test splits
    - No inverse transform needed (measurements are inputs only)
    - Persists normalization statistics for Stage 2 reuse
    
    Expected input shape: [batch_size, n_measurements, 2]
    where features = [log_amplitude, phase]
    
    Attributes:
        means (torch.Tensor): Per-feature means [2]
        stds (torch.Tensor): Per-feature standard deviations [2]
        fitted (bool): Whether the standardizer has been fitted to data
        device (torch.device): Device for tensor operations
    
    Example:
        >>> standardizer = MeasurementStandardizer()
        >>> standardizer.fit(train_measurements)  # Shape: [N, n_meas, 2]
        >>> 
        >>> # During training/validation
        >>> std_measurements = standardizer.transform(measurements)
        >>> model_inputs = torch.cat([std_measurements, positions], dim=-1)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the measurement standardizer.
        
        Args:
            device (Optional[torch.device]): Device for tensor operations.
                If None, uses CPU by default.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.means = None
        self.stds = None
        self.fitted = False
        
        logger.debug(f"🔧 Initialized MeasurementStandardizer on device: {self.device}")
    
    def fit(self, measurements: torch.Tensor) -> 'MeasurementStandardizer':
        """
        Fit the standardizer to training measurement data.
        
        Args:
            measurements (torch.Tensor): Training measurements of shape [N, n_meas, 2]
                where features = [log_amplitude, phase]
        
        Returns:
            MeasurementStandardizer: Self for method chaining
        
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if len(measurements.shape) != 3:
            raise ValueError(f"Expected 3D tensor [N, n_meas, 2], got shape {measurements.shape}")
        
        if measurements.shape[2] != NIR_FEATURE_DIM:
            raise ValueError(f"Expected {NIR_FEATURE_DIM} features, got {measurements.shape[2]}")
        
        # Move to correct device
        measurements = measurements.to(self.device)
        
        # Compute statistics per feature across all samples and measurements
        # Shape: [N, n_meas, 2] -> compute over dimensions [0, 1] -> [2]
        self.means = torch.mean(measurements, dim=[0, 1])  # Shape: [2]
        self.stds = torch.std(measurements, dim=[0, 1])    # Shape: [2]
        
        # Ensure std is not zero (add small epsilon if needed)
        self.stds = torch.clamp(self.stds, min=EPSILON)
        
        self.fitted = True
        
        # Log normalization statistics
        logger.info("📊 Fitted measurement standardization statistics:")
        for i, (feature_name, mean_val, std_val) in enumerate(zip(MEASUREMENT_FEATURES, self.means, self.stds)):
            logger.info(f"   Feature {i} ({feature_name}): mean={mean_val:.6f}, std={std_val:.6f}")
        
        return self
    
    def transform(self, measurements: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """
        Apply z-score standardization to measurements.
        
        Args:
            measurements (torch.Tensor): Measurements to standardize [N, n_meas, 2]
            clamp (bool): Whether to clamp values to prevent extreme outliers
        
        Returns:
            torch.Tensor: Standardized measurements with same shape as input
        
        Raises:
            RuntimeError: If standardizer has not been fitted
            ValueError: If input tensor has incorrect shape
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before transforming data")
        
        if len(measurements.shape) != 3 or measurements.shape[2] != NIR_FEATURE_DIM:
            raise ValueError(f"Expected shape [N, n_meas, {NIR_FEATURE_DIM}], got {measurements.shape}")
        
        # Move to correct device
        measurements = measurements.to(self.device)
        
        # Apply per-feature standardization
        # Broadcasting: [N, n_meas, 2] - [2] -> [N, n_meas, 2]
        means = self.means.view(1, 1, NIR_FEATURE_DIM)  # Shape: [1, 1, 2]
        stds = self.stds.view(1, 1, NIR_FEATURE_DIM)    # Shape: [1, 1, 2]
        
        standardized = (measurements - means) / stds
        
        # Optional clamping to prevent extreme outliers from destabilizing AMP
        if clamp:
            standardized = torch.clamp(standardized, Z_SCORE_CLAMP_RANGE[0], Z_SCORE_CLAMP_RANGE[1])
        
        return standardized
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dictionary for saving standardizer parameters."""
        if not self.fitted:
            raise RuntimeError("Cannot save state of unfitted standardizer")
        
        return {
            'means': self.means,
            'stds': self.stds,
            'fitted': torch.tensor(self.fitted)
        }
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> 'MeasurementStandardizer':
        """Load state dictionary to restore standardizer parameters."""
        required_keys = ['means', 'stds', 'fitted']
        for key in required_keys:
            if key not in state_dict:
                raise KeyError(f"Missing required key '{key}' in state_dict")
        
        self.means = state_dict['means'].to(self.device)
        self.stds = state_dict['stds'].to(self.device)
        self.fitted = bool(state_dict['fitted'].item())
        
        logger.info("✅ Loaded measurement standardizer state:")
        for i, (feature_name, mean_val, std_val) in enumerate(zip(MEASUREMENT_FEATURES, self.means, self.stds)):
            logger.info(f"   Feature {i} ({feature_name}): mean={mean_val:.6f}, std={std_val:.6f}")
        
        return self
    
    def to(self, device: torch.device) -> 'MeasurementStandardizer':
        """Move standardizer to specified device."""
        self.device = device
        if self.fitted:
            self.means = self.means.to(device)
            self.stds = self.stds.to(device)
        return self
    
    def __repr__(self) -> str:
        """String representation of the standardizer."""
        if self.fitted:
            return (f"MeasurementStandardizer(fitted=True, device={self.device}, "
                   f"means={self.means.tolist()}, stds={self.stds.tolist()})")
        else:
            return f"MeasurementStandardizer(fitted=False, device={self.device})"


# =============================================================================
# POSITION COORDINATE SCALER
# =============================================================================

class PositionScaler:
    """
    Min-max scaling for spatial coordinates to [-1, 1] range.
    
    This class implements coordinate scaling for NIR position data,
    mapping x,y coordinates to [-1, 1] using fixed geometry bounds
    derived from the training data geometry.
    
    Scaling formula: x' = 2*(x - xmin) / (xmax - xmin) - 1
    
    The scaler:
    - Computes min/max bounds per coordinate from training data only
    - Applies scaling consistently across train/val/test splits
    - No inverse transform needed (positions are inputs only)
    - Persists bounds for Stage 2 reuse
    
    Expected input shape: [batch_size, n_measurements, 6]
    where coordinates = [src_x, src_y, src_z, det_x, det_y, det_z]
    
    Attributes:
        mins (torch.Tensor): Per-coordinate minimums [6]
        maxs (torch.Tensor): Per-coordinate maximums [6]
        fitted (bool): Whether the scaler has been fitted to data
        device (torch.device): Device for tensor operations
    
    Example:
        >>> scaler = PositionScaler()
        >>> scaler.fit(train_positions)  # Shape: [N, n_meas, 6]
        >>> 
        >>> # During training/validation
        >>> scaled_positions = scaler.transform(positions)
        >>> model_inputs = torch.cat([measurements, scaled_positions], dim=-1)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the position scaler.
        
        Args:
            device (Optional[torch.device]): Device for tensor operations.
                If None, uses CPU by default.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.mins = None
        self.maxs = None
        self.fitted = False
        
        logger.debug(f"🔧 Initialized PositionScaler on device: {self.device}")
    
    def fit(self, positions: torch.Tensor) -> 'PositionScaler':
        """
        Fit the scaler to training position data.
        
        Args:
            positions (torch.Tensor): Training positions of shape [N, n_meas, 6]
                where coordinates = [src_x, src_y, src_z, det_x, det_y, det_z]
        
        Returns:
            PositionScaler: Self for method chaining
        
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if len(positions.shape) != 3:
            raise ValueError(f"Expected 3D tensor [N, n_meas, 6], got shape {positions.shape}")
        
        if positions.shape[2] != POSITION_DIM:
            raise ValueError(f"Expected {POSITION_DIM} coordinates, got {positions.shape[2]}")
        
        # Move to correct device
        positions = positions.to(self.device)
        
        # Compute min/max per coordinate across all samples and measurements
        # Shape: [N, n_meas, 6] -> compute over dimensions [0, 1] -> [6]
        self.mins = torch.min(positions, dim=0)[0]  # Shape: [n_meas, 6]
        self.mins = torch.min(self.mins, dim=0)[0]  # Shape: [6]
        
        self.maxs = torch.max(positions, dim=0)[0]  # Shape: [n_meas, 6]
        self.maxs = torch.max(self.maxs, dim=0)[0]  # Shape: [6]
        
        # Ensure ranges are not zero (add small epsilon if needed)
        ranges = self.maxs - self.mins
        ranges = torch.clamp(ranges, min=EPSILON)
        self.maxs = self.mins + ranges
        
        self.fitted = True
        
        # Log scaling bounds
        logger.info("📊 Fitted position scaling bounds:")
        for i, (coord_name, min_val, max_val) in enumerate(zip(POSITION_COORDINATES, self.mins, self.maxs)):
            logger.info(f"   Coordinate {i} ({coord_name}): min={min_val:.6f}, max={max_val:.6f}")
        
        return self
    
    def transform(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply min-max scaling to map coordinates to [-1, 1].
        
        Args:
            positions (torch.Tensor): Positions to scale [N, n_meas, 6]
        
        Returns:
            torch.Tensor: Scaled positions with same shape as input
        
        Raises:
            RuntimeError: If scaler has not been fitted
            ValueError: If input tensor has incorrect shape
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transforming data")
        
        if len(positions.shape) != 3 or positions.shape[2] != POSITION_DIM:
            raise ValueError(f"Expected shape [N, n_meas, {POSITION_DIM}], got {positions.shape}")
        
        # Move to correct device
        positions = positions.to(self.device)
        
        # Apply min-max scaling to [-1, 1]
        # Formula: x' = 2*(x - xmin) / (xmax - xmin) - 1
        # Broadcasting: [N, n_meas, 6] operations with [6]
        mins = self.mins.view(1, 1, POSITION_DIM)      # Shape: [1, 1, 6]
        maxs = self.maxs.view(1, 1, POSITION_DIM)      # Shape: [1, 1, 6]
        
        ranges = maxs - mins
        scaled = 2.0 * (positions - mins) / ranges - 1.0
        
        return scaled
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dictionary for saving scaler parameters."""
        if not self.fitted:
            raise RuntimeError("Cannot save state of unfitted scaler")
        
        return {
            'mins': self.mins,
            'maxs': self.maxs,
            'fitted': torch.tensor(self.fitted)
        }
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> 'PositionScaler':
        """Load state dictionary to restore scaler parameters."""
        required_keys = ['mins', 'maxs', 'fitted']
        for key in required_keys:
            if key not in state_dict:
                raise KeyError(f"Missing required key '{key}' in state_dict")
        
        self.mins = state_dict['mins'].to(self.device)
        self.maxs = state_dict['maxs'].to(self.device)
        self.fitted = bool(state_dict['fitted'].item())
        
        logger.info("✅ Loaded position scaler state:")
        for i, (coord_name, min_val, max_val) in enumerate(zip(POSITION_COORDINATES, self.mins, self.maxs)):
            logger.info(f"   Coordinate {i} ({coord_name}): min={min_val:.6f}, max={max_val:.6f}")
        
        return self
    
    def to(self, device: torch.device) -> 'PositionScaler':
        """Move scaler to specified device."""
        self.device = device
        if self.fitted:
            self.mins = self.mins.to(device)
            self.maxs = self.maxs.to(device)
        return self
    
    def __repr__(self) -> str:
        """String representation of the scaler."""
        if self.fitted:
            return (f"PositionScaler(fitted=True, device={self.device}, "
                   f"mins={self.mins.tolist()}, maxs={self.maxs.tolist()})")
        else:
            return f"PositionScaler(fitted=False, device={self.device})"


# =============================================================================
# TISSUE PATCH STANDARDIZER
# =============================================================================

class TissuePatchStandardizer:
    """
    Tissue patch normalization using ground truth statistics from Stage 1.
    
    This class applies the same per-channel μₐ/μ′ₛ z-score normalization
    used for ground truth volumes to tissue patches. This ensures consistency
    between the ground truth normalization and tissue context normalization.
    
    The standardizer:
    - Uses the exact same mean/std statistics as ground truth normalization
    - Applies standardization to tissue patches before feeding to model
    - No inverse transform needed (patches are inputs only)
    - Maintains consistency between Stage 1 and Stage 2 pipelines
    
    Expected input shape: [batch_size, n_measurements, 2, 2, patch_size, patch_size, patch_size]
    where first 2 is for [source_patch, detector_patch] and second 2 is for [μₐ, μ′ₛ]
    
    Attributes:
        ground_truth_standardizer (PerChannelZScore): Reference standardizer from Stage 1
        fitted (bool): Whether the standardizer has been initialized
        device (torch.device): Device for tensor operations
    
    Example:
        >>> # Load Stage 1 ground truth standardizer
        >>> gt_standardizer = PerChannelZScore()
        >>> gt_standardizer.load("checkpoints/stage1_standardizer.pth")
        >>> 
        >>> # Create tissue patch standardizer using same stats
        >>> patch_standardizer = TissuePatchStandardizer()
        >>> patch_standardizer.fit_from_ground_truth_standardizer(gt_standardizer)
        >>> 
        >>> # Apply to tissue patches
        >>> std_patches = patch_standardizer.transform(tissue_patches)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the tissue patch standardizer.
        
        Args:
            device (Optional[torch.device]): Device for tensor operations.
                If None, uses CPU by default.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.ground_truth_standardizer = None
        self.fitted = False
        
        logger.debug(f"🔧 Initialized TissuePatchStandardizer on device: {self.device}")
    
    def fit_from_ground_truth_standardizer(self, gt_standardizer: PerChannelZScore) -> 'TissuePatchStandardizer':
        """
        Initialize from a fitted ground truth standardizer.
        
        Args:
            gt_standardizer (PerChannelZScore): Fitted ground truth standardizer from Stage 1
        
        Returns:
            TissuePatchStandardizer: Self for method chaining
        
        Raises:
            RuntimeError: If ground truth standardizer is not fitted
        """
        if not gt_standardizer.fitted:
            raise RuntimeError("Ground truth standardizer must be fitted")
        
        # Create a copy of the ground truth standardizer for tissue patches
        self.ground_truth_standardizer = PerChannelZScore(device=self.device)
        self.ground_truth_standardizer.means = gt_standardizer.means.to(self.device)
        self.ground_truth_standardizer.stds = gt_standardizer.stds.to(self.device)
        self.ground_truth_standardizer.fitted = True
        
        self.fitted = True
        
        logger.info("✅ Tissue patch standardizer initialized from ground truth statistics:")
        for i, (channel_name, mean_val, std_val) in enumerate(zip(CHANNEL_NAMES, 
                                                                   self.ground_truth_standardizer.means,
                                                                   self.ground_truth_standardizer.stds)):
            logger.info(f"   Channel {i} ({channel_name}): mean={mean_val:.6f}, std={std_val:.6f}")
        
        return self
    
    def transform(self, tissue_patches: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """
        Apply ground truth standardization to tissue patches.
        
        Args:
            tissue_patches (torch.Tensor): Tissue patches of shape 
                [batch_size, n_measurements, 2, 2, patch_size, patch_size, patch_size]
                where first 2 = [source_patch, detector_patch], second 2 = [μₐ, μ′ₛ]
            clamp (bool): Whether to clamp values to prevent extreme outliers
        
        Returns:
            torch.Tensor: Standardized tissue patches with same shape as input
        
        Raises:
            RuntimeError: If standardizer has not been fitted
            ValueError: If input tensor has incorrect shape
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before transforming data")
        
        if len(tissue_patches.shape) != 7:
            raise ValueError(f"Expected 7D tensor [batch, n_meas, 2, 2, patch, patch, patch], got shape {tissue_patches.shape}")
        
        if tissue_patches.shape[3] != NUM_CHANNELS:
            raise ValueError(f"Expected {NUM_CHANNELS} channels, got {tissue_patches.shape[3]}")
        
        # Move to correct device
        tissue_patches = tissue_patches.to(self.device)
        
        batch_size, n_measurements, n_patch_types, n_channels, patch_size_d, patch_size_h, patch_size_w = tissue_patches.shape
        
        # Reshape for standardization: [batch*n_meas*n_patch_types, n_channels, patch_size, patch_size, patch_size]
        # This allows us to use the ground truth standardizer directly
        reshaped_patches = tissue_patches.view(batch_size * n_measurements * n_patch_types,
                                             n_channels, patch_size_d, patch_size_h, patch_size_w)
        
        # Apply ground truth standardization
        standardized_patches = self.ground_truth_standardizer.transform(reshaped_patches)
        
        # Optional clamping to prevent extreme outliers
        if clamp:
            standardized_patches = torch.clamp(standardized_patches, Z_SCORE_CLAMP_RANGE[0], Z_SCORE_CLAMP_RANGE[1])
        
        # Reshape back to original format
        standardized_patches = standardized_patches.view(batch_size, n_measurements, n_patch_types,
                                                       n_channels, patch_size_d, patch_size_h, patch_size_w)
        
        return standardized_patches
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dictionary for saving standardizer parameters."""
        if not self.fitted:
            raise RuntimeError("Cannot save state of unfitted standardizer")
        
        return {
            'gt_means': self.ground_truth_standardizer.means,
            'gt_stds': self.ground_truth_standardizer.stds,
            'fitted': torch.tensor(self.fitted)
        }
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> 'TissuePatchStandardizer':
        """Load state dictionary to restore standardizer parameters."""
        required_keys = ['gt_means', 'gt_stds', 'fitted']
        for key in required_keys:
            if key not in state_dict:
                raise KeyError(f"Missing required key '{key}' in state_dict")
        
        # Initialize ground truth standardizer
        self.ground_truth_standardizer = PerChannelZScore(device=self.device)
        self.ground_truth_standardizer.means = state_dict['gt_means'].to(self.device)
        self.ground_truth_standardizer.stds = state_dict['gt_stds'].to(self.device)
        self.ground_truth_standardizer.fitted = True
        
        self.fitted = bool(state_dict['fitted'].item())
        
        logger.info("✅ Loaded tissue patch standardizer state:")
        for i, (channel_name, mean_val, std_val) in enumerate(zip(CHANNEL_NAMES, 
                                                                   self.ground_truth_standardizer.means,
                                                                   self.ground_truth_standardizer.stds)):
            logger.info(f"   Channel {i} ({channel_name}): mean={mean_val:.6f}, std={std_val:.6f}")
        
        return self
    
    def to(self, device: torch.device) -> 'TissuePatchStandardizer':
        """Move standardizer to specified device."""
        self.device = device
        if self.fitted and self.ground_truth_standardizer is not None:
            self.ground_truth_standardizer.to(device)
        return self
    
    def __repr__(self) -> str:
        """String representation of the standardizer."""
        if self.fitted:
            return (f"TissuePatchStandardizer(fitted=True, device={self.device}, "
                   f"gt_means={self.ground_truth_standardizer.means.tolist()}, "
                   f"gt_stds={self.ground_truth_standardizer.stds.tolist()})")
        else:
            return f"TissuePatchStandardizer(fitted=False, device={self.device})"


# =============================================================================
# COMPREHENSIVE STANDARDIZER COLLECTION
# =============================================================================

class Stage2StandardizerCollection:
    """
    Collection of all standardizers needed for Stage 2 training.
    
    This class manages all the standardization components required for Stage 2:
    - Ground truth standardizer (loaded from Stage 1)
    - Measurement standardizer (fitted on training data)
    - Position scaler (fitted on training data) 
    - Tissue patch standardizer (uses ground truth stats)
    
    Provides unified fitting, transformation, and persistence for all components.
    
    Attributes:
        ground_truth_standardizer (PerChannelZScore): For inverse-transforming outputs
        measurement_standardizer (MeasurementStandardizer): For standardizing NIR measurements
        position_scaler (PositionScaler): For scaling spatial coordinates
        tissue_patch_standardizer (TissuePatchStandardizer): For standardizing tissue patches
        fitted (bool): Whether all standardizers have been fitted
        device (torch.device): Device for tensor operations
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the standardizer collection.
        
        Args:
            device (Optional[torch.device]): Device for tensor operations.
                If None, uses CPU by default.
        """
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize all standardizers
        self.ground_truth_standardizer = PerChannelZScore(device=self.device)
        self.measurement_standardizer = MeasurementStandardizer(device=self.device)
        self.position_scaler = PositionScaler(device=self.device)
        self.tissue_patch_standardizer = TissuePatchStandardizer(device=self.device)
        
        self.fitted = False
        
        logger.info(f"🏗️  Initialized Stage2StandardizerCollection on device: {self.device}")
    
    def fit_from_stage1_checkpoint(self, stage1_checkpoint_path: str,
                                 train_dataloader) -> 'Stage2StandardizerCollection':
        """
        Initialize standardizers from Stage 1 checkpoint and training data.
        
        Args:
            stage1_checkpoint_path (str): Path to Stage 1 checkpoint containing ground truth standardizer
            train_dataloader: Training dataloader for fitting measurement/position standardizers
        
        Returns:
            Stage2StandardizerCollection: Self for method chaining
        """
        logger.info(f"🔧 Fitting Stage 2 standardizers from checkpoint: {stage1_checkpoint_path}")
        
        # Load ground truth standardizer from Stage 1 checkpoint
        checkpoint = torch.load(stage1_checkpoint_path, map_location=self.device)
        if 'standardizer' not in checkpoint:
            raise KeyError("No 'standardizer' found in Stage 1 checkpoint")
        
        self.ground_truth_standardizer.load_state_dict(checkpoint['standardizer'])
        logger.info("✅ Loaded ground truth standardizer from Stage 1 checkpoint")
        
        # Initialize tissue patch standardizer using ground truth stats
        self.tissue_patch_standardizer.fit_from_ground_truth_standardizer(self.ground_truth_standardizer)
        logger.info("✅ Initialized tissue patch standardizer from ground truth stats")
        
        # Fit measurement and position standardizers on ALL training data
        logger.info("📊 Fitting measurement and position standardizers on training data...")
        
        all_measurements = []
        all_positions = []
        
        # Process all training batches for maximum statistical robustness
        # Since tissue patches are skipped, this is now memory-efficient
        total_batches = len(train_dataloader)
        logger.info(f"   🎯 Processing all {total_batches} batches for optimal standardizer fitting")
        
        for batch_idx, batch in enumerate(train_dataloader):
                
            # Extract NIR measurements from batch
            if isinstance(batch, dict):
                nir_measurements = batch.get('nir_measurements', batch.get('inputs'))
            else:
                nir_measurements = batch[0]  # Assume first element
            
            if nir_measurements is None:
                raise ValueError("Could not find NIR measurements in training batch")
            
            # Split NIR measurements into measurement features and positions
            measurements = nir_measurements[:, :, :NIR_FEATURE_DIM]  # [batch, n_meas, 2]
            positions = nir_measurements[:, :, NIR_FEATURE_DIM:]     # [batch, n_meas, 6]
            
            all_measurements.append(measurements.cpu())
            all_positions.append(positions.cpu())
            
            # Progress logging every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"   📊 Processed {batch_idx + 1}/{total_batches} batches for standardizer fitting...")
        
        logger.info(f"   ✅ Completed processing all {total_batches} batches for standardizer fitting")
        
        # Concatenate and fit on complete training data
        all_measurements = torch.cat(all_measurements, dim=0)
        all_positions = torch.cat(all_positions, dim=0)
        
        total_measurements = all_measurements.shape[0] * all_measurements.shape[1]
        logger.info(f"📊 Fitting standardizers on {all_measurements.shape[0]} phantoms ({total_measurements:,} measurements)")
        logger.info(f"   🎯 Using complete training dataset for optimal statistical robustness")
        
        self.measurement_standardizer.fit(all_measurements)
        self.position_scaler.fit(all_positions)
        
        self.fitted = True
        logger.info("✅ All Stage 2 standardizers fitted successfully!")
        
        return self
    
    def transform_nir_inputs(self, nir_measurements: torch.Tensor) -> torch.Tensor:
        """
        Transform NIR measurements with standardization.
        
        Args:
            nir_measurements (torch.Tensor): Raw NIR data [batch, n_meas, 8]
                where 8D = [log_amp, phase, src_x, src_y, src_z, det_x, det_y, det_z]
        
        Returns:
            torch.Tensor: Standardized NIR data [batch, n_meas, 8]
        """
        if not self.fitted:
            raise RuntimeError("Standardizers must be fitted before transforming data")
        
        # Split into measurements and positions
        measurements = nir_measurements[:, :, :NIR_FEATURE_DIM]  # [batch, n_meas, 2]
        positions = nir_measurements[:, :, NIR_FEATURE_DIM:]     # [batch, n_meas, 6]
        
        # Apply standardization
        std_measurements = self.measurement_standardizer.transform(measurements)
        scaled_positions = self.position_scaler.transform(positions)
        
        # Recombine
        standardized_nir = torch.cat([std_measurements, scaled_positions], dim=-1)
        
        return standardized_nir
    
    def transform_tissue_patches(self, tissue_patches: torch.Tensor) -> torch.Tensor:
        """
        Transform tissue patches using ground truth standardization.
        
        Args:
            tissue_patches (torch.Tensor): Raw tissue patches 
                [batch, n_meas, 2, 2, patch_size, patch_size, patch_size]
        
        Returns:
            torch.Tensor: Standardized tissue patches with same shape
        """
        if not self.fitted:
            raise RuntimeError("Standardizers must be fitted before transforming data")
        
        return self.tissue_patch_standardizer.transform(tissue_patches)
    
    def inverse_transform_ground_truth(self, standardized_volumes: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform ground truth predictions back to physical units.
        
        Args:
            standardized_volumes (torch.Tensor): Model predictions in standardized space [batch, 2, D, H, W]
        
        Returns:
            torch.Tensor: Predictions in physical units (μₐ, μ′ₛ)
        """
        if not self.fitted:
            raise RuntimeError("Standardizers must be fitted before inverse transforming data")
        
        return self.ground_truth_standardizer.inverse_transform(standardized_volumes)
    
    def save(self, save_dir: str) -> None:
        """
        Save all standardizers to directory.
        
        Args:
            save_dir (str): Directory to save standardizers
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted standardizers")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual standardizers
        torch.save(self.ground_truth_standardizer.state_dict(), 
                  os.path.join(save_dir, "ground_truth_standardizer.pth"))
        torch.save(self.measurement_standardizer.state_dict(),
                  os.path.join(save_dir, "measurement_standardizer.pth"))
        torch.save(self.position_scaler.state_dict(),
                  os.path.join(save_dir, "position_scaler.pth"))
        torch.save(self.tissue_patch_standardizer.state_dict(),
                  os.path.join(save_dir, "tissue_patch_standardizer.pth"))
        
        logger.info(f"💾 Saved all Stage 2 standardizers to: {save_dir}")
    
    def load(self, save_dir: str) -> 'Stage2StandardizerCollection':
        """
        Load all standardizers from directory.
        
        Args:
            save_dir (str): Directory containing saved standardizers
        
        Returns:
            Stage2StandardizerCollection: Self for method chaining
        """
        # Load individual standardizers
        self.ground_truth_standardizer.load_state_dict(
            torch.load(os.path.join(save_dir, "ground_truth_standardizer.pth"), map_location=self.device))
        self.measurement_standardizer.load_state_dict(
            torch.load(os.path.join(save_dir, "measurement_standardizer.pth"), map_location=self.device))
        self.position_scaler.load_state_dict(
            torch.load(os.path.join(save_dir, "position_scaler.pth"), map_location=self.device))
        self.tissue_patch_standardizer.load_state_dict(
            torch.load(os.path.join(save_dir, "tissue_patch_standardizer.pth"), map_location=self.device))
        
        self.fitted = True
        logger.info(f"📂 Loaded all Stage 2 standardizers from: {save_dir}")
        
        return self
    
    def to(self, device: torch.device) -> 'Stage2StandardizerCollection':
        """Move all standardizers to specified device."""
        self.device = device
        self.ground_truth_standardizer.to(device)
        self.measurement_standardizer.to(device)
        self.position_scaler.to(device)
        self.tissue_patch_standardizer.to(device)
        return self
    
    def __repr__(self) -> str:
        """String representation of the collection."""
        return (f"Stage2StandardizerCollection(fitted={self.fitted}, device={self.device}, "
               f"components=[ground_truth, measurement, position, tissue_patch])")


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


def create_measurement_standardizer(device: Optional[torch.device] = None) -> MeasurementStandardizer:
    """
    Create a new measurement standardizer.
    
    Args:
        device (Optional[torch.device]): Device for tensor operations
    
    Returns:
        MeasurementStandardizer: New standardizer instance
    """
    return MeasurementStandardizer(device=device)


def create_position_scaler(device: Optional[torch.device] = None) -> PositionScaler:
    """
    Create a new position scaler.
    
    Args:
        device (Optional[torch.device]): Device for tensor operations
    
    Returns:
        PositionScaler: New scaler instance
    """
    return PositionScaler(device=device)


def create_tissue_patch_standardizer(device: Optional[torch.device] = None) -> TissuePatchStandardizer:
    """
    Create a new tissue patch standardizer.
    
    Args:
        device (Optional[torch.device]): Device for tensor operations
    
    Returns:
        TissuePatchStandardizer: New standardizer instance
    """
    return TissuePatchStandardizer(device=device)


def create_stage2_standardizer_collection(device: Optional[torch.device] = None) -> Stage2StandardizerCollection:
    """
    Create a new Stage 2 standardizer collection.
    
    Args:
        device (Optional[torch.device]): Device for tensor operations
    
    Returns:
        Stage2StandardizerCollection: New collection instance
    """
    return Stage2StandardizerCollection(device=device)


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



