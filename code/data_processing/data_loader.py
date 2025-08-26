"""
NIR Phantom Data Loader - Deterministic Fixed Split Version
===========================================================

This module provides PyTorch DataLoader classes for NIR phantom datasets with
phantom-level loading and deterministic fixed dataset splits.

Key Features:
- Phantom-level loading (complete phantoms with {N_MEASUREMENTS} subsampled measurements)
- FIXED DETERMINISTIC SPLITS: Train 0-7999, Val 8000-8999, Test 9000-9999
- No randomization in dataset partitioning - identical splits across all runs
- Data augmentation via random measurement subsampling within each phantom
- Efficient multi-CPU data loading with hardware optimization
- HDF5-based storage for efficient access

Author: Max Hart
Date: 2024
"""

import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset, DataLoader

# Import tissue patch config flag
from code.training.training_config import USE_TISSUE_PATCHES_STAGE2, VOLUME_SHAPE, N_MEASUREMENTS, GLOBAL_SEED

# Import centralized logging
from code.utils.logging_config import get_training_logger

# ===============================================================================
# CONFIGURATION AND CONSTANTS
# ===============================================================================

# Data structure configuration
DEFAULT_NIR_FEATURE_DIMENSION = 8      # [log_amp, phase, src_x, src_y, src_z, det_x, det_y, det_z]
DEFAULT_OPTICAL_CHANNELS = 2            # [μ_a, μ_s] absorption and scattering
DEFAULT_PHANTOM_SHAPE = VOLUME_SHAPE    # Volume dimensions - centralized from training_config
DEFAULT_N_GENERATED_MEASUREMENTS = 1000 # Generated measurements per phantom
# Note: Removed DEFAULT_N_TRAINING_MEASUREMENTS for dynamic undersampling at model level

# Dataset split configuration - FIXED DETERMINISTIC SPLITS
# For 10,000 phantom dataset: deterministic 8000/1000/1000 split by index
TRAIN_SPLIT_START = 0                   # Train set: indices 0-7999 (8000 phantoms)
TRAIN_SPLIT_END = 8000
VAL_SPLIT_START = 8000                  # Validation set: indices 8000-8999 (1000 phantoms) 
VAL_SPLIT_END = 9000
TEST_SPLIT_START = 9000                 # Test set: indices 9000-9999 (1000 phantoms)
TEST_SPLIT_END = 10000

# Legacy ratios kept for compatibility but not used in fixed splitting
TRAIN_SPLIT_RATIO = 0.8                 # 80% for training (legacy - not used for fixed splits)
VALIDATION_SPLIT_RATIO = 0.1             # 10% for validation (legacy - not used for fixed splits)
TEST_SPLIT_RATIO = 0.1                   # 10% for test (legacy - not used for fixed splits)

# DataLoader configuration
DEFAULT_BATCH_SIZE = 8
DEFAULT_PHANTOM_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_RANDOM_SEED = GLOBAL_SEED  # Use global seed for consistency


def _extract_phantom_number(phantom_file_path):
    """Extract phantom number from file path like phantom_01/phantom_1_scan.h5 -> 1"""
    import re
    # OS-agnostic path parsing with stricter error handling
    match = re.search(r'[\\/ ]phantom_(\d+)_scan\.h5$', str(phantom_file_path))
    if not match:
        match = re.search(r'[\\/ ]phantom_(\d+)[\\/ ]', str(phantom_file_path))
    if not match:
        logger.error(f"Failed to parse phantom number from: {phantom_file_path}")
        return None
    return int(match.group(1))


def _get_phantom_range_from_indices(phantom_files, start_idx, end_idx):
    """Get actual phantom number range from file indices"""
    if not phantom_files or start_idx >= len(phantom_files):
        return None, None
    
    actual_end_idx = min(end_idx, len(phantom_files))
    start_phantom = _extract_phantom_number(phantom_files[start_idx])
    end_phantom = _extract_phantom_number(phantom_files[actual_end_idx - 1])  # -1 because end_idx is exclusive
    
    return start_phantom, end_phantom


# Performance optimization settings
ENABLE_PIN_MEMORY = True
ENABLE_PERSISTENT_WORKERS = True
DROP_INCOMPLETE_BATCHES = True

# HDF5 dataset keys
H5_KEYS = {
    'log_amplitude': 'log_amplitude',
    'phase': 'phase',
    'source_pos': 'source_positions',
    'det_pos': 'detector_positions',
    'ground_truth': 'ground_truth'
}

# Get logger - Use training logger since data_loader is used by training components
logger = get_training_logger(__name__)

# Worker initialization function for reproducible multiprocessing
def _worker_init_fn(worker_id):
    """Initialize workers with deterministic seeds based on global seed and worker ID."""
    import numpy as np
    import torch
    import random
    worker_seed = GLOBAL_SEED + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ===============================================================================
# TISSUE PATCH EXTRACTION
# ===============================================================================

def extract_tissue_patches_from_measurements(ground_truth: np.ndarray, 
                                            source_positions: np.ndarray,
                                            detector_positions: np.ndarray,
                                            measurement_indices: np.ndarray,
                                            patch_size: int = 16) -> np.ndarray:
    """
    Extract 16x16x16 tissue patches around source and detector locations for each measurement.
    
    This function extracts tissue property patches from the ground truth volume around
    source and detector positions for each NIR measurement. These patches provide
    local anatomical context that enhances reconstruction quality.
    
    Args:
        ground_truth (np.ndarray): Ground truth volume of shape (2, 64, 64, 64) 
                                  where 2 channels are [μ_a, μ_s]
        source_positions (np.ndarray): Source positions of shape (n_measurements, 3) in mm
        detector_positions (np.ndarray): Detector positions of shape (n_measurements, 3) in mm  
        measurement_indices (np.ndarray): Indices of selected measurements
        patch_size (int): Size of cubic patch to extract (default 16)
        
    Returns:
        np.ndarray: Tissue patches of shape (n_selected_measurements, 2_patches, 2_channels, 16, 16, 16)
                   Each measurement gets 2 patches (source + detector) × 2 tissue properties
                   Preserves spatial structure for efficient CNN processing
    """
    n_selected = len(measurement_indices)
    channels, vol_d, vol_h, vol_w = ground_truth.shape  # (2, 64, 64, 64)
    
    # Initialize output tensor: [n_measurements, 2_patches, 2_channels, 16, 16, 16]
    tissue_patches = np.zeros((n_selected, 2, 2, patch_size, patch_size, patch_size), dtype=np.float32)
    
    # Volume dimensions and coordinate system
    # CRITICAL: Determine coordinate system from actual data
    vol_extent_voxels = 64  # 64 voxels
    
    # Debug: Check position ranges to understand coordinate system
    src_min, src_max = source_positions[measurement_indices].min(axis=0), source_positions[measurement_indices].max(axis=0)
    det_min, det_max = detector_positions[measurement_indices].min(axis=0), detector_positions[measurement_indices].max(axis=0)
    logger.debug(f"🔍 Source position range: {src_min} to {src_max}")
    logger.debug(f"🔍 Detector position range: {det_min} to {det_max}")
    
    # Determine coordinate system conversion
    # If positions are in [-32, +32] range (centered), need different conversion than [0, 64] range
    pos_center = (src_min + src_max) / 2
    pos_extent = src_max - src_min
    logger.debug(f"🔍 Position center: {pos_center}, extent: {pos_extent}")
    
    # Choose coordinate conversion based on data range
    if np.all(src_min >= -35) and np.all(src_max <= 35):
        # Centered coordinate system [-32, +32] → [0, 64]
        logger.debug("📍 Using centered coordinate system conversion")
        def pos_to_voxel(pos_mm):
            return np.round(pos_mm + vol_extent_voxels / 2).astype(int)
    else:
        # Already in voxel coordinates [0, 64] 
        logger.debug("📍 Using direct coordinate system conversion")
        def pos_to_voxel(pos_mm):
            return np.round(pos_mm).astype(int)
    
    # Track patch quality statistics
    total_patches = 0
    zero_patches = 0
    partial_zero_patches = 0
    
    for i, meas_idx in enumerate(measurement_indices):
        # Get source and detector positions for this measurement (in mm)
        src_pos_mm = source_positions[meas_idx]  # [x, y, z] in mm
        det_pos_mm = detector_positions[meas_idx]  # [x, y, z] in mm
        
        # Convert positions to voxel indices using detected coordinate system
        src_voxel = pos_to_voxel(src_pos_mm)
        det_voxel = pos_to_voxel(det_pos_mm)
        
        # Clamp positions to volume bounds [0, 63]
        src_voxel = np.clip(src_voxel, 0, vol_extent_voxels - 1)
        det_voxel = np.clip(det_voxel, 0, vol_extent_voxels - 1)
        
        logger.debug(f"Measurement {i}: src_mm={src_pos_mm} → voxel={src_voxel}, det_mm={det_pos_mm} → voxel={det_voxel}")
        
        positions = [src_voxel, det_voxel]
        position_names = ["source", "detector"]
        
        for patch_idx, (pos_voxel, pos_name) in enumerate(zip(positions, position_names)):
            x, y, z = pos_voxel
            
            # Calculate patch boundaries (centered on position)
            half_patch = patch_size // 2
            x_start = max(0, x - half_patch)
            x_end = min(vol_w, x + half_patch)
            y_start = max(0, y - half_patch)  
            y_end = min(vol_h, y + half_patch)
            z_start = max(0, z - half_patch)
            z_end = min(vol_d, z + half_patch)
            
            logger.debug(f"  {pos_name} patch bounds: x[{x_start}:{x_end}], y[{y_start}:{y_end}], z[{z_start}:{z_end}]")
            
            # Extract patch for both tissue property channels
            # IMPORTANT: ground_truth shape is (2, 64, 64, 64) = (channels, z, y, x)
            patch_absorption = ground_truth[0, z_start:z_end, y_start:y_end, x_start:x_end]  # μ_a
            patch_scattering = ground_truth[1, z_start:z_end, y_start:y_end, x_start:x_end]  # μ_s
            
            logger.debug(f"  Extracted patch shape: {patch_absorption.shape}")
            
            # Handle edge cases by padding if necessary
            actual_patch_shape = patch_absorption.shape
            if actual_patch_shape != (patch_size, patch_size, patch_size):
                logger.debug(f"  Padding patch from {actual_patch_shape} to ({patch_size}, {patch_size}, {patch_size})")
                # Pad to full patch size
                padded_absorption = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                padded_scattering = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                
                # Calculate padding offsets to center the extracted region
                pad_z = (patch_size - actual_patch_shape[0]) // 2
                pad_y = (patch_size - actual_patch_shape[1]) // 2
                pad_x = (patch_size - actual_patch_shape[2]) // 2
                
                end_z = pad_z + actual_patch_shape[0]
                end_y = pad_y + actual_patch_shape[1] 
                end_x = pad_x + actual_patch_shape[2]
                
                padded_absorption[pad_z:end_z, pad_y:end_y, pad_x:end_x] = patch_absorption
                padded_scattering[pad_z:end_z, pad_y:end_y, pad_x:end_x] = patch_scattering
                                
                patch_absorption = padded_absorption
                patch_scattering = padded_scattering
            
            # Keep spatial structure instead of flattening!
            # Store as [2_channels, 16, 16, 16] for each patch
            tissue_patches[i, patch_idx, 0] = patch_absorption  # Shape: [16, 16, 16]
            tissue_patches[i, patch_idx, 1] = patch_scattering  # Shape: [16, 16, 16]
            
            # Quality monitoring and validation warnings
            total_patches += 1
            absorption_nonzero = np.count_nonzero(patch_absorption)
            scattering_nonzero = np.count_nonzero(patch_scattering)
            total_voxels = patch_absorption.size
            
            # Calculate zero percentage for this patch (both channels)
            total_zeros = np.sum(patch_absorption == 0) + np.sum(patch_scattering == 0)
            patch_zero_percentage = (total_zeros / (total_voxels * 2)) * 100
            
            # VALIDATION WARNINGS: Only log when tissue patches are being actively extracted
            # Demote to debug level to reduce log noise during standardizer fitting
            if patch_zero_percentage > 85.0:
                logger.debug(f"⚠️ HIGH ZEROS: {pos_name} patch at meas {i} has {patch_zero_percentage:.1f}% zeros (>85%)")
            elif patch_zero_percentage < 20.0:
                logger.debug(f"⚠️ LOW ZEROS: {pos_name} patch at meas {i} has {patch_zero_percentage:.1f}% zeros (<20%)")
            
            # Legacy quality tracking
            if absorption_nonzero == 0 and scattering_nonzero == 0:
                zero_patches += 1
                if i < 3:  # Log first few zero patches
                    logger.warning(f"  ⚠️ Completely zero patch: {pos_name} at voxel {pos_voxel}")
            elif absorption_nonzero < total_voxels * 0.1 or scattering_nonzero < total_voxels * 0.1:
                partial_zero_patches += 1
                if i < 3:  # Log first few partial patches
                    logger.debug(f"  📊 Low content patch: {pos_name} abs={absorption_nonzero}/{total_voxels}, scat={scattering_nonzero}/{total_voxels}")
            else:
                if i < 3:  # Log first few good patches
                    logger.debug(f"  ✅ Good patch: {pos_name} abs={absorption_nonzero}/{total_voxels}, scat={scattering_nonzero}/{total_voxels}")
                    logger.debug(f"     Value ranges: abs[{patch_absorption.min():.4f}, {patch_absorption.max():.4f}], scat[{patch_scattering.min():.4f}, {patch_scattering.max():.4f}]")
    
    # Final quality report
    zero_ratio = zero_patches / total_patches if total_patches > 0 else 0
    partial_ratio = partial_zero_patches / total_patches if total_patches > 0 else 0
    
    logger.debug(f"🎯 Tissue Patch Extraction Summary:")
    logger.debug(f"   Total patches: {total_patches}")
    logger.debug(f"   Zero patches: {zero_patches} ({zero_ratio:.1%})")
    logger.debug(f"   Partial patches: {partial_zero_patches} ({partial_ratio:.1%})")
    logger.debug(f"   Good patches: {total_patches - zero_patches - partial_zero_patches} ({(1 - zero_ratio - partial_ratio):.1%})")
    
    return tissue_patches

# ===============================================================================
# PHANTOM DATASET CLASS
# ===============================================================================

class NIRPhantomDataset(Dataset):
    """
    Dataset for NIR phantom data with deterministic fixed splitting and phantom-level loading.
    
    Uses FIXED DETERMINISTIC SPLITS (no randomization):
    - Train: phantom indices 0-7999 (8000 phantoms)  
    - Val: phantom indices 8000-8999 (1000 phantoms)
    - Test: phantom indices 9000-9999 (1000 phantoms)
    
    Each phantom contains ground truth optical properties and NIR measurements.
    Measurement subsampling within phantoms uses random_seed for data augmentation.
    """
    
    def __init__(self, data_dir: str = "../data", split: str = "train", 
                 random_seed: int = DEFAULT_RANDOM_SEED, 
                 extract_tissue_patches: bool = True,
                 stage: str = "stage2"):  # NEW: Add stage parameter
        """Initialize dataset with phantom files for different training stages."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.random_seed = random_seed
        self.extract_tissue_patches = extract_tissue_patches
        self.stage = stage  # NEW: Store stage for optimized loading
        
        # Set up random state for data augmentation (NOT for splitting)
        self.rng = np.random.RandomState(random_seed)
        
        # Find all phantom files with numeric sorting (filter unparsable files)
        all_files = list(self.data_dir.glob("phantom_*/phantom_*_scan.h5"))
        files = [p for p in all_files if _extract_phantom_number(p) is not None]
        def _phantom_id(p):
            return _extract_phantom_number(p)
        self.phantom_files = sorted(files, key=_phantom_id)
        
        if len(files) != len(all_files):
            logger.warning(f"Some phantom files had unparsable IDs and were skipped: {len(all_files)} found, {len(files)} usable")
        
        if not self.phantom_files:
            raise ValueError(f"No phantom files found in {self.data_dir}")
        
        logger.info(f"Found {len(self.phantom_files)} phantom files in {self.data_dir}")
        
        # Store phantom IDs aligned with phantom files for sanity checking
        self.phantom_ids = [_extract_phantom_number(p) for p in self.phantom_files]
        
        # FIXED DETERMINISTIC SPLITS - No randomization, always same indices
        n_phantoms = len(self.phantom_files)
        
        # Use fixed index ranges for deterministic splitting
        if split == "train":
            phantom_indices = np.arange(TRAIN_SPLIT_START, min(TRAIN_SPLIT_END, n_phantoms))
            ids = [self.phantom_ids[i] for i in phantom_indices]
            start_id, end_id = (min(ids), max(ids)) if ids else (None, None)
            logger.info(f"📊 Train set: phantoms {start_id}–{end_id} (indices {TRAIN_SPLIT_START}–{min(TRAIN_SPLIT_END-1, n_phantoms-1)}, {len(phantom_indices)} phantoms)")
            logger.info(f"   Train sample IDs: {ids[:3]} … {ids[-3:]}")
        elif split == "val":
            phantom_indices = np.arange(VAL_SPLIT_START, min(VAL_SPLIT_END, n_phantoms))
            ids = [self.phantom_ids[i] for i in phantom_indices]
            start_id, end_id = (min(ids), max(ids)) if ids else (None, None)
            logger.info(f"📊 Val set: phantoms {start_id}–{end_id} (indices {VAL_SPLIT_START}–{min(VAL_SPLIT_END-1, n_phantoms-1)}, {len(phantom_indices)} phantoms)")
            logger.info(f"   Val sample IDs: {ids[:3]} … {ids[-3:]}")
        elif split == "test":
            phantom_indices = np.arange(TEST_SPLIT_START, min(TEST_SPLIT_END, n_phantoms))
            ids = [self.phantom_ids[i] for i in phantom_indices]
            start_id, end_id = (min(ids), max(ids)) if ids else (None, None)
            logger.info(f"📊 Test set: phantoms {start_id}–{end_id} (indices {TEST_SPLIT_START}–{min(TEST_SPLIT_END-1, n_phantoms-1)}, {len(phantom_indices)} phantoms)")
            logger.info(f"   Test sample IDs: {ids[:3]} … {ids[-3:]}")
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Sanity check for first split only (avoid repeated warnings)
        if split == "train":
            n_total = len(self.phantom_files)
            n_unique = len(set(self.phantom_ids))
            if n_unique != n_total:
                logger.warning(f"Duplicate or unparsable phantom IDs detected: {n_total} files but {n_unique} unique IDs")
        
        # Validate split boundaries for safety
        if len(phantom_indices) == 0:
            raise ValueError(f"No phantoms available for {split} split with {n_phantoms} total phantoms")
        
        # Log warning if dataset size doesn't match expected 10k phantoms
        if n_phantoms != 10000:
            logger.warning(f"⚠️  Dataset has {n_phantoms} phantoms (expected 10,000). Fixed splits may not align perfectly.")
        
        # Select phantom files and IDs for this split (keep ids aligned with files!)
        self.phantom_files = [self.phantom_files[i] for i in phantom_indices]
        self.phantom_ids = [self.phantom_ids[i] for i in phantom_indices]
        
        # Final logging with phantom count
        logger.info(f"✅ {split.upper()} dataset ready: {len(self.phantom_files)} phantoms loaded")
    
    def __len__(self) -> int:
        """Return total number of phantoms in dataset."""
        return len(self.phantom_files)

    def __getitem__(self, phantom_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load phantom data optimized for different training stages.
        
        STAGE 1 (CNN Autoencoder): Only loads ground truth - 97% memory reduction!
        STAGE 2 (Transformer): Loads full data including NIR measurements and tissue patches
        
        Args:
            phantom_idx: Index of phantom to load
            
        Returns:
            Dict containing stage-appropriate data
        """
        if self.stage == "stage1":
            return self._load_stage1_data(phantom_idx)
        else:
            return self._load_stage2_data(phantom_idx)
    
    def _load_stage1_data(self, phantom_idx: int) -> Dict[str, torch.Tensor]:
        """
        OPTIMIZED Stage 1 loading: Ground truth only for CNN autoencoder training.
        
        Memory savings: ~97% reduction (2.1MB vs 67.6MB per phantom)
        - Skips NIR measurements (not used in Stage 1)
        - Skips tissue patches (not used in Stage 1)
        - Only loads ground truth volumes for reconstruction training
        
        Returns:
            Dict with ground_truth and phantom_id only
        """
        if phantom_idx >= len(self.phantom_files):
            raise IndexError(f"Phantom index {phantom_idx} out of range (0-{len(self.phantom_files)-1})")
        
        phantom_file = self.phantom_files[phantom_idx]
        
        try:
            with h5py.File(phantom_file, 'r') as f:
                # Stage 1: ONLY load ground truth (2.1MB per phantom vs 67.6MB)
                ground_truth = f[H5_KEYS['ground_truth']][:]  # Shape: (2, 64, 64, 64)
                phantom_id = int(phantom_file.stem.split('_')[1])
                
        except Exception as e:
            logger.error(f"Error loading Stage 1 phantom {phantom_idx}: {e}")
            return {
                'ground_truth': torch.zeros(2, 64, 64, 64, dtype=torch.float32),
                'phantom_id': torch.tensor(0, dtype=torch.long)
            }
        
        return {
            'ground_truth': torch.tensor(ground_truth, dtype=torch.float32),
            'phantom_id': torch.tensor(phantom_id, dtype=torch.long)
            # NO nir_measurements, NO tissue_patches for Stage 1 speed!
        }
    
    def _load_stage2_data(self, phantom_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load complete phantom data with subsampled measurements for training.
        
        This method loads phantom data (1000 generated measurements from optimized probe placement)
        and randomly subsamples {N_MEASUREMENTS} measurements for training to enable data augmentation.
        
        🎯 **Data Augmentation Strategy:**
        • Generate 1000 measurements per phantom (50 sources × 20 detectors)
        • Randomly subsample {N_MEASUREMENTS} measurements for each training batch
        • Different subsets provide 3.9x more training combinations
        • Maintains consistent training pipeline dimensions
        
        Args:
            phantom_idx (int): Index of phantom to load (0 to len(phantom_files)-1)
            
        Returns:
            Dict[str, torch.Tensor]: Complete phantom data containing:
                - 'nir_measurements': Subsampled NIR measurements ({N_MEASUREMENTS}, 8) - 8D features per measurement
                - 'ground_truth': Target volume (2, 64, 64, 64) - same for all measurements
                - 'phantom_id': Phantom identifier for tracking
        """
        if phantom_idx >= len(self.phantom_files):
            raise IndexError(f"Phantom index {phantom_idx} out of range (0-{len(self.phantom_files)-1})")
        
        phantom_file = self.phantom_files[phantom_idx]
        
        try:
            with h5py.File(phantom_file, 'r') as f:
                # Load all NIR measurement data (expects 1000 measurements from optimized probe placement)
                log_amplitude = f[H5_KEYS['log_amplitude']][:]  # Shape: (1000,) - generated measurements
                phase = f[H5_KEYS['phase']][:]                   # Shape: (1000,) - generated measurements
                
                # Load geometry data (1000 source-detector pairs)
                source_pos = f[H5_KEYS['source_pos']][:]         # Shape: (1000, 3) for [x, y, z] in mm
                detector_pos = f[H5_KEYS['det_pos']][:]          # Shape: (1000, 3) for [x, y, z] in mm
                
                # Load ground truth volume (same for all measurements in phantom)
                ground_truth = f[H5_KEYS['ground_truth']][:]     # Shape: (2, 64, 64, 64) - already channels-first
                # No transpose needed since data is already in channels-first format!
                
                # Build complete measurement array (1000 measurements initially)
                n_measurements = log_amplitude.shape[0]
                
                all_measurements = np.zeros((n_measurements, DEFAULT_NIR_FEATURE_DIMENSION), dtype=np.float32)
                
                for measurement_idx in range(n_measurements):
                    all_measurements[measurement_idx] = np.array([
                        float(log_amplitude[measurement_idx]),      # log amplitude
                        float(phase[measurement_idx]),              # phase
                        source_pos[measurement_idx, 0],             # source x
                        source_pos[measurement_idx, 1],             # source y
                        source_pos[measurement_idx, 2],             # source z
                        detector_pos[measurement_idx, 0],           # detector x
                        detector_pos[measurement_idx, 1],           # detector y
                        detector_pos[measurement_idx, 2]            # detector z
                    ])
                
                # 🎯 PASS ALL MEASUREMENTS: No longer subsample here - dynamic undersampling at model level
                # This allows for adaptive sequence undersampling during Stage 2 training
                nir_measurements = all_measurements
                logger.debug(f"Loaded {n_measurements} measurements for phantom {phantom_file.stem}")
                
                # Extract phantom ID
                phantom_id = int(phantom_file.stem.split('_')[1])
                
                # Prepare result dict with core data
                result = {
                    'nir_measurements': torch.tensor(nir_measurements, dtype=torch.float32),
                    'ground_truth': torch.tensor(ground_truth, dtype=torch.float32),
                    'phantom_id': torch.tensor(phantom_id, dtype=torch.long)
                }
                
                # 🧬 TISSUE PATCH EXTRACTION: Only when enabled (Stage 2 Enhanced mode)
                if self.extract_tissue_patches:
                    tissue_patches = extract_tissue_patches_from_measurements(
                        ground_truth=ground_truth,           # (2, 64, 64, 64)
                        source_positions=source_pos,         # (1000, 3) 
                        detector_positions=detector_pos,     # (1000, 3)
                        measurement_indices=np.arange(n_measurements), # Use all measurements
                        patch_size=16
                    )
                    
                    logger.debug(f"Extracted tissue patches shape: {tissue_patches.shape}")  # Should be (1000, 2, 2, 16, 16, 16)
                    result['tissue_patches'] = torch.tensor(tissue_patches, dtype=torch.float32)
                    
                    # 📊 TISSUE PATCH QUALITY MONITORING: Only when patches are extracted
                    tissue_tensor = result['tissue_patches']
                    zero_ratio = (tissue_tensor == 0).float().mean()
                    if zero_ratio > 0.98:  # Only warn for extremely high zero content (>98%)
                        logger.debug(f"⚠️ Very high zero content in tissue patches: {zero_ratio:.1%} for phantom {phantom_id}")
                    elif zero_ratio < 0.3:
                        logger.debug(f"📍 Unusually low zero content: {zero_ratio:.1%} for phantom {phantom_id}")
                    else:
                        logger.debug(f"✅ Normal tissue patch quality: {zero_ratio:.1%} zeros for phantom {phantom_id}")
                else:
                    # No tissue patch extraction - don't add the key at all (clean, zero-alloc)
                    logger.debug(f"Skipping tissue patch extraction (tissue patches disabled)")
                
                return result
                
        except Exception as e:
            logger.error(f"Error loading complete phantom {phantom_idx} from {phantom_file}: {e}")
            # Return zero-filled phantom with minimal memory allocation
            result = {
                'nir_measurements': torch.zeros(DEFAULT_N_GENERATED_MEASUREMENTS, DEFAULT_NIR_FEATURE_DIMENSION, dtype=torch.float32),
                'ground_truth': torch.zeros(DEFAULT_OPTICAL_CHANNELS, *DEFAULT_PHANTOM_SHAPE, dtype=torch.float32),
                'phantom_id': torch.tensor(0, dtype=torch.long)
            }
            
            # Only add tissue patches if extraction is enabled
            if self.extract_tissue_patches:
                result['tissue_patches'] = torch.zeros(DEFAULT_N_GENERATED_MEASUREMENTS, 2, 2, 16, 16, 16, dtype=torch.float32)
            
            return result


# ===============================================================================
# DATALOADER FACTORY FUNCTION
# ===============================================================================

def create_phantom_dataloaders(data_dir: str = "../data",
                              batch_size: int = DEFAULT_PHANTOM_BATCH_SIZE,
                              num_workers: int = DEFAULT_NUM_WORKERS,
                              prefetch_factor: int = 2,
                              pin_memory: bool = True,
                              persistent_workers: bool = True,
                              random_seed: int = DEFAULT_RANDOM_SEED,
                              extract_tissue_patches: bool = None,  # Deprecated param name
                              use_tissue_patches: bool = USE_TISSUE_PATCHES_STAGE2,  # New explicit flag
                              stage: str = "stage2") -> Dict[str, DataLoader]:
    """
    Create DataLoaders for complete phantom batching (batches of complete phantoms).
    
    OPTIMIZED for different training stages:
    - Stage 1: Only loads ground truth (97% memory reduction!)
    - Stage 2: Loads full data including NIR measurements and optional tissue patches
    
    FIXED DETERMINISTIC SPLITS (NO RANDOMIZATION):
    - Train: indices 0-7999 (8000 phantoms, 80%)
    - Validation: indices 8000-8999 (1000 phantoms, 10%)
    - Test: indices 9000-9999 (1000 phantoms, 10%)
    
    Args:
        data_dir (str): Path to phantom data directory
        batch_size (int): Number of phantoms per batch
        num_workers (int): Number of worker processes
        prefetch_factor (int): Prefetch factor for DataLoader
        pin_memory (bool): Enable pin memory for GPU efficiency
        persistent_workers (bool): Keep workers alive between epochs
        random_seed (int): Random seed for data augmentation (NOT for splitting)
        extract_tissue_patches (bool): DEPRECATED - use use_tissue_patches instead
        use_tissue_patches (bool): Whether to extract tissue patches (Stage 2 Enhanced only)
        stage (str): Training stage ('stage1' for ground truth only, 'stage2' for full data)
        
    Returns:
        Dict[str, DataLoader]: Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Handle backward compatibility for extract_tissue_patches
    if extract_tissue_patches is not None:
        logger.warning("extract_tissue_patches is deprecated, use use_tissue_patches instead")
        use_tissue_patches = extract_tissue_patches
    
    # Get all phantom files with numeric sorting
    data_dir = Path(data_dir)
    files = list(data_dir.glob("phantom_*/phantom_*_scan.h5"))
    def _phantom_id(p):
        return _extract_phantom_number(p) or -1
    phantom_files = sorted(files, key=_phantom_id)
    
    # Extract actual phantom number ranges for logging
    train_ids = [_extract_phantom_number(phantom_files[i]) for i in range(TRAIN_SPLIT_START, min(TRAIN_SPLIT_END, len(phantom_files)))]
    val_ids = [_extract_phantom_number(phantom_files[i]) for i in range(VAL_SPLIT_START, min(VAL_SPLIT_END, len(phantom_files)))]
    test_ids = [_extract_phantom_number(phantom_files[i]) for i in range(TEST_SPLIT_START, min(TEST_SPLIT_END, len(phantom_files)))]
    
    train_start, train_end = (min(train_ids), max(train_ids)) if train_ids else (None, None)
    val_start, val_end = (min(val_ids), max(val_ids)) if val_ids else (None, None)
    test_start, test_end = (min(test_ids), max(test_ids)) if test_ids else (None, None)
    
    # Log fixed dataset split configuration once at startup
    logger.info("🔧 FIXED DETERMINISTIC DATASET SPLITS:")
    logger.info(f"   📊 Train set: phantoms {train_start}–{train_end} (indices {TRAIN_SPLIT_START}–{TRAIN_SPLIT_END-1}, {TRAIN_SPLIT_END-TRAIN_SPLIT_START} phantoms)")
    logger.info(f"   📊 Val set: phantoms {val_start}–{val_end} (indices {VAL_SPLIT_START}–{VAL_SPLIT_END-1}, {VAL_SPLIT_END-VAL_SPLIT_START} phantoms)")
    logger.info(f"   📊 Test set: phantoms {test_start}–{test_end} (indices {TEST_SPLIT_START}–{TEST_SPLIT_END-1}, {TEST_SPLIT_END-TEST_SPLIT_START} phantoms)")
    logger.info(f"   🎯 No randomization - splits are identical across all runs")
    
    # Ensure persistent workers is disabled for single worker
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = 2
    
    dataloaders = {}
    all_dataset_ids = []
    split_id_sets = {}

    for split in ['train', 'val', 'test']:
        dataset = NIRPhantomDataset(data_dir, split, random_seed, 
                                   extract_tissue_patches=use_tissue_patches, stage=stage)
        all_dataset_ids.extend(dataset.phantom_ids)
        split_id_sets[split] = set(dataset.phantom_ids)
        
        shuffle = (split == 'train') 
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train') and DROP_INCOMPLETE_BATCHES,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None
        )
        
        dataloaders[split] = dataloader
        
        logger.info(f"Created phantom-level {split} DataLoader: {len(dataset)} phantoms, "
                   f"~{len(dataloader)} batches per epoch")
    
    # Improved sanity check: verify splits are disjoint and cover all files
    all_ids_set = set(all_dataset_ids)
    total_phantom_files = len(phantom_files)
    
    # Expect union size == total files
    if len(all_ids_set) != total_phantom_files:
        logger.warning(f"Split sanity check failed: union={len(all_ids_set)} vs files={total_phantom_files}")
    
    # Expect pairwise disjoint splits
    if (split_id_sets['train'] & split_id_sets['val']) or \
       (split_id_sets['train'] & split_id_sets['test']) or \
       (split_id_sets['val']   & split_id_sets['test']):
        logger.error("Split overlap detected between train/val/test!")
    
    # Expect exact cardinalities when you have 10k phantoms
    if total_phantom_files == 10000:
        if not (len(split_id_sets['train']) == 8000 and
                len(split_id_sets['val'])   == 1000 and
                len(split_id_sets['test'])  == 1000):
            logger.warning("Unexpected split sizes; expected 8000/1000/1000.")
    
    # Confirmation log for which splits were created
    logger.info(f"✅ Created dataloaders for splits: {list(dataloaders.keys())}")
    
    return dataloaders



