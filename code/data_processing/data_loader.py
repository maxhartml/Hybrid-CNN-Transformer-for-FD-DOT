"""
NIR Phantom Data Loader - Cleaned Version
========================================

This module provides PyTorch DataLoader classes for NIR phantom datasets with
phantom-level loading and data augmentation through measurement subsampling.

Key Features:
- Phantom-level loading (complete phantoms with 256 subsampled measurements)
- Data augmentation via random measurement subsampling from 1000 generated measurements
- Efficient multi-CPU data loading with hardware optimization
- Train/val/test splits at phantom level to prevent data leakage
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

# ===============================================================================
# CONFIGURATION AND CONSTANTS
# ===============================================================================

# Data structure configuration
DEFAULT_NIR_FEATURE_DIMENSION = 8      # [log_amp, phase, src_x, src_y, src_z, det_x, det_y, det_z]
DEFAULT_OPTICAL_CHANNELS = 2            # [μ_a, μ_s] absorption and scattering
DEFAULT_PHANTOM_SHAPE = (64, 64, 64)    # Volume dimensions
DEFAULT_N_GENERATED_MEASUREMENTS = 1000 # Generated measurements per phantom
DEFAULT_N_TRAINING_MEASUREMENTS = 256   # Subsampled measurements for training

# Dataset split configuration
TRAIN_SPLIT_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.1

# DataLoader configuration
DEFAULT_BATCH_SIZE = 8
DEFAULT_PHANTOM_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_RANDOM_SEED = 42

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

# Get logger
logger = logging.getLogger(__name__)

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
        np.ndarray: Tissue patches of shape (n_selected_measurements, 2, patch_size^3 * 2)
                   Each measurement gets 2 patches (source + detector) × 2 tissue properties
                   Flattened format: [μ_a_patch, μ_s_patch] for each location
    """
    n_selected = len(measurement_indices)
    channels, vol_d, vol_h, vol_w = ground_truth.shape  # (2, 64, 64, 64)
    
    # Initialize output tensor: [n_measurements, 2_patches, patch_volume * 2_channels]
    patch_volume = patch_size ** 3
    tissue_patches = np.zeros((n_selected, 2, patch_volume * 2), dtype=np.float32)
    
    # Volume dimensions in mm (assuming 1mm voxel size, centered at origin)
    vol_extent_mm = 64  # 64mm volume
    voxel_size_mm = 1.0
    vol_center_mm = vol_extent_mm / 2
    
    # Debug: Check position ranges
    src_min, src_max = source_positions[measurement_indices].min(axis=0), source_positions[measurement_indices].max(axis=0)
    det_min, det_max = detector_positions[measurement_indices].min(axis=0), detector_positions[measurement_indices].max(axis=0)
    logger.debug(f"Source position range: {src_min} to {src_max}")
    logger.debug(f"Detector position range: {det_min} to {det_max}")
    
    for i, meas_idx in enumerate(measurement_indices):
        # Get source and detector positions for this measurement (in mm)
        src_pos_mm = source_positions[meas_idx]  # [x, y, z] in mm
        det_pos_mm = detector_positions[meas_idx]  # [x, y, z] in mm
        
        # Convert positions from mm to voxel indices (assuming volume centered at origin)
        # Clamp to ensure positions are within volume bounds
        src_voxel = np.round(src_pos_mm + vol_center_mm).astype(int)  # Center at (32, 32, 32)
        det_voxel = np.round(det_pos_mm + vol_center_mm).astype(int)
        
        # Clamp positions to volume bounds
        src_voxel = np.clip(src_voxel, 0, vol_extent_mm - 1)
        det_voxel = np.clip(det_voxel, 0, vol_extent_mm - 1)
        
        positions = [src_voxel, det_voxel]
        
        for patch_idx, pos_voxel in enumerate(positions):
            x, y, z = pos_voxel
            
            # Calculate patch boundaries (centered on position)
            half_patch = patch_size // 2
            x_start = max(0, x - half_patch)
            x_end = min(vol_w, x + half_patch)
            y_start = max(0, y - half_patch)  
            y_end = min(vol_h, y + half_patch)
            z_start = max(0, z - half_patch)
            z_end = min(vol_d, z + half_patch)
            
            # Extract patch for both tissue property channels
            patch_absorption = ground_truth[0, z_start:z_end, y_start:y_end, x_start:x_end]  # μ_a
            patch_scattering = ground_truth[1, z_start:z_end, y_start:y_end, x_start:x_end]  # μ_s
            
            # Handle edge cases by padding if necessary
            actual_patch_shape = patch_absorption.shape
            if actual_patch_shape != (patch_size, patch_size, patch_size):
                # Pad to full patch size
                padded_absorption = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                padded_scattering = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                
                # Calculate padding offsets to center the extracted region
                pad_z = (patch_size - actual_patch_shape[0]) // 2
                pad_y = (patch_size - actual_patch_shape[1]) // 2
                pad_x = (patch_size - actual_patch_shape[2]) // 2
                
                padded_absorption[pad_z:pad_z+actual_patch_shape[0], 
                                pad_y:pad_y+actual_patch_shape[1],
                                pad_x:pad_x+actual_patch_shape[2]] = patch_absorption
                                
                padded_scattering[pad_z:pad_z+actual_patch_shape[0],
                                pad_y:pad_y+actual_patch_shape[1], 
                                pad_x:pad_x+actual_patch_shape[2]] = patch_scattering
                
                patch_absorption = padded_absorption
                patch_scattering = padded_scattering
            
            # Flatten and concatenate both channels: [μ_a_flat, μ_s_flat]
            # IMPORTANT: NIR processor expects interleaved format for proper reshaping
            # Convert concatenated format to interleaved format for compatibility
            absorption_flat = patch_absorption.flatten()  # 16^3 values
            scattering_flat = patch_scattering.flatten()   # 16^3 values
            
            # Debug: Check if patches have meaningful content
            if i == 0 and patch_idx == 0:  # Log for first patch only
                absorption_nonzero = np.count_nonzero(absorption_flat)
                scattering_nonzero = np.count_nonzero(scattering_flat)
                total_voxels = len(absorption_flat)
                logger.debug(f"First patch content: absorption {absorption_nonzero}/{total_voxels} nonzero, "
                           f"scattering {scattering_nonzero}/{total_voxels} nonzero")
                logger.debug(f"Patch stats: abs range [{absorption_flat.min():.4f}, {absorption_flat.max():.4f}], "
                           f"scat range [{scattering_flat.min():.4f}, {scattering_flat.max():.4f}]")
            
            # Create interleaved format: [μ_a[0], μ_s[0], μ_a[1], μ_s[1], ...]
            interleaved_patch = np.zeros(patch_volume * 2, dtype=np.float32)
            interleaved_patch[0::2] = absorption_flat   # Even indices: absorption
            interleaved_patch[1::2] = scattering_flat   # Odd indices: scattering
            
            tissue_patches[i, patch_idx] = interleaved_patch
    
    return tissue_patches

# ===============================================================================
# PHANTOM DATASET CLASS
# ===============================================================================

class NIRPhantomDataset(Dataset):
    """
    Dataset for NIR phantom data with phantom-level loading and measurement subsampling.
    
    This dataset loads complete phantoms (1000 measurements) and subsamples to 256 
    measurements for training to enable data augmentation. Each phantom contains
    ground truth optical properties and NIR measurements from optimized probe placement.
    """
    
    def __init__(self, data_dir: str = "../data", split: str = "train", 
                 random_seed: int = DEFAULT_RANDOM_SEED):
        """Initialize dataset with phantom files."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.random_seed = random_seed
        
        # Set up random state for reproducible splits
        self.rng = np.random.RandomState(random_seed)
        
        # Find all phantom files
        self.phantom_files = sorted(list(self.data_dir.glob("phantom_*/phantom_*_scan.h5")))
        
        if not self.phantom_files:
            raise ValueError(f"No phantom files found in {self.data_dir}")
        
        logger.info(f"Found {len(self.phantom_files)} phantom files in {self.data_dir}")
        
        # Split phantoms deterministically
        n_phantoms = len(self.phantom_files)
        n_train = int(TRAIN_SPLIT_RATIO * n_phantoms)
        n_val = int(VALIDATION_SPLIT_RATIO * n_phantoms)
        
        # Deterministic shuffle for consistent splits across runs
        phantom_indices = np.arange(n_phantoms)
        split_rng = np.random.RandomState(random_seed)
        split_rng.shuffle(phantom_indices)
        
        if split == "train":
            phantom_indices = phantom_indices[:n_train]
        elif split == "val":
            phantom_indices = phantom_indices[n_train:n_train + n_val]
        elif split == "test":
            phantom_indices = phantom_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.phantom_files = [self.phantom_files[i] for i in phantom_indices]
        
        logger.info(f"{split.upper()} split: {len(self.phantom_files)} phantoms")
    
    def __len__(self) -> int:
        """Return total number of phantoms in dataset."""
        return len(self.phantom_files)

    def __getitem__(self, phantom_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load complete phantom data with subsampled measurements for training.
        
        This method loads phantom data (1000 generated measurements from optimized probe placement)
        and randomly subsamples 256 measurements for training to enable data augmentation.
        
        🎯 **Data Augmentation Strategy:**
        • Generate 1000 measurements per phantom (50 sources × 20 detectors)
        • Randomly subsample 256 measurements for each training batch
        • Different subsets provide 3.9x more training combinations
        • Maintains consistent training pipeline dimensions
        
        Args:
            phantom_idx (int): Index of phantom to load (0 to len(phantom_files)-1)
            
        Returns:
            Dict[str, torch.Tensor]: Complete phantom data containing:
                - 'nir_measurements': Subsampled NIR measurements (256, 8) - 8D features per measurement
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
                
                # 🎯 CRITICAL: Subsample from 1000 to 256 measurements for training
                # This enables data augmentation - different random subsets each epoch
                if n_measurements > DEFAULT_N_TRAINING_MEASUREMENTS:
                    # Use phantom-specific random state for consistent subsampling per phantom per epoch
                    # But allow different subsets across epochs via global random state
                    subsample_indices = np.random.choice(
                        n_measurements, 
                        size=DEFAULT_N_TRAINING_MEASUREMENTS, 
                        replace=False
                    )
                    nir_measurements = all_measurements[subsample_indices]
                    logger.debug(f"Subsampled {DEFAULT_N_TRAINING_MEASUREMENTS} from {n_measurements} measurements for phantom {phantom_file.stem}")
                else:
                    nir_measurements = all_measurements
                    logger.warning(f"Phantom has only {n_measurements} measurements, expected {DEFAULT_N_GENERATED_MEASUREMENTS}")
                
                # Extract phantom ID
                phantom_id = int(phantom_file.stem.split('_')[1])
                
                # 🧬 TISSUE PATCH EXTRACTION: Extract 16x16x16 patches around source/detector locations
                # This provides local anatomical context for each measurement
                if n_measurements > DEFAULT_N_TRAINING_MEASUREMENTS:
                    # Use the same subsample indices for tissue patches
                    tissue_patches = extract_tissue_patches_from_measurements(
                        ground_truth=ground_truth,           # (2, 64, 64, 64)
                        source_positions=source_pos,         # (1000, 3) 
                        detector_positions=detector_pos,     # (1000, 3)
                        measurement_indices=subsample_indices, # (256,) selected indices
                        patch_size=16
                    )
                else:
                    # Use all measurements if we have fewer than training target
                    tissue_patches = extract_tissue_patches_from_measurements(
                        ground_truth=ground_truth,
                        source_positions=source_pos,
                        detector_positions=detector_pos,
                        measurement_indices=np.arange(n_measurements),
                        patch_size=16
                    )
                
                logger.debug(f"Extracted tissue patches shape: {tissue_patches.shape}")  # Should be (256, 2, 8192)
                
                # 📊 TISSUE PATCH QUALITY MONITORING: Basic sanity check
                tissue_tensor = torch.tensor(tissue_patches, dtype=torch.float32)
                zero_ratio = (tissue_tensor == 0).float().mean()
                if zero_ratio > 0.98:  # Only warn for extremely high zero content (>98%)
                    logger.debug(f"⚠️ Very high zero content in tissue patches: {zero_ratio:.1%} for phantom {phantom_id}")
                elif zero_ratio < 0.3:
                    logger.debug(f"📍 Unusually low zero content: {zero_ratio:.1%} for phantom {phantom_id}")
                else:
                    logger.debug(f"✅ Normal tissue patch quality: {zero_ratio:.1%} zeros for phantom {phantom_id}")
                
        except Exception as e:
            logger.error(f"Error loading complete phantom {phantom_idx} from {phantom_file}: {e}")
            # Return zero-filled phantom with training dimensions
            return {
                'nir_measurements': torch.zeros(DEFAULT_N_TRAINING_MEASUREMENTS, DEFAULT_NIR_FEATURE_DIMENSION, dtype=torch.float32),
                'ground_truth': torch.zeros(DEFAULT_OPTICAL_CHANNELS, *DEFAULT_PHANTOM_SHAPE, dtype=torch.float32),
                'tissue_patches': torch.zeros(DEFAULT_N_TRAINING_MEASUREMENTS, 2, 16**3 * 2, dtype=torch.float32),
                'phantom_id': torch.tensor(0, dtype=torch.long)
            }
        
        return {
            'nir_measurements': torch.tensor(nir_measurements, dtype=torch.float32),    # (256, 8) - subsampled NIR measurements
            'ground_truth': torch.tensor(ground_truth, dtype=torch.float32),           # (2, 64, 64, 64) - full ground truth
            'tissue_patches': torch.tensor(tissue_patches, dtype=torch.float32),       # (256, 2, 8192) - tissue patches for each measurement
            'phantom_id': torch.tensor(phantom_id, dtype=torch.long)                   # Phantom identifier
        }


# ===============================================================================
# HARDWARE OPTIMIZATION
# ===============================================================================

def get_optimal_dataloader_config() -> Dict:
    """
    Get optimal DataLoader configuration based on system hardware.
    
    Returns:
        Dict: Configuration with optimal settings for num_workers, pin_memory, etc.
    """
    import psutil
    
    # Get CPU information
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    logical_count = psutil.cpu_count(logical=True)  # Logical cores
    
    # Get memory information
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Determine optimal number of workers
    # Conservative approach: use physical cores but cap at 8 to avoid overhead
    optimal_workers = min(max(1, cpu_count - 1), 8)
    
    # Enable pin memory if sufficient RAM and CUDA available
    pin_memory = torch.cuda.is_available() and available_gb > 4.0
    
    # Set prefetch factor based on memory
    prefetch_factor = 4 if available_gb > 16 else 2
    
    # Enable persistent workers for multi-worker setups
    persistent_workers = optimal_workers > 1
    
    config = {
        'num_workers': optimal_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers
    }
    
    logger.info(f"Optimal DataLoader config: {config}")
    logger.info(f"System: {cpu_count} physical cores, {logical_count} logical cores, {available_gb:.1f}GB available RAM")
    
    return config


# ===============================================================================
# DATALOADER FACTORY FUNCTION
# ===============================================================================

def create_phantom_dataloaders(data_dir: str = "../data",
                              batch_size: int = DEFAULT_PHANTOM_BATCH_SIZE,
                              num_workers: int = None,  # Auto-detect optimal workers
                              random_seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for complete phantom batching (batches of complete phantoms).
    
    This creates a custom DataLoader that returns batches of complete phantoms,
    where each phantom contains 256 subsampled measurements from the generated 1000.
    Enables data augmentation through different random subsets each epoch.
    
    Args:
        data_dir (str): Path to phantom data directory
        batch_size (int): Number of phantoms per batch
        num_workers (int): Number of worker processes (auto-detected if None)
        random_seed (int): Random seed for splits
        
    Returns:
        Dict[str, DataLoader]: Dictionary with 'train', 'val', 'test' DataLoaders
    """
    
    # Get optimal DataLoader configuration
    if num_workers is None:
        dataloader_config = get_optimal_dataloader_config()
        num_workers = dataloader_config['num_workers']
        pin_memory = dataloader_config['pin_memory']
        prefetch_factor = dataloader_config['prefetch_factor']
        persistent_workers = dataloader_config['persistent_workers'] and num_workers > 0
    else:
        # Use provided num_workers with safe defaults
        pin_memory = torch.cuda.is_available() and ENABLE_PIN_MEMORY
        prefetch_factor = 2
        persistent_workers = num_workers > 0
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = NIRPhantomDataset(data_dir, split, random_seed)
        
        shuffle = (split == 'train') 
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train') and DROP_INCOMPLETE_BATCHES,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        dataloaders[split] = dataloader
        
        logger.info(f"Created phantom-level {split} DataLoader: {len(dataset)} phantoms, "
                   f"~{len(dataloader)} batches per epoch")
    
    return dataloaders


# ===============================================================================
# MAIN FUNCTION FOR TESTING
# ===============================================================================

def main():
    """Test the cleaned data loader."""
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data loader creation
    try:
        dataloaders = create_phantom_dataloaders(
            data_dir="../data",
            batch_size=2,
            num_workers=2
        )
        
        # Test loading a single batch
        train_loader = dataloaders['train']
        for batch in train_loader:
            print(f"Batch shapes:")
            print(f"  NIR measurements: {batch['nir_measurements'].shape}")
            print(f"  Ground truth: {batch['ground_truth'].shape}")
            print(f"  Phantom IDs: {batch['phantom_id'].shape}")
            break
            
        print("✅ Data loader test successful!")
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")


if __name__ == "__main__":
    main()
