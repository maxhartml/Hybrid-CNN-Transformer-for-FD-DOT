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
    'source_pos': 'source_pos',
    'det_pos': 'det_pos',
    'ground_truth': 'ground_truth'
}

# Get logger
logger = logging.getLogger(__name__)

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
                
        except Exception as e:
            logger.error(f"Error loading complete phantom {phantom_idx} from {phantom_file}: {e}")
            # Return zero-filled phantom with training dimensions
            return {
                'nir_measurements': torch.zeros(DEFAULT_N_TRAINING_MEASUREMENTS, DEFAULT_NIR_FEATURE_DIMENSION, dtype=torch.float32),
                'ground_truth': torch.zeros(DEFAULT_OPTICAL_CHANNELS, *DEFAULT_PHANTOM_SHAPE, dtype=torch.float32),
                'phantom_id': torch.tensor(0, dtype=torch.long)
            }
        
        return {
            'nir_measurements': torch.tensor(nir_measurements, dtype=torch.float32),  # (256, DEFAULT_NIR_FEATURE_DIMENSION) - subsampled
            'ground_truth': torch.tensor(ground_truth, dtype=torch.float32),          # (DEFAULT_OPTICAL_CHANNELS, 64, 64, 64)
            'phantom_id': torch.tensor(phantom_id, dtype=torch.long)
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
