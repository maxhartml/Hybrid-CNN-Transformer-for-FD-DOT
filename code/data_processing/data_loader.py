#!/usr/bin/env python3
"""
🔬 NIR PHANTOM DATASET LOADER 🔬

Streamlined PyTorch DataLoader for NIR phantom datasets featuring:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 CORE FUNCTIONALITY:
• Lazy loading from distributed HDF5 phantom files (efficient memory usage)
• Automatic tissue patch extraction around source/detector positions
• Cross-phantom shuffling for robust training
• 8/1/1 train/validation/test splits at phantom level
• Toggle functionality for two-stage training approach

🎯 TWO-STAGE TRAINING SUPPORT:
• Stage 1: Full ground truth volumes for CNN autoencoder training
• Stage 2: Ground truth + optional tissue patches for transformer training
• Clean toggle switches throughout pipeline (use_tissue_patches parameter)

🏗️ ARCHITECTURE:
• NIRPhantomDataset: Main PyTorch Dataset class with toggle support
• Efficient H5 file management with lazy loading
• Phantom-level train/val/test splits to prevent data leakage

Author: Max Hart - NIR Tomography Research
Version: 2.0 - Two-Stage Training Integration
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Handle imports for both package and standalone usage
try:
    from ..utils.logging_config import get_data_logger
    logger = get_data_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# ===============================================================================
# CONFIGURATION CONSTANTS
# ===============================================================================

# Patch extraction parameters
DEFAULT_PATCH_SIZE = (7, 7, 7)                    # Local tissue patch dimensions
AIR_VALUE = 0.0                                   # Air padding value for out-of-bounds regions

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.80                                # 80% for training
VAL_RATIO = 0.10                                  # 10% for validation  
TEST_RATIO = 0.10                                 # 10% for testing

# HDF5 dataset keys (must match data_simulator.py output)
H5_KEYS = {
    'log_amplitude': 'log_amplitude',             # (N_probes, 3) log-amplitude measurements
    'phase': 'phase',                             # (N_probes, 3) phase measurements  
    'source_pos': 'source_positions',             # (N_probes, 3) source coordinates
    'det_pos': 'detector_positions',              # (N_probes, 3, 3) detector coordinates
    'ground_truth': 'ground_truth',               # (Nx, Ny, Nz, 2) optical property maps
    'tissue_labels': 'tissue_labels'              # (Nx, Ny, Nz) tissue type labels
}


# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def extract_tissue_patch(ground_truth: np.ndarray, 
                        position: np.ndarray, 
                        patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE) -> np.ndarray:
    """
    Extract tissue patch around a position for tissue context encoder.
    
    Simplified extraction focused on compatibility with TissueContextEncoder.
    Extracts a local patch of optical properties around source/detector positions.
    
    Args:
        ground_truth (np.ndarray): Full phantom optical properties, shape (Nx, Ny, Nz, 2)
        position (np.ndarray): Center position coordinates [x, y, z] in voxel indices
        patch_size (tuple): Patch dimensions (px, py, pz), default (7,7,7)
        
    Returns:
        np.ndarray: Flattened patch for tissue context encoder, shape (patch_size^3 * 2,)
    """
    Nx, Ny, Nz, n_channels = ground_truth.shape
    px, py, pz = patch_size
    
    # Convert position to integer indices
    cx, cy, cz = int(position[0]), int(position[1]), int(position[2])
    
    # Calculate patch boundaries
    half_px, half_py, half_pz = px // 2, py // 2, pz // 2
    x_start, x_end = cx - half_px, cx + half_px + 1
    y_start, y_end = cy - half_py, cy + half_py + 1  
    z_start, z_end = cz - half_pz, cz + half_pz + 1
    
    # Initialize patch with air padding
    patch = np.full((px, py, pz, n_channels), AIR_VALUE, dtype=ground_truth.dtype)
    
    # Extract valid region
    gt_x_start, gt_x_end = max(0, x_start), min(Nx, x_end)
    gt_y_start, gt_y_end = max(0, y_start), min(Ny, y_end)
    gt_z_start, gt_z_end = max(0, z_start), min(Nz, z_end)
    
    patch_x_start = max(0, -x_start)
    patch_x_end = patch_x_start + (gt_x_end - gt_x_start)
    patch_y_start = max(0, -y_start)
    patch_y_end = patch_y_start + (gt_y_end - gt_y_start)
    patch_z_start = max(0, -z_start)
    patch_z_end = patch_z_start + (gt_z_end - gt_z_start)
    
    # Copy valid region
    if (gt_x_end > gt_x_start and gt_y_end > gt_y_start and gt_z_end > gt_z_start):
        patch[patch_x_start:patch_x_end, 
              patch_y_start:patch_y_end, 
              patch_z_start:patch_z_end] = ground_truth[gt_x_start:gt_x_end,
                                                       gt_y_start:gt_y_end,
                                                       gt_z_start:gt_z_end]
    
    # Return flattened patch for tissue context encoder
    return patch.reshape(-1)


# ===============================================================================
# MAIN DATASET CLASS
# ===============================================================================

class NIRPhantomDataset(Dataset):
    """
    PyTorch Dataset for NIR phantom data with lazy loading and patch extraction.
    Enhanced with toggle functionality for Robin Dale's two-stage training approach.
    
    This dataset provides efficient access to distributed HDF5 phantom files with
    on-demand patch extraction around source and detector positions. Implements
    cross-phantom shuffling and phantom-level train/val/test splits to prevent
    data leakage and ensure robust model evaluation.
    
    Key Features:
    - Lazy loading: H5 files opened only when needed (memory efficient)
    - Patch extraction: 7×7×7 local tissue patches around optodes (toggle-enabled)
    - Cross-phantom mixing: Samples shuffled across all phantoms
    - Phantom-level splits: Prevents overfitting to specific phantom types
    - Boundary handling: Robust patch extraction with air padding
    - Toggle functionality: Clean enable/disable of tissue patches for A/B testing
    
    Dataset Structure:
    Each sample represents one source-detector measurement:
    - DOT measurements: Ground truth volume (for both stages)
    - Tissue patches: Optional 7×7×7×2 local tissue around optodes (stage 2 enhanced)
    - Metadata: phantom_id, probe_id, detector_id for tracking
    """
    
    def __init__(self, 
                 data_dir: str = "../data",
                 split: str = "train",
                 patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
                 use_tissue_patches: bool = True,
                 random_seed: int = 42):
        """
        Initialize NIR phantom dataset with automatic file discovery and indexing.
        Enhanced with toggle functionality for tissue patches.
        
        Args:
            data_dir (str): Path to directory containing phantom_XX subdirectories
            split (str): Dataset split - 'train', 'val', or 'test'
            patch_size (tuple): Patch dimensions for local tissue extraction
            use_tissue_patches (bool): Toggle for enabling tissue patch extraction
            random_seed (int): Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.use_tissue_patches = use_tissue_patches
        
        logger.info(f"📊 Initializing NIRPhantomDataset: split={split}, patch_size={patch_size}, "
                   f"use_tissue_patches={use_tissue_patches}")
        
        # Discover all phantom H5 files
        self.phantom_files = self._discover_phantom_files()
        
        # Split phantoms by ID (not individual samples) to prevent data leakage
        self.phantom_files = self._split_phantoms_by_id(self.phantom_files, split, random_seed)
        
        # Build comprehensive sample index for lazy loading
        self.sample_index = self._build_sample_index()
        
        logger.info(f"Dataset initialized: {len(self.phantom_files)} phantoms, {len(self.sample_index)} samples")
    
    def _discover_phantom_files(self) -> List[Path]:
        """
        Discover all phantom H5 files using robust pattern matching.
        
        Returns:
            List[Path]: Sorted list of phantom H5 file paths
        """
        # Search for phantom_XXX/phantom_XXX_scan.h5 pattern
        pattern = str(self.data_dir / "phantom_*" / "phantom_*_scan.h5")
        h5_files = [Path(f) for f in glob.glob(pattern)]
        
        if not h5_files:
            logger.warning(f"No phantom H5 files found in {self.data_dir}")
            return []
        
        # Sort by phantom number for consistent ordering
        h5_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        logger.info(f"🗂️  Discovered {len(h5_files)} phantom H5 files")
        
        # Validate files are readable
        valid_files = []
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Quick validation of required datasets
                    required_keys = [H5_KEYS['log_amplitude'], H5_KEYS['ground_truth']]
                    if all(key in f for key in required_keys):
                        valid_files.append(h5_file)
                    else:
                        logger.warning(f"Skipping {h5_file}: missing required datasets")
            except Exception as e:
                logger.warning(f"Skipping {h5_file}: error reading file - {e}")
        
        logger.info(f"✅ Validated {len(valid_files)} readable phantom files")
        return valid_files
    
    def _split_phantoms_by_id(self, phantom_files: List[Path], split: str, random_seed: int) -> List[Path]:
        """
        Split phantoms into train/val/test sets at the phantom level to prevent data leakage.
        
        Args:
            phantom_files (List[Path]): All discovered phantom files
            split (str): Desired split ('train', 'val', 'test')
            random_seed (int): Random seed for reproducible splits
            
        Returns:
            List[Path]: Phantom files belonging to specified split
        """
        if not phantom_files:
            return []
        
        # Set random seed for reproducible splits
        np.random.seed(random_seed)
        
        # Shuffle phantom files to randomize assignment
        phantom_files = phantom_files.copy()
        np.random.shuffle(phantom_files)
        
        n_phantoms = len(phantom_files)
        
        # Handle small datasets by ensuring minimum 1 phantom per split if possible
        if n_phantoms < 3:
            # Very small dataset - put everything in train
            if split == "train":
                selected_files = phantom_files
            else:
                selected_files = []
        elif n_phantoms < 10:
            # Small dataset - ensure at least 1 phantom in val/test if possible
            if split == "train":
                selected_files = phantom_files[:-2] if n_phantoms >= 3 else phantom_files
            elif split == "val":
                selected_files = phantom_files[-2:-1] if n_phantoms >= 3 else []
            elif split == "test":
                selected_files = phantom_files[-1:] if n_phantoms >= 2 else []
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        elif n_phantoms == 10:
            # Special case for exactly 10 phantoms: 8/1/1 split
            if split == "train":
                selected_files = phantom_files[:8]
            elif split == "val":
                selected_files = phantom_files[8:9]
            elif split == "test":
                selected_files = phantom_files[9:10]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        else:
            # Normal dataset - use percentage splits
            train_end = int(n_phantoms * TRAIN_RATIO)
            val_end = train_end + int(n_phantoms * VAL_RATIO)
            
            # Ensure at least 1 phantom in each split if possible
            if val_end == train_end and n_phantoms > train_end:
                val_end = train_end + 1
            
            # Assign phantoms to splits
            if split == "train":
                selected_files = phantom_files[:train_end]
            elif split == "val":
                selected_files = phantom_files[train_end:val_end]
            elif split == "test":
                selected_files = phantom_files[val_end:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        logger.info(f"Split '{split}': {len(selected_files)}/{n_phantoms} phantoms "
                   f"({len(selected_files)/n_phantoms*100:.1f}%)")
        
        return selected_files
    
    def _build_sample_index(self) -> List[Tuple[Path, int, int]]:
        """
        Build comprehensive index of all samples for efficient lazy loading.
        
        Creates a list of (phantom_file, probe_idx, detector_idx) tuples that
        allows direct access to any sample without loading entire files.
        
        Returns:
            List[Tuple[Path, int, int]]: Sample index for lazy loading
                Each tuple: (phantom_h5_path, probe_index, detector_index)
        """
        sample_index = []
        
        for phantom_file in self.phantom_files:
            try:
                with h5py.File(phantom_file, 'r') as f:
                    # Read probe count from HDF5 attributes or shape
                    if 'n_probes' in f.attrs:
                        n_probes = f.attrs['n_probes']
                    else:
                        n_probes = f[H5_KEYS['log_amplitude']].shape[0]
                    
                    # Each probe has 3 detectors, creating individual samples
                    for probe_idx in range(n_probes):
                        for det_idx in range(3):  # 3 detectors per probe
                            sample_index.append((phantom_file, probe_idx, det_idx))
                            
            except Exception as e:
                logger.warning(f"Error indexing {phantom_file}: {e}")
                continue
        
        logger.info(f"Built sample index: {len(sample_index)} total samples")
        return sample_index
    
    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load single sample with optional patch extraction and return as tensor dictionary.
        Enhanced with toggle functionality for Robin Dale's two-stage training.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Sample data containing:
                - 'dot_measurements': Ground truth volume (Nx, Ny, Nz) for reconstruction
                - 'ground_truth': Same as dot_measurements (for consistency)
                - 'tissue_patches': Combined tissue patches (num_patches, patch_size^3) if enabled
                - 'metadata': [phantom_id, probe_id, detector_id] (3,)
        """
        phantom_file, probe_idx, det_idx = self.sample_index[idx]
        
        try:
            with h5py.File(phantom_file, 'r') as f:
                # Load ground truth volume (primary input for both stages)
                ground_truth = f[H5_KEYS['ground_truth']][:]  # (Nx, Ny, Nz, 2)
                
                # For Robin Dale's approach, we use the full ground truth volume as input
                # This represents the "measurement" data for reconstruction
                dot_measurements = ground_truth[..., 0:1]  # Use absorption coefficient as measurement
                
                # Extract tissue patches if enabled (for stage 2 enhanced training)
                tissue_patches = None
                if self.use_tissue_patches:
                    # Load geometry for patch extraction
                    source_pos = f[H5_KEYS['source_pos']][probe_idx]
                    detector_pos = f[H5_KEYS['det_pos']][probe_idx, det_idx]
                    
                    # Extract patches around source and detector
                    source_patch = extract_tissue_patch(ground_truth, source_pos, self.patch_size)
                    detector_patch = extract_tissue_patch(ground_truth, detector_pos, self.patch_size)
                    
                    # Combine patches into a single tensor for tissue context encoder
                    # Each patch is already flattened, so we just stack them
                    tissue_patches = np.stack([source_patch, detector_patch], axis=0)  # (2, patch_size^3 * 2)
                
                # Extract phantom ID from filename
                phantom_id = int(phantom_file.stem.split('_')[1])
                
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {phantom_file}: {e}")
            # Return zero-filled sample to prevent training crashes
            return self._get_empty_sample()
        
        # Build return dictionary
        sample = {
            'measurements': torch.tensor(dot_measurements, dtype=torch.float32).permute(3, 0, 1, 2),  # (C, H, W, D)
            'dot_measurements': torch.tensor(dot_measurements, dtype=torch.float32).permute(3, 0, 1, 2),  # (C, H, W, D) - backward compatibility
            'ground_truth': torch.tensor(ground_truth, dtype=torch.float32),
            'volumes': torch.tensor(ground_truth, dtype=torch.float32),  # Add for training compatibility
            'metadata': torch.tensor([phantom_id, probe_idx, det_idx], dtype=torch.long)
        }
        
        # Add tissue patches if enabled
        if self.use_tissue_patches and tissue_patches is not None:
            sample['tissue_patches'] = torch.tensor(tissue_patches, dtype=torch.float32)
        
        return sample
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return zero-filled sample for error handling."""
        # Create empty sample with correct dimensions based on actual data
        sample = {
            'measurements': torch.zeros(1, 60, 60, 60, dtype=torch.float32),     # (C, H, W, D)
            'dot_measurements': torch.zeros(1, 60, 60, 60, dtype=torch.float32), # (C, H, W, D) - backward compatibility
            'ground_truth': torch.zeros(60, 60, 60, 2, dtype=torch.float32),     # (H, W, D, channels)
            'volumes': torch.zeros(60, 60, 60, 2, dtype=torch.float32),          # Add for training compatibility
            'metadata': torch.zeros(3, dtype=torch.long)
        }
        
        # Add empty tissue patches if using them
        if self.use_tissue_patches:
            px, py, pz = self.patch_size
            patch_size_flat = px * py * pz * 2  # 2 channels (mu_a, mu_s)
            sample['tissue_patches'] = torch.zeros(2, patch_size_flat, dtype=torch.float32)
        
        return sample


# ===============================================================================
# DATALOADER FACTORY FUNCTIONS
# ===============================================================================

def create_nir_dataloaders(data_dir: str = "../data",
                          batch_size: int = 32,
                          num_workers: int = 4,
                          patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
                          use_tissue_patches: bool = True,
                          random_seed: int = 42) -> Dict[str, DataLoader]:
    """
    Create train/validation/test DataLoaders for NIR phantom dataset.
    Enhanced with toggle functionality for Robin Dale's two-stage approach.
    
    This factory function creates properly configured DataLoaders with consistent
    settings across all splits. Implements cross-phantom shuffling for training
    and deterministic loading for validation/test.
    
    Args:
        data_dir (str): Path to phantom data directory
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        patch_size (tuple): Patch dimensions for tissue extraction
        use_tissue_patches (bool): Toggle for enabling tissue patch extraction
        random_seed (int): Random seed for reproducible splits
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train', 'val', 'test' dataloaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = NIRPhantomDataset(
            data_dir=data_dir,
            split=split,
            patch_size=patch_size,
            use_tissue_patches=use_tissue_patches,
            random_seed=random_seed
        )
        
        # Training uses shuffling for cross-phantom mixing
        # Validation/test use deterministic ordering for reproducible evaluation
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # GPU optimization
            drop_last=(split == 'train'),          # Drop incomplete batches in training
            persistent_workers=(num_workers > 0)   # Keep workers alive between epochs
        )
        
        dataloaders[split] = dataloader
        
        tissue_info = "with tissue patches" if use_tissue_patches else "baseline (no tissue patches)"
        logger.info(f"Created {split} DataLoader: {len(dataset)} samples, "
                   f"~{len(dataloader)} batches per epoch, {tissue_info}")
    
    return dataloaders


# ===============================================================================
# MAIN EXECUTION AND TESTING
# ===============================================================================

# DataLoader is ready for production use with Robin Dale's two-stage training!
