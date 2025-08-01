#!/usr/bin/env python3
"""
🔬 NIR PHANTOM DATASET LOADER 🔬

Streamlined PyTorch DataLoader for NIR phantom datasets featuring:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 CORE FUNCTIONALITY:
• Lazy loading from distributed HDF5 phantom files (efficient memory usage)
• Automatic tissue patch extraction around source/detector positions
• Cross-phantom shuffling for robust training
• 90/5/5 train/validation/test splits at phantom level
• Toggle functionality for two-stage training approach

🎯 DATA AUGMENTATION VIA MEASUREMENT SUBSAMPLING:
• Phantoms contain 1000 measurements (50 sources × 20 detectors - optimized probe placement)
• Randomly subsample 256 measurements per training batch for data augmentation
• Provides 3.9x more training combinations while maintaining pipeline compatibility
• Different subsets each epoch enhance model generalization

🎯 TWO-STAGE TRAINING SUPPORT:
• Stage 1: Full ground truth volumes for CNN autoencoder training
• Stage 2: Ground truth + optional tissue patches for transformer training
• Clean toggle switches throughout pipeline (use_tissue_patches parameter)

🏗️ ARCHITECTURE:
• NIRPhantomDataset: Main PyTorch Dataset class with toggle support
• Efficient H5 file management with lazy loading
• Phantom-level train/val/test splits to prevent data leakage

Author: Max Hart - NIR Tomography Research
Version: 2.1 - Optimized Probe Placement & Data Augmentation Integration
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
from code.utils.logging_config import get_data_logger
logger = get_data_logger(__name__)

# ===============================================================================
# CONFIGURATION CONSTANTS
# ===============================================================================

# Phantom data structure parameters
DEFAULT_PHANTOM_SHAPE = (64, 64, 64)             # Standard phantom dimensions (Nx, Ny, Nz)
DEFAULT_N_MEASUREMENTS = 256                      # Number of measurements for training (subsampled from generated data)
DEFAULT_N_GENERATED_MEASUREMENTS = 1000          # Number of measurements generated per phantom (50 sources × 20 detectors)
DEFAULT_N_TRAINING_MEASUREMENTS = 256            # Number of measurements subsampled for training (data augmentation)
DEFAULT_NIR_FEATURE_DIMENSION = 8                # Features per NIR measurement (log_amp + phase + 6 coords)
DEFAULT_OPTICAL_CHANNELS = 2                     # Optical property channels (absorption + scattering)

# Voxel and coordinate parameters (must match data_simulator.py)
VOXEL_SIZE_MM = 2.0                               # Physical size of each voxel in millimeters
PHANTOM_SIZE_MM = 128                             # Total phantom size in millimeters (64 voxels × 2mm each)

# Patch extraction parameters
DEFAULT_PATCH_SIZE = (7, 7, 7)                   # Local tissue patch dimensions
AIR_VALUE = 0.0                                   # Air padding value for out-of-bounds regions

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.90                                # 90% for training
VAL_RATIO = 0.05                                  # 5% for validation  
TEST_RATIO = 0.05                                 # 5% for testing

# Dataset size requirements
MINIMUM_PHANTOMS_REQUIRED = 100                  # Minimum phantoms needed for reliable 90/5/5 split

# DataLoader configuration parameters
DEFAULT_BATCH_SIZE = 32                           # Default batch size for individual measurements
DEFAULT_PHANTOM_BATCH_SIZE = 4                   # Default batch size for complete phantoms  
DEFAULT_NUM_WORKERS = 4                          # Default number of DataLoader worker processes
DEFAULT_RANDOM_SEED = 42                         # Default random seed for reproducible splits

# Memory and performance optimization
ENABLE_PIN_MEMORY = True                          # Enable GPU memory pinning when CUDA available
DROP_INCOMPLETE_BATCHES = True                    # Drop incomplete batches during training
ENABLE_PERSISTENT_WORKERS = True                 # Keep workers alive between epochs

# HDF5 dataset keys (must match data_simulator.py output)
H5_KEYS = {
    'log_amplitude': 'log_amplitude',             # (N_measurements,) log-amplitude measurements
    'phase': 'phase',                             # (N_measurements,) phase measurements  
    'source_pos': 'source_positions',             # (N_measurements, 3) source coordinates
    'det_pos': 'detector_positions',              # (N_measurements, 3) detector coordinates
    'ground_truth': 'ground_truth',               # (Nx, Ny, Nz, 2) optical property maps
    'tissue_labels': 'tissue_labels'              # (Nx, Ny, Nz) tissue type labels
}


# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def convert_physical_to_voxel_coordinates(physical_coords: np.ndarray) -> np.ndarray:
    """
    Convert physical coordinates (mm) to voxel indices.
    
    Physical coordinates are stored as mm from phantom center.
    Voxel indices range from 0 to phantom_shape-1.
    
    Args:
        physical_coords (np.ndarray): Physical coordinates in mm, shape (3,) or (N, 3)
        
    Returns:
        np.ndarray: Voxel indices, same shape as input
    """
    # Convert from center-based physical coords to voxel indices
    # Physical: [-64mm, +64mm] → Voxel: [0, 63]
    voxel_coords = (physical_coords + PHANTOM_SIZE_MM/2) / VOXEL_SIZE_MM
    
    # Round to nearest integer and ensure within bounds
    voxel_coords = np.round(voxel_coords).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, DEFAULT_PHANTOM_SHAPE[0] - 1)
    
    return voxel_coords


def extract_tissue_patch(ground_truth: np.ndarray, 
                        position: np.ndarray, 
                        patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE) -> np.ndarray:
    """
    Extract tissue patch around a position for tissue context encoder.
    
    Simplified extraction focused on compatibility with TissueContextEncoder.
    Extracts a local patch of optical properties around source/detector positions.
    
    Args:
        ground_truth (np.ndarray): Full phantom optical properties, shape (2, Nx, Ny, Nz) - channels-first
        position (np.ndarray): Center position coordinates [x, y, z] in physical coordinates (mm)
        patch_size (tuple): Patch dimensions (px, py, pz), default (7,7,7)
        
    Returns:
        np.ndarray: Flattened patch for tissue context encoder, shape (patch_size^3 * 2,)
    """
    # Convert physical coordinates (mm) to voxel indices
    voxel_position = convert_physical_to_voxel_coordinates(position)
    
    n_channels, Nx, Ny, Nz = ground_truth.shape
    px, py, pz = patch_size
    
    # Convert position to integer indices
    cx, cy, cz = int(voxel_position[0]), int(voxel_position[1]), int(voxel_position[2])
    
    # Calculate patch boundaries
    half_px, half_py, half_pz = px // 2, py // 2, pz // 2
    x_start, x_end = cx - half_px, cx + half_px + 1
    y_start, y_end = cy - half_py, cy + half_py + 1  
    z_start, z_end = cz - half_pz, cz + half_pz + 1
    
    # Initialize patch with air padding - channels-first format
    patch = np.full((n_channels, px, py, pz), AIR_VALUE, dtype=ground_truth.dtype)
    
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
    
    # Copy valid region - using channels-first indexing
    if (gt_x_end > gt_x_start and gt_y_end > gt_y_start and gt_z_end > gt_z_start):
        patch[:, patch_x_start:patch_x_end, 
              patch_y_start:patch_y_end, 
              patch_z_start:patch_z_end] = ground_truth[:, gt_x_start:gt_x_end,
                                                       gt_y_start:gt_y_end,
                                                       gt_z_start:gt_z_end]
    
    # Return flattened patch for tissue context encoder (flatten spatial dimensions but preserve channels)
    return patch.reshape(n_channels, -1).T.reshape(-1)  # Interleave channels: [ch0_vox0, ch1_vox0, ch0_vox1, ch1_vox1, ...]


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
                 random_seed: int = DEFAULT_RANDOM_SEED):
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
        Uses a clean 90/5/5 split and requires a minimum of 100 phantoms for reliable statistics.
        
        Args:
            phantom_files (List[Path]): All discovered phantom files
            split (str): Desired split ('train', 'val', 'test')
            random_seed (int): Random seed for reproducible splits
            
        Returns:
            List[Path]: Phantom files belonging to specified split
            
        Raises:
            ValueError: If insufficient phantoms for reliable 90/5/5 split
        """
        if not phantom_files:
            return []
        
        n_phantoms = len(phantom_files)
        
        # Enforce minimum phantom requirement for reliable training
        if n_phantoms < MINIMUM_PHANTOMS_REQUIRED:
            raise ValueError(
                f"Insufficient phantoms for reliable training! "
                f"Found {n_phantoms} phantoms, but need at least {MINIMUM_PHANTOMS_REQUIRED} "
                f"for a proper 90/5/5 split. Please generate more phantom data."
            )
        
        # Set random seed for reproducible splits
        np.random.seed(random_seed)
        
        # Shuffle phantom files to randomize assignment
        phantom_files = phantom_files.copy()
        np.random.shuffle(phantom_files)
        
        # Calculate split indices for 90/5/5 distribution
        train_end = int(n_phantoms * TRAIN_RATIO)
        val_end = train_end + int(n_phantoms * VAL_RATIO)
        
        # Assign phantoms to splits with clean logic
        if split == "train":
            selected_files = phantom_files[:train_end]
        elif split == "val":
            selected_files = phantom_files[train_end:val_end]
        elif split == "test":
            selected_files = phantom_files[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Log split statistics
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
                    # Read measurement count from HDF5 data (new 1:1 system)
                    if 'n_measurements' in f.attrs:
                        n_measurements = f.attrs['n_measurements']
                    else:
                        n_measurements = f[H5_KEYS['log_amplitude']].shape[0]
                    
                    # Each measurement is an independent source-detector pair
                    for measurement_idx in range(n_measurements):
                        sample_index.append((phantom_file, measurement_idx))
                            
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
                - 'metadata': [phantom_id, measurement_id] (2,)
        """
        phantom_file, measurement_idx = self.sample_index[idx]
        
        try:
            with h5py.File(phantom_file, 'r') as f:
                # Load actual NIR measurement data (log amplitude and phase) - new 1:1 system
                log_amplitude = f[H5_KEYS['log_amplitude']][measurement_idx]     # Shape: scalar for single measurement
                phase = f[H5_KEYS['phase']][measurement_idx]                     # Shape: scalar for single measurement
                
                # Load geometry data for the measurement - new 1:1 system
                source_pos = f[H5_KEYS['source_pos']][measurement_idx]           # Shape: (3,) for [x, y, z] in mm
                detector_pos = f[H5_KEYS['det_pos']][measurement_idx]            # Shape: (3,) for [x, y, z] in mm
                
                # Combine measurements into a feature vector
                # This is what the transformer should learn to map to the 512D CNN features
                # Single wavelength system (800nm), 1:1 source-detector pairing
                nir_measurements = np.array([
                    float(log_amplitude),  # 1 scalar value (log amplitude)
                    float(phase),          # 1 scalar value (phase)
                    source_pos[0],         # source x coordinate (mm)
                    source_pos[1],         # source y coordinate (mm)
                    source_pos[2],         # source z coordinate (mm)  
                    detector_pos[0],       # detector x coordinate (mm)
                    detector_pos[1],       # detector y coordinate (mm)
                    detector_pos[2]        # detector z coordinate (mm)
                ])  # Total: DEFAULT_NIR_FEATURE_DIMENSION features per measurement
                
                # Load ground truth volume (this is what we're trying to reconstruct)
                ground_truth = f[H5_KEYS['ground_truth']][:]  # Shape: (2, 64, 64, 64) - already channels-first
                # No transpose needed since data is already in channels-first format!
                
                # The key insight: NIR measurements → Transformer → 512D features → CNN decoder → Ground truth
                # Stage 1: Ground truth → CNN encoder → 512D → CNN decoder → Ground truth (learns compression)
                # Stage 2: NIR measurements → Transformer → 512D → Frozen CNN decoder → Ground truth
                
                # Extract tissue patches if enabled (for stage 2 enhanced training)
                tissue_patches = None
                if self.use_tissue_patches:
                    # Extract patches around source and detector (now 1:1 mapping)
                    # Note: positions are in physical coordinates (mm), conversion to voxel indices handled in extract_tissue_patch()
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
        
        # Build return dictionary with proper data flow
        sample = {
            # For Stage 1: Use ground truth as both input and target (CNN autoencoder training)
            # For Stage 2: Use NIR measurements as input, ground truth as target
            'measurements': torch.tensor(nir_measurements, dtype=torch.float32),        # (DEFAULT_NIR_FEATURE_DIMENSION,) NIR measurement vector
            'nir_measurements': torch.tensor(nir_measurements, dtype=torch.float32),    # (DEFAULT_NIR_FEATURE_DIMENSION,) Raw NIR data
            'ground_truth': torch.tensor(ground_truth, dtype=torch.float32),            # (DEFAULT_OPTICAL_CHANNELS, 64, 64, 64) Target volume  
            'volumes': torch.tensor(ground_truth, dtype=torch.float32),                 # (DEFAULT_OPTICAL_CHANNELS, 64, 64, 64) For Stage 1 input
            'metadata': torch.tensor([phantom_id, measurement_idx], dtype=torch.long)   # Updated for 1:1 measurement system
        }
        
        # Add tissue patches if enabled
        if self.use_tissue_patches and tissue_patches is not None:
            sample['tissue_patches'] = torch.tensor(tissue_patches, dtype=torch.float32)
        
        return sample
    
    def get_complete_phantom(self, phantom_idx: int) -> Dict[str, torch.Tensor]:
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

    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return zero-filled sample for error handling."""
        # Create empty sample with correct dimensions based on actual data (updated for 1:1 system)
        sample = {
            'measurements': torch.zeros(DEFAULT_NIR_FEATURE_DIMENSION, dtype=torch.float32),                # (DEFAULT_NIR_FEATURE_DIMENSION,) NIR measurement vector
            'nir_measurements': torch.zeros(DEFAULT_NIR_FEATURE_DIMENSION, dtype=torch.float32),            # (DEFAULT_NIR_FEATURE_DIMENSION,) Raw NIR data
            'ground_truth': torch.zeros(DEFAULT_OPTICAL_CHANNELS, *DEFAULT_PHANTOM_SHAPE, dtype=torch.float32),     # (channels, H, W, D)
            'volumes': torch.zeros(DEFAULT_OPTICAL_CHANNELS, *DEFAULT_PHANTOM_SHAPE, dtype=torch.float32),          # (channels, H, W, D)
            'metadata': torch.zeros(2, dtype=torch.long)  # Updated for [phantom_id, measurement_idx]
        }
        
        # Add empty tissue patches if using them
        if self.use_tissue_patches:
            px, py, pz = self.patch_size
            patch_size_flat = px * py * pz * DEFAULT_OPTICAL_CHANNELS  # DEFAULT_OPTICAL_CHANNELS channels (mu_a, mu_s)
            sample['tissue_patches'] = torch.zeros(2, patch_size_flat, dtype=torch.float32)
        
        return sample


# ===============================================================================
# PHANTOM BATCH DATASET CLASS (for multiprocessing compatibility)
# ===============================================================================

class PhantomBatchDataset(Dataset):
    """Custom dataset for phantom-level batching."""
    
    def __init__(self, data_dir: str, split: str, use_tissue_patches: bool, random_seed: int):
        self.base_dataset = NIRPhantomDataset(
            data_dir=data_dir,
            split=split,
            use_tissue_patches=use_tissue_patches,
            random_seed=random_seed
        )
        # We work at phantom level, not measurement level
        self.n_phantoms = len(self.base_dataset.phantom_files)
        
    def __len__(self):
        return self.n_phantoms
        
    def __getitem__(self, idx):
        return self.base_dataset.get_complete_phantom(idx)


# ===============================================================================
# DATALOADER FACTORY FUNCTIONS
# ===============================================================================

def create_nir_dataloaders(data_dir: str = "../data",
                          batch_size: int = DEFAULT_BATCH_SIZE,
                          num_workers: int = DEFAULT_NUM_WORKERS,
                          patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
                          use_tissue_patches: bool = True,
                          random_seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, DataLoader]:
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
            pin_memory=torch.cuda.is_available() and ENABLE_PIN_MEMORY,  # GPU optimization
            drop_last=(split == 'train') and DROP_INCOMPLETE_BATCHES,          # Drop incomplete batches in training
            persistent_workers=(num_workers > 0) and ENABLE_PERSISTENT_WORKERS   # Keep workers alive between epochs
        )
        
        dataloaders[split] = dataloader
        
        tissue_info = "with tissue patches" if use_tissue_patches else "baseline (no tissue patches)"
        logger.info(f"Created {split} DataLoader: {len(dataset)} samples, "
                   f"~{len(dataloader)} batches per epoch, {tissue_info}")
    
    return dataloaders


def create_phantom_dataloaders(data_dir: str = "../data",
                              batch_size: int = DEFAULT_PHANTOM_BATCH_SIZE,
                              num_workers: int = 0,  # Set to 0 to avoid multiprocessing issues
                              use_tissue_patches: bool = True,
                              random_seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for complete phantom batching (batches of complete phantoms).
    
    This creates a custom DataLoader that returns batches of complete phantoms,
    where each phantom contains 256 subsampled measurements from the generated 1000.
    Enables data augmentation through different random subsets each epoch.
    
    Args:
        data_dir (str): Path to phantom data directory
        batch_size (int): Number of phantoms per batch
        num_workers (int): Number of worker processes (set to 0 for main process only)
        use_tissue_patches (bool): Whether to include tissue patches
        random_seed (int): Random seed for splits
        
    Returns:
        Dict[str, DataLoader]: Dictionary with 'train', 'val', 'test' DataLoaders
    """
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = PhantomBatchDataset(data_dir, split, use_tissue_patches, random_seed)
        
        shuffle = (split == 'train') 
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() and ENABLE_PIN_MEMORY,
            drop_last=(split == 'train') and DROP_INCOMPLETE_BATCHES,
            persistent_workers=False  # Disable persistent workers when num_workers=0
        )
        
        dataloaders[split] = dataloader
        
        logger.info(f"Created phantom-level {split} DataLoader: {len(dataset)} phantoms, "
                   f"~{len(dataloader)} batches per epoch")
    
    return dataloaders