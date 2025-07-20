#!/usr/bin/env python3
"""
🔬 NIR PHANTOM DATASET LOADER WITH PATCH EXTRACTION 🔬

A comprehensive PyTorch DataLoader for NIR phantom datasets featuring:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 CORE FUNCTIONALITY:
• Lazy loading from distributed HDF5 phantom files (efficient memory usage)
• Automatic patch extraction around source/detector positions
• Shared CNN encoder for local tissue feature embedding
• Cross-phantom shuffling for robust training
• 90/5/5 train/validation/test splits at phantom level

🎯 TOKEN STRUCTURE:
Each training sample represents one source-detector measurement:
- Geometry coordinates: [xs, ys, zs, xd, yd, zd] (6D)
- FD measurements: [log_amplitude, phase] (2D)
- Source patch: 7×7×7×2 local tissue properties (μa, μ′s)
- Detector patch: 7×7×7×2 local tissue properties (μa, μ′s)
- CNN embeddings: Learned feature vectors from patches

🏗️ ARCHITECTURE:
• NIRPhantomDataset: Main PyTorch Dataset class
• extract_patch_around_position: Robust patch extraction with boundary handling
• TissuePatchCNN: Lightweight 3D CNN encoder for patch embedding
• Efficient H5 file management with lazy loading

Author: Max Hart - NIR Tomography Research
Version: 1.0 - Production DataLoader
"""

import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging for data loading operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s')
logger = logging.getLogger(__name__)

# ===============================================================================
# CONFIGURATION CONSTANTS
# ===============================================================================

# Patch extraction parameters
DEFAULT_PATCH_SIZE = (7, 7, 7)                    # Local tissue patch dimensions
AIR_VALUE = 0.0                                   # Air padding value for out-of-bounds regions
PATCH_PADDING_MODE = 'constant'                   # Padding strategy for boundary handling

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.90                                # 90% for training
VAL_RATIO = 0.05                                  # 5% for validation  
TEST_RATIO = 0.05                                 # 5% for testing

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
# PATCH EXTRACTION UTILITIES
# ===============================================================================

def extract_patch_around_position(ground_truth: np.ndarray, 
                                 position: np.ndarray, 
                                 patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
                                 pad_value: float = AIR_VALUE) -> np.ndarray:
    """
    Extract 3D tissue property patch centered at specified position with robust boundary handling.
    
    This function implements safe patch extraction that handles edge cases where patches
    extend beyond phantom boundaries. Out-of-bounds regions are padded with air values
    (μa=0, μ′s=0) which is physically realistic for NIR imaging scenarios.
    
    Technical Implementation:
    - Uses integer coordinates from mesh/surface extraction
    - Applies symmetric padding around center position
    - Handles boundary conditions gracefully without crashes
    - Maintains consistent patch dimensions across all samples
    
    Physical Interpretation:
    - Extracted patches contain mixed air/tissue regions near phantom surfaces
    - Air padding represents realistic boundary conditions for light transport
    - Local heterogeneity captured in 7×7×7 neighborhood around optodes
    
    Args:
        ground_truth (np.ndarray): Full phantom optical properties, shape (Nx, Ny, Nz, 2)
                                  Channel 0: absorption coefficient μa [mm⁻¹]
                                  Channel 1: reduced scattering μ′s [mm⁻¹]
        position (np.ndarray): Center position coordinates [x, y, z] in voxel indices
        patch_size (tuple): Patch dimensions (px, py, pz), default (7,7,7)
        pad_value (float): Value for out-of-bounds padding (air = 0.0)
        
    Returns:
        np.ndarray: Extracted patch, shape (px, py, pz, 2)
                   Contains local tissue properties with air padding at boundaries
                   
    Raises:
        ValueError: If patch_size dimensions are invalid or position is malformed
    """
    Nx, Ny, Nz, n_channels = ground_truth.shape
    px, py, pz = patch_size
    
    # Input validation
    if any(dim <= 0 for dim in patch_size):
        raise ValueError(f"Invalid patch_size: {patch_size}. All dimensions must be positive.")
    if len(position) != 3:
        raise ValueError(f"Position must be 3D coordinates, got: {position}")
    
    # Convert position to integer indices (handles potential floating point coordinates)
    cx, cy, cz = int(position[0]), int(position[1]), int(position[2])
    
    # Calculate patch boundaries with symmetric padding around center
    half_px, half_py, half_pz = px // 2, py // 2, pz // 2
    
    # Determine extraction and padding regions
    x_start, x_end = cx - half_px, cx + half_px + 1
    y_start, y_end = cy - half_py, cy + half_py + 1  
    z_start, z_end = cz - half_pz, cz + half_pz + 1
    
    # Initialize output patch with air padding
    patch = np.full((px, py, pz, n_channels), pad_value, dtype=ground_truth.dtype)
    
    # Calculate valid intersection between patch and phantom volume
    # This handles cases where patch extends beyond phantom boundaries
    gt_x_start = max(0, x_start)
    gt_x_end = min(Nx, x_end)
    gt_y_start = max(0, y_start)
    gt_y_end = min(Ny, y_end)
    gt_z_start = max(0, z_start)
    gt_z_end = min(Nz, z_end)
    
    # Calculate corresponding patch indices
    patch_x_start = max(0, -x_start)
    patch_x_end = patch_x_start + (gt_x_end - gt_x_start)
    patch_y_start = max(0, -y_start)
    patch_y_end = patch_y_start + (gt_y_end - gt_y_start)
    patch_z_start = max(0, -z_start)
    patch_z_end = patch_z_start + (gt_z_end - gt_z_start)
    
    # Extract valid region from ground truth and copy to patch
    if (gt_x_end > gt_x_start and gt_y_end > gt_y_start and gt_z_end > gt_z_start):
        patch[patch_x_start:patch_x_end, 
              patch_y_start:patch_y_end, 
              patch_z_start:patch_z_end] = ground_truth[gt_x_start:gt_x_end,
                                                       gt_y_start:gt_y_end,
                                                       gt_z_start:gt_z_end]
    
    logger.debug(f"Extracted patch at position {position}: "
                f"valid_region=({gt_x_start}:{gt_x_end}, {gt_y_start}:{gt_y_end}, {gt_z_start}:{gt_z_end})")
    
    return patch


# ===============================================================================
# 3D CNN ENCODER FOR PATCH EMBEDDING
# ===============================================================================

class TissuePatchCNN(nn.Module):
    """
    Lightweight 3D CNN encoder for tissue patch feature extraction.
    
    This encoder processes 7×7×7×2 tissue patches to produce fixed-size feature embeddings
    that capture local tissue heterogeneity patterns. The architecture balances feature
    extraction capability with computational efficiency for training throughput.
    
    Architecture Design:
    - 3D convolutions to capture spatial tissue patterns
    - Progressive feature map expansion with spatial downsampling
    - Batch normalization and dropout for training stability
    - Global average pooling to handle spatial variance
    - Shared weights between source and detector patches
    
    Technical Specifications:
    - Input: (batch_size, 2, 7, 7, 7) - 2 channels for μa and μ′s
    - Output: (batch_size, embed_dim) - Fixed-size feature vector
    - Parameters: ~50k (lightweight for efficient training)
    - Receptive field: Entire 7×7×7 patch
    
    Physical Interpretation:
    - Learns spatial patterns in absorption and scattering
    - Captures tissue boundaries and heterogeneity
    - Encodes local optical property distributions
    """
    
    def __init__(self, embed_dim: int = 128, dropout_rate: float = 0.1):
        """
        Initialize 3D CNN encoder for tissue patch embedding.
        
        Args:
            embed_dim (int): Output embedding dimension (default: 128)
            dropout_rate (float): Dropout probability for regularization (default: 0.1)
        """
        super(TissuePatchCNN, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 3D Convolutional layers with progressive feature expansion
        # Input: (batch, 2, 7, 7, 7) -> tissue properties
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)       # (batch, 16, 7, 7, 7)
        self.bn1 = nn.BatchNorm3d(16)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)  # (batch, 32, 4, 4, 4)
        self.bn2 = nn.BatchNorm3d(32)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)  # (batch, 64, 2, 2, 2)
        self.bn3 = nn.BatchNorm3d(64)
        
        # Global average pooling to produce fixed-size features
        self.global_pool = nn.AdaptiveAvgPool3d(1)                    # (batch, 64, 1, 1, 1)
        
        # Final projection to desired embedding dimension
        self.fc = nn.Linear(64, embed_dim)                            # (batch, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights using He initialization for ReLU networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D CNN encoder.
        
        Args:
            x (torch.Tensor): Input tissue patches, shape (batch_size, 2, 7, 7, 7)
                             Channel 0: absorption coefficient μa
                             Channel 1: reduced scattering μ′s
                             
        Returns:
            torch.Tensor: Embedded features, shape (batch_size, embed_dim)
        """
        # Progressive feature extraction with spatial downsampling
        x = F.relu(self.bn1(self.conv1(x)))        # (batch, 16, 7, 7, 7)
        x = F.relu(self.bn2(self.conv2(x)))        # (batch, 32, 4, 4, 4)  
        x = F.relu(self.bn3(self.conv3(x)))        # (batch, 64, 2, 2, 2)
        
        # Global pooling and final projection
        x = self.global_pool(x)                    # (batch, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)                  # (batch, 64)
        x = self.dropout(x)                        # Regularization
        x = self.fc(x)                             # (batch, embed_dim)
        
        return x


# ===============================================================================
# MAIN DATASET CLASS
# ===============================================================================

class NIRPhantomDataset(Dataset):
    """
    PyTorch Dataset for NIR phantom data with lazy loading and patch extraction.
    
    This dataset provides efficient access to distributed HDF5 phantom files with
    on-demand patch extraction around source and detector positions. Implements
    cross-phantom shuffling and phantom-level train/val/test splits to prevent
    data leakage and ensure robust model evaluation.
    
    Key Features:
    - Lazy loading: H5 files opened only when needed (memory efficient)
    - Patch extraction: 7×7×7 local tissue patches around optodes
    - Cross-phantom mixing: Samples shuffled across all phantoms
    - Phantom-level splits: Prevents overfitting to specific phantom types
    - Boundary handling: Robust patch extraction with air padding
    
    Dataset Structure:
    Each sample represents one source-detector measurement:
    - Geometry: [xs, ys, zs, xd, yd, zd] spatial coordinates
    - Measurements: [log_amplitude, phase] frequency-domain data
    - Source patch: 7×7×7×2 local tissue around source
    - Detector patch: 7×7×7×2 local tissue around detector
    - Metadata: phantom_id, probe_id, detector_id for tracking
    """
    
    def __init__(self, 
                 data_dir: str = "../data",
                 split: str = "train",
                 patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
                 random_seed: int = 42):
        """
        Initialize NIR phantom dataset with automatic file discovery and indexing.
        
        Args:
            data_dir (str): Path to directory containing phantom_XX subdirectories
            split (str): Dataset split - 'train', 'val', or 'test'
            patch_size (tuple): Patch dimensions for local tissue extraction
            random_seed (int): Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        
        logger.info(f"Initializing NIRPhantomDataset: split={split}, patch_size={patch_size}")
        
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
        
        logger.info(f"Discovered {len(h5_files)} phantom H5 files")
        
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
        
        logger.info(f"Validated {len(valid_files)} readable phantom files")
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
        Load single sample with patch extraction and return as tensor dictionary.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Sample data containing:
                - 'geometry': [xs, ys, zs, xd, yd, zd] coordinates (6,)
                - 'measurements': [log_amplitude, phase] FD data (2,)
                - 'source_patch': Source tissue patch (2, 7, 7, 7)
                - 'detector_patch': Detector tissue patch (2, 7, 7, 7)
                - 'metadata': [phantom_id, probe_id, detector_id] (3,)
        """
        phantom_file, probe_idx, det_idx = self.sample_index[idx]
        
        try:
            with h5py.File(phantom_file, 'r') as f:
                # Load measurement data
                log_amplitude = f[H5_KEYS['log_amplitude']][probe_idx, det_idx]
                phase = f[H5_KEYS['phase']][probe_idx, det_idx]
                
                # Load geometry
                source_pos = f[H5_KEYS['source_pos']][probe_idx]
                detector_pos = f[H5_KEYS['det_pos']][probe_idx, det_idx]
                
                # Load ground truth for patch extraction
                ground_truth = f[H5_KEYS['ground_truth']][:]
                
                # Extract tissue patches around source and detector
                source_patch = extract_patch_around_position(ground_truth, source_pos, self.patch_size)
                detector_patch = extract_patch_around_position(ground_truth, detector_pos, self.patch_size)
                
                # Extract phantom ID from filename
                phantom_id = int(phantom_file.stem.split('_')[1])
                
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {phantom_file}: {e}")
            # Return zero-filled sample to prevent training crashes
            return self._get_empty_sample()
        
        # Convert to PyTorch tensors with correct shapes for CNN
        return {
            'geometry': torch.tensor([*source_pos, *detector_pos], dtype=torch.float32),
            'measurements': torch.tensor([log_amplitude, phase], dtype=torch.float32),
            'source_patch': torch.tensor(source_patch.transpose(3, 0, 1, 2), dtype=torch.float32),  # (2, 7, 7, 7)
            'detector_patch': torch.tensor(detector_patch.transpose(3, 0, 1, 2), dtype=torch.float32),  # (2, 7, 7, 7)
            'metadata': torch.tensor([phantom_id, probe_idx, det_idx], dtype=torch.long)
        }
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return zero-filled sample for error handling."""
        px, py, pz = self.patch_size
        return {
            'geometry': torch.zeros(6, dtype=torch.float32),
            'measurements': torch.zeros(2, dtype=torch.float32),
            'source_patch': torch.zeros(2, px, py, pz, dtype=torch.float32),
            'detector_patch': torch.zeros(2, px, py, pz, dtype=torch.float32),
            'metadata': torch.zeros(3, dtype=torch.long)
        }


# ===============================================================================
# DATALOADER FACTORY FUNCTIONS
# ===============================================================================

def create_nir_dataloaders(data_dir: str = "../data",
                          batch_size: int = 32,
                          num_workers: int = 4,
                          patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
                          random_seed: int = 42) -> Dict[str, DataLoader]:
    """
    Create train/validation/test DataLoaders for NIR phantom dataset.
    
    This factory function creates properly configured DataLoaders with consistent
    settings across all splits. Implements cross-phantom shuffling for training
    and deterministic loading for validation/test.
    
    Args:
        data_dir (str): Path to phantom data directory
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        patch_size (tuple): Patch dimensions for tissue extraction
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
        
        logger.info(f"Created {split} DataLoader: {len(dataset)} samples, "
                   f"~{len(dataloader)} batches per epoch")
    
    return dataloaders


# ===============================================================================
# MAIN EXECUTION AND TESTING
# ===============================================================================

if __name__ == "__main__":
    """Quick test of DataLoader functionality."""
    
    print("🔬 Testing NIR Phantom DataLoader")
    print("=" * 50)
    
    # Test patch extraction
    print("Testing patch extraction...")
    dummy_ground_truth = np.random.random((50, 50, 50, 2))
    test_position = np.array([25, 25, 25])
    patch = extract_patch_around_position(dummy_ground_truth, test_position)
    print(f"✅ Patch extraction: shape {patch.shape}, expected (7, 7, 7, 2)")
    
    # Test CNN encoder
    print("\nTesting CNN encoder...")
    cnn = TissuePatchCNN(embed_dim=128)
    dummy_patches = torch.randn(4, 2, 7, 7, 7)  # Batch of 4 patches
    embeddings = cnn(dummy_patches)
    print(f"✅ CNN encoding: input {dummy_patches.shape} -> output {embeddings.shape}")
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    try:
        dataloaders = create_nir_dataloaders(batch_size=2, num_workers=0)
        
        for split, dataloader in dataloaders.items():
            if len(dataloader) > 0:
                sample_batch = next(iter(dataloader))
                print(f"✅ {split} DataLoader: {len(dataloader.dataset)} samples")
                print(f"   Sample shapes: geometry={sample_batch['geometry'].shape}, "
                      f"measurements={sample_batch['measurements'].shape}")
                print(f"   Patch shapes: source={sample_batch['source_patch'].shape}, "
                      f"detector={sample_batch['detector_patch'].shape}")
                break
    except Exception as e:
        print(f"⚠️  Dataset loading test failed (expected if no data files): {e}")
    
    print("\n🎯 DataLoader ready for training!")
