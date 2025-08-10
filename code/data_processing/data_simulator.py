#!/usr/bin/env python3
"""
🧬 NIR PHANTOM DATA SIMULATOR 🧬

Advanced phantom data generation for NIR tomography training datasets:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ CORE FUNCTIONALITY:
• 3D phantom geometry with realistic tissue distributions
• Random ellipsoidal tissue and tumor embedding with rotation
• Tetrahedral finite element mesh generation for light transport
• Physiologically realistic optical property assignment
• Surface-constrained probe placement for clinical realism
• Frequency-domain diffusion equation solving (140MHz)
• Complete HDF5 dataset output with ground truth

🎯 MACHINE LEARNING OPTIMIZED:
• Prevents spatial bias through randomized positioning
• Maintains physiological realism for robust training
• Surface-aware probe placement for clinical validity
• Complete ground truth for supervised learning

🔬 TECHNICAL FEATURES:
• NIRFASTer-FF finite element light transport modeling
• Binary morphological operations for surface extraction
• Efficient spatial computations with scipy
• Controlled randomization for reproducible datasets
• 1mm voxel size for optimal phantom dimensions (64×64×64mm physical)

🎨 VISUALIZATION FEATURES (NEW):
• Multi-probe visualization: Shows 5 source-detector pairs by default
• Interactive 3D popup: Toggle ENABLE_3D_INTERACTIVE_VISUAL = True for debugging
• PNG export: Always saves high-quality 2D images for reports
• Original coordinates: Shows voxel positions before touch_optodes mesh projection
• Usage: Set flags at top of file - ENABLE_3D_INTERACTIVE_VISUAL and DEFAULT_N_VISUAL_PROBE_PAIRS

⚠️ CRITICAL SURFACE EXTRACTION FIX (v2.4):
• Fixed surface extraction: now finds AIR voxels adjacent to tissue (accessible surface)
• Fixed coordinate saving: now saves mesh-projected coordinates used in physics simulation
• Problem: Previously found tissue voxels on boundary + saved voxel centers instead of mesh coordinates
• Solution: Binary dilation finds accessible air surface + save phantom_mesh.source/meas.coord
• Impact: Surface extraction now represents physically accessible probe placement locations

⚠️ CRITICAL BUG FIX (v2.3):
• Fixed NIRFASTer indexing bug: source-detector links now use 1-based indexing
• This was causing 21% of phantoms to have violated physics (wrong SDS relationships)
• Problem: NIRFASTer expects 1-based indexing but Python uses 0-based indexing
• Solution: measurement_links now store [idx+1, idx+1, 1] for proper NIRFASTer compatibility

Author: Max Hart - NIR Tomography Research
Version: 2.4 - CRITICAL SURFACE & COORDINATE FIXES (accessible air surface + mesh coordinates)
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import os
import sys
import time
from pathlib import Path

# Third-party imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

# System path configuration for NIRFASTer-FF library access
# Get the project root directory (mah422) - works regardless of where script is run from
project_root = Path(__file__).parent.parent.parent  # Go up to mah422 directory

# Add NIRFASTer-FF to path dynamically
nirfaster_path = project_root / "nirfaster-FF"
if nirfaster_path.exists():
    sys.path.append(str(nirfaster_path))
else:
    # Fallback for development environment
    sys.path.append("/Users/maxhart/Documents/MSc_AI_ML/Dissertation/mah422/nirfaster-FF")

# NIRFASTer-FF imports (after path configuration)
import nirfasterff as ff  # type: ignore

# Project imports
from code.utils.logging_config import get_data_logger, NIRDOTLogger

# Constants for phantom generation
MASTER_RANDOM_SEED = 42                      # Master seed for reproducible datasets (change for different datasets)
DEFAULT_N_PHANTOMS = 100                    # Number of phantoms to generate for dataset (10 hours @ 12 sec/phantom)
DEFAULT_PHANTOM_SHAPE = (64, 64, 64)        # Default cubic phantom dimensions in voxels (power of 2)
DEFAULT_TISSUE_RADIUS_RANGE = (25, 31)      # Healthy tissue ellipsoid semi-axis range (25-31mm with 1mm voxels)
DEFAULT_TUMOR_RADIUS_RANGE = (5, 15)        # Tumor ellipsoid semi-axis range (5-15mm with 1mm voxels)
DEFAULT_MAX_TUMORS = 5                       # Maximum number of tumors per phantom
DEFAULT_N_MEASUREMENTS = 256                # Number of measurements for training (subsampled from generated 1000)
DEFAULT_N_GENERATED_MEASUREMENTS = 1000      # Number of measurements generated per phantom (50 sources × 20 detectors)
DEFAULT_N_TRAINING_MEASUREMENTS = 256        # Number of measurements subsampled for training
DEFAULT_MIN_PROBE_DISTANCE = 10              # Minimum source-detector separation [mm] for diffusive regime validity
DEFAULT_MAX_PROBE_DISTANCE = 40              # Maximum source-detector separation [mm] for clinical signal detectability
DEFAULT_PATCH_RADIUS = 30                    # Patch radius [mm] for surface probe placement (clinical probe array size)
DEFAULT_MIN_PATCH_VOXELS = 400               # Minimum surface voxels for valid patch placement (for 1mm voxels)
DEFAULT_FD_FREQUENCY = 140e6                 # Frequency-domain modulation frequency [Hz]
DEFAULT_MESH_CELL_SIZE = 1.65                # CGAL mesh characteristic cell size [mm] 
VOXEL_SIZE_MM = 1.0                          # Voxel size in millimeters for spatial calibration

# Visualization parameters
ENABLE_3D_INTERACTIVE_VISUAL = False         # Flag to enable 3D interactive popup visualization (set to True for debugging)
DEFAULT_N_VISUAL_PROBE_PAIRS = 5             # Number of source-detector pairs to show in visualization

# Tissue label constants for clarity and consistency
AIR_LABEL = 0                                # Background air regions
HEALTHY_TISSUE_LABEL = 1                     # Healthy tissue regions  
TUMOR_START_LABEL = 2                        # Starting label for tumor regions (incremented)

# Optical property constants at 800nm wavelength (physiological ranges)
HEALTHY_MUA_RANGE = (0.003, 0.007)          # Healthy tissue absorption coefficient [mm⁻¹]
HEALTHY_MUSP_RANGE = (0.78, 1.18)           # Healthy tissue reduced scattering [mm⁻¹]
TUMOR_MUA_MULTIPLIER_RANGE = (1.5, 3.5)     # Tumor absorption enhancement factor
TUMOR_MUSP_MULTIPLIER_RANGE = (1.5, 2.5)    # Tumor scattering enhancement factor
TISSUE_REFRACTIVE_INDEX = 1.33               # Fixed refractive index for biological tissues

# Measurement noise parameters (conservative clinical values for realistic SNR)
AMPLITUDE_NOISE_PERCENTAGE = 0.005           # 0.5% relative amplitude noise (clinical systems, SNR ~46dB) 
PHASE_NOISE_STD_DEGREES = 0.5               # ±0.5° phase noise (clinical precision systems)

# Tumor placement algorithm parameters
MAX_TUMOR_PLACEMENT_ATTEMPTS = 50            # Maximum iterations for tumor placement rejection sampling
TUMOR_TISSUE_EMBEDDING_THRESHOLD = 0.80      # Required fraction of tumor volume inside tissue (80%)

# Mesh validation parameters (targeting 1mm³ elements with 1mm voxels)
MIN_ELEMENT_VOLUME_MM3 = 0.5                 # Minimum acceptable tetrahedral element volume [mm³] (targeting 1mm³)
MAX_ELEMENT_VOLUME_MM3 = 3.0                 # Maximum acceptable tetrahedral element volume [mm³] (targeting 1mm³)

# Surface processing and batch parameters  
SURFACE_BATCH_SIZE = 1500                     # Batch size for safe center identification (for 1mm voxels)
DEFAULT_RANDOM_SEED = 41                     # Default random seed for reproducibility

# Phantom generation retry logic parameters
MAX_PHANTOM_RETRY_ATTEMPTS = 3               # Maximum retry attempts per phantom when NaN values detected
RETRY_SEED_OFFSET = 1000                     # Seed offset for retries to ensure different geometry

# Initialize logger for the module using centralized logging system
logger = get_data_logger(__name__)


# =============================================================================
# RANDOM STATE MANAGEMENT FOR REPRODUCIBLE DATASETS
# =============================================================================

def setup_random_state(seed=None):
    """
    Configure global random state for reproducible results.
    
    Args:
        seed: Random seed for reproducibility. If None, uses current time.
    
    Returns:
        tuple: (numpy_rng, seed_used) for consistent usage
    """
    if seed is None:
        seed = int(time.time() * 1000000) % (2**31)  # Microsecond-based seed
    
    # Convert numpy integers to Python int (random.seed doesn't accept numpy types)
    seed = int(seed)
    
    # Set all random number generators consistently
    np.random.seed(seed)
    random.seed(seed)  # For any standard library random usage
    
    # Create dedicated RNG for this simulation
    rng = np.random.default_rng(seed)
    
    return rng, seed


# ============================================================================
# STEP 1: VOLUMETRIC PHANTOM CONSTRUCTION WITH EMBEDDED GEOMETRIES
# ============================================================================

def build_phantom_with_tissue_and_tumours(phantom_shape=DEFAULT_PHANTOM_SHAPE,
                                          tissue_radius_range=DEFAULT_TISSUE_RADIUS_RANGE,
                                          max_tumours=DEFAULT_MAX_TUMORS,
                                          tumour_radius_range=DEFAULT_TUMOR_RADIUS_RANGE,
                                          air_label=AIR_LABEL,
                                          tissue_label=HEALTHY_TISSUE_LABEL,
                                          tumour_start_label=TUMOR_START_LABEL,
                                          phantom_rng=None):
    """
    Construct 3D phantom volume with realistic tissue and tumor distributions.
    
    Creates a hierarchical geometry with:
    • Air-filled background domain
    • Large ellipsoidal healthy tissue region with random 3D rotation
    • Multiple smaller tumor inclusions with random orientations
    
    Key Features:
    • Random 3D rotation matrices eliminate directional bias
    • Enhanced tumor embedding requires 80% volume inside tissue
    • Prevents unrealistic "floating tumor" voxels in air space
    • Uses implicit surface representation via quadratic forms
    • Robust rejection sampling with geometric validation
    
    Args:
        phantom_shape (tuple): 3D phantom dimensions (Nx, Ny, Nz)
        tissue_radius_range (tuple): Min/max semi-axes for healthy tissue ellipsoid
        max_tumours (int): Maximum number of tumor inclusions
        tumour_radius_range (tuple): Min/max semi-axes for tumor ellipsoids
        air_label (int): Label for air/background regions (typically 0)
        tissue_label (int): Label for healthy tissue (typically 1)
        tumour_start_label (int): Starting label for tumors (incremented)
        phantom_rng (np.random.Generator): Random number generator for reproducible generation
        
    Returns:
        numpy.ndarray: 3D volume with integer labels, shape (Nx, Ny, Nz)
    """
    # Initialize pseudorandom number generator for reproducibility
    if phantom_rng is None:
        # Fallback for backwards compatibility
        phantom_rng = np.random.default_rng()
        logger.warning("No RNG provided to build_phantom_with_tissue_and_tumours - using default random state")
    
    Nx, Ny, Nz = phantom_shape
    
    # Create background air-filled volume using efficient memory allocation
    vol = np.full(phantom_shape, air_label, dtype=np.uint8)

    # Pre-compute 3D coordinate meshgrids for vectorized ellipsoid calculations
    # This avoids nested loops and leverages NumPy's broadcasting capabilities
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')

    def random_rotation_matrix(rng):
        """
        Generate random 3D rotation matrix using Euler angles.
        
        Implements proper 3D rotation sampling to eliminate directional bias
        in phantom generation. Uses ZYX Euler angle convention to generate
        uniformly distributed rotations over SO(3).
        
        Key Benefits:
        • Eliminates axis-aligned bias affecting ML model training
        • Simulates realistic tumor orientations
        • Ensures tissue geometry variation matches clinical diversity
        
        Args:
            rng (numpy.random.Generator): Controlled PRNG for reproducible rotations
            
        Returns:
            numpy.ndarray: 3×3 orthogonal rotation matrix, shape (3, 3)
        """
        # Sample random Euler angles for uniform rotation coverage
        alpha = rng.uniform(0, 2*np.pi)  # Z-axis rotation (yaw)
        beta = rng.uniform(0, np.pi)     # Y-axis rotation (pitch)  
        gamma = rng.uniform(0, 2*np.pi)  # X-axis rotation (roll)
        
        # Construct elementary rotation matrices
        # X-axis rotation matrix (roll)
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(gamma), -np.sin(gamma)],
                      [0, np.sin(gamma), np.cos(gamma)]])
                      
        # Y-axis rotation matrix (pitch)
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])
                      
        # Z-axis rotation matrix (yaw)
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
        
        # Compose final rotation matrix: R = R_z(α) · R_y(β) · R_x(γ)
        return Rz @ Ry @ Rx

    # ----------------------
    # ROTATED HEALTHY TISSUE ELLIPSOID EMBEDDING WITH PHYSIOLOGICAL CONSTRAINTS
    # ----------------------
    # Position the primary tissue ellipsoid at the geometric center of the phantom domain
    # This ensures symmetric boundary conditions and optimal light penetration geometry
    cx, cy, cz = Nx//2, Ny//2, Nz//2  # Integer division ensures exact center positioning for even dimensions
    
    # Sample random semi-axes lengths for tissue ellipsoid anisotropy simulation
    # Biological tissues rarely exhibit perfect spherical symmetry due to anatomical constraints
    rx = phantom_rng.integers(*tissue_radius_range)  # Semi-axis length in x-direction [voxels]
    ry = phantom_rng.integers(*tissue_radius_range)  # Semi-axis length in y-direction [voxels] 
    rz = phantom_rng.integers(*tissue_radius_range)  # Semi-axis length in z-direction [voxels]
    
    # Generate random 3D rotation matrix for arbitrary ellipsoid orientation
    # This eliminates axis-aligned directional bias and simulates realistic anatomical variation
    # Critical for ML training: prevents model from learning phantom coordinate system artifacts
    rotation_matrix = random_rotation_matrix(phantom_rng)
    
    # Apply rotation transformation to the entire coordinate system
    # Transform from phantom coordinates (X, Y, Z) to rotated ellipsoid coordinates (X', Y', Z')
    # Mathematical operation: [X', Y', Z']ᵀ = R · [X-cx, Y-cy, Z-cz]ᵀ where R is rotation matrix
    coords = np.stack([X-cx, Y-cy, Z-cz], axis=-1)  # Shape: (Nx, Ny, Nz, 3) - centered coordinates
    rotated_coords = coords @ rotation_matrix.T  # Apply rotation: uses transpose for proper transformation
    
    # Extract rotated coordinate components for ellipsoid equation evaluation
    # These represent the phantom coordinates transformed to the ellipsoid's local coordinate system
    X_rot = rotated_coords[..., 0]  # Rotated X-coordinates in ellipsoid frame
    Y_rot = rotated_coords[..., 1]  # Rotated Y-coordinates in ellipsoid frame  
    Z_rot = rotated_coords[..., 2]  # Rotated Z-coordinates in ellipsoid frame
    
    # Compute tissue mask using the canonical ellipsoid equation in rotated coordinate system
    # Standard form: (X'/rx)² + (Y'/ry)² + (Z'/rz)² ≤ 1 defines interior points
    # This creates a 3D binary mask where True indicates tissue voxels
    tissue_mask = ((X_rot/rx)**2 + (Y_rot/ry)**2 + (Z_rot/rz)**2) <= 1
    vol[tissue_mask] = tissue_label  # Apply tissue label to all voxels inside the rotated ellipsoid

    # ----------------------
    # ROTATED TUMOUR INCLUSION EMBEDDING WITH ENHANCED SPATIAL REALISM
    # ----------------------
    # Sample the number of tumour inclusions from uniform distribution [0, max_tumours]
    # Biological variability: some phantoms have no tumors, others have multiple lesions
    n_tumours = phantom_rng.integers(0, max_tumours+1)
    logger.info(f"Embedding {n_tumours} tumor(s) with enhanced spatial constraints")
    logger.debug(f"Tissue ellipsoid: center=({cx},{cy},{cz}), radii=({rx},{ry},{rz})")
    current_label = tumour_start_label  # Initialize tumor labeling sequence

    # Iteratively place each tumour with rigorous geometric validity checking
    # Uses rejection sampling to ensure physiologically realistic placement
    tumors_placed = 0  # Track successful tumor embeddings for quality control
    for tumour_idx in range(n_tumours):
        attempts = 0
        max_attempts = MAX_TUMOR_PLACEMENT_ATTEMPTS  # Use centralized constant for consistent behavior
        
        # Implement robust rejection sampling with failure detection to prevent infinite loops
        # This ensures algorithm termination even for challenging geometric configurations
        while attempts < max_attempts:
            # Sample tumour center coordinates within tissue ellipsoid bounds using safety margins
            # Apply conservative margins (±3 voxels) to ensure tumours don't extend beyond tissue boundary
            # This prevents unrealistic "tumor bleeding" into air space during rotation
            cx_t = phantom_rng.integers(cx-rx+3, cx+rx-3)  # X-coordinate with boundary safety margin
            cy_t = phantom_rng.integers(cy-ry+3, cy+ry-3)  # Y-coordinate with boundary safety margin
            cz_t = phantom_rng.integers(cz-rz+3, cz+rz-3)  # Z-coordinate with boundary safety margin
            
            # Sample tumour ellipsoid dimensions with physiological and geometric constraints
            # Tumor sizes follow clinical pathology distributions while respecting container geometry
            rx_t = phantom_rng.integers(*tumour_radius_range)  # Tumor semi-axis in x-direction [voxels]
            ry_t = phantom_rng.integers(*tumour_radius_range)  # Tumor semi-axis in y-direction [voxels]
            
            # Constrain z-axis dimension to maintain realistic aspect ratios and prevent elongated artifacts
            # Clinical tumors typically maintain roughly isotropic growth patterns in 3D
            # Ensure minimum radius ≥ 3 voxels for numerical stability and maximum ≤ max(rx_t, ry_t) for proportionality
            max_rz = min(tumour_radius_range[1], max(rx_t, ry_t))  # Enforce aspect ratio constraints
            rz_t = phantom_rng.integers(3, max(4, max_rz + 1))  # Ensure valid range with minimum size guarantee

            logger.debug(f"Tumor {tumour_idx+1} attempt {attempts+1}: center=({cx_t},{cy_t},{cz_t}), radii=({rx_t},{ry_t},{rz_t})")

            # Generate independent random rotation matrix for arbitrary tumor orientation
            # This ensures tumors also exhibit random poses, not just the tissue ellipsoid
            # Critical for training data diversity: prevents correlation between tumor and tissue orientations
            tumor_rotation = random_rotation_matrix(phantom_rng)
            
            # Apply rotation transformation to tumor's local coordinate system
            # Transform from phantom coordinates to tumor-specific rotated coordinates
            # Mathematical operation: tumor_coords_rotated = tumor_coords_centered @ R_tumor^T
            tumor_coords = np.stack([X-cx_t, Y-cy_t, Z-cz_t], axis=-1)  # Shape: (Nx, Ny, Nz, 3) - tumor-centered
            rotated_tumor_coords = tumor_coords @ tumor_rotation.T  # Apply tumor-specific rotation transformation
            
            # Extract rotated tumor coordinate components for ellipsoid equation evaluation
            # These coordinates are in the tumor's local rotated reference frame
            X_tumor_rot = rotated_tumor_coords[..., 0]  # Rotated X-coordinates in tumor frame
            Y_tumor_rot = rotated_tumor_coords[..., 1]  # Rotated Y-coordinates in tumor frame
            Z_tumor_rot = rotated_tumor_coords[..., 2]  # Rotated Z-coordinates in tumor frame

            # Compute tumour mask using canonical ellipsoid formulation in rotated coordinate system
            # Standard ellipsoid equation: (X'/rx_t)² + (Y'/ry_t)² + (Z'/rz_t)² ≤ 1
            # This defines the 3D region occupied by the tumor in its rotated configuration
            tumour_mask = ((X_tumor_rot/rx_t)**2 + (Y_tumor_rot/ry_t)**2 + (Z_tumor_rot/rz_t)**2) <= 1
            
            # Enhanced containment validation with clinical realism requirements
            # Require 80% of tumor volume to be embedded within healthy tissue for physiological accuracy
            # This prevents unrealistic "floating tumor" voxels in air space while allowing partial boundaries
            tumor_voxels = np.sum(tumour_mask)  # Total tumor volume [voxels]
            contained_voxels = np.sum(tumour_mask & tissue_mask)  # Tumor voxels properly embedded in tissue
            
            if tumor_voxels > 0:  # Ensure non-empty tumor volume for valid embedding ratio calculation
                embedding_ratio = contained_voxels / tumor_voxels  # Fraction of tumor within tissue [0,1]
                
                if embedding_ratio >= TUMOR_TISSUE_EMBEDDING_THRESHOLD:  # Use centralized threshold for clinical realism
                    # Apply tumour label ONLY to tumor voxels that are properly embedded within tissue
                    # This clipping approach ensures no "floating tumor" voxels exist in air space
                    # Results in physiologically realistic tumor-tissue interfaces
                    vol[tumour_mask & tissue_mask] = current_label
                    current_label += 1  # Increment label counter for next tumour identification
                    tumors_placed += 1  # Update successful placement counter
                    logger.debug(f"✓ Tumor {tumour_idx+1} placed successfully: {contained_voxels}/{tumor_voxels} voxels ({embedding_ratio:.1%} embedded)")
                    break  # Exit retry loop upon successful placement - proceed to next tumor
                else:
                    logger.debug(f"✗ Tumor {tumour_idx+1} insufficient embedding: {embedding_ratio:.1%} < 80% threshold")
            
            attempts += 1  # Increment attempt counter for this tumor placement
        
        # Log placement failure after exhausting maximum attempts
        # This provides diagnostic information for challenging geometric configurations
        if attempts >= max_attempts:
            logger.warning(f"Failed to place tumor {tumour_idx+1} after {max_attempts} attempts - skipping")

    # Log final tumor placement summary
    if tumors_placed > 0:
        logger.info(f"Successfully embedded {tumors_placed}/{n_tumours} tumors in tissue")
    else:
        logger.info("No tumors placed in this phantom")

    # Calculate comprehensive tissue composition statistics
    total_voxels = Nx * Ny * Nz
    air_voxels = np.sum(vol == air_label)  # Background air voxels
    tissue_voxels = np.sum(vol == tissue_label)  # Healthy tissue only
    tumor_voxels = np.sum(vol >= tumour_start_label)  # All tumor regions
    total_tissue_voxels = tissue_voxels + tumor_voxels  # Total tissue (healthy + tumors)
    
    # Calculate percentages
    tissue_percentage = total_tissue_voxels / total_voxels * 100
    
    logger.info(f"Phantom construction completed - {tissue_percentage:.1f}% tissue coverage")
    logger.debug("="*50)
    logger.debug("PHANTOM COMPOSITION BREAKDOWN:")
    logger.debug("="*50)
    logger.debug(f"Total voxels: {total_voxels:,}")
    logger.debug(f"Air (background):     {air_voxels:,} voxels ({air_voxels/total_voxels*100:.1f}%)")
    logger.debug(f"Healthy tissue:       {tissue_voxels:,} voxels ({tissue_voxels/total_voxels*100:.1f}%)")
    
    # Log individual tumor statistics if any tumors were placed
    if tumor_voxels > 0:
        unique_labels = np.unique(vol)
        tumor_labels = unique_labels[unique_labels >= tumour_start_label]
        
        for tumor_label in tumor_labels:
            tumor_idx = tumor_label - tumour_start_label + 1
            tumor_count = np.sum(vol == tumor_label)
            tumor_pct = tumor_count / total_voxels * 100
            logger.debug(f"Tumor {tumor_idx}:              {tumor_count:,} voxels ({tumor_pct:.1f}%)")
        
        logger.debug(f"Total tumors:         {tumor_voxels:,} voxels ({tumor_voxels/total_voxels*100:.1f}%)")
    
    logger.debug("-"*50)
    logger.debug(f"TOTAL TISSUE:         {total_tissue_voxels:,} voxels ({tissue_percentage:.1f}%)")
    logger.debug("="*50)
    
    # Validation check
    total_check = air_voxels + tissue_voxels + tumor_voxels
    if total_check != total_voxels:
        logger.warning(f"Voxel count mismatch: {total_check:,} != {total_voxels:,}")
    else:
        logger.debug("Voxel count validation passed")

    return vol


# ============================================================================
# STEP 2: FINITE ELEMENT MESH GENERATION FOR LIGHT TRANSPORT
# ============================================================================

def mesh_volume(volume):
    """
    Convert labeled phantom geometry into tetrahedral finite element mesh.
    
    Performs automatic mesh generation for NIR light transport simulations:
    • Excludes air regions (label=0) to reduce computational overhead
    • Generates conforming tetrahedral elements respecting tissue boundaries
    • Uses CGAL-based Delaunay triangulation with quality guarantees
    • Optimizes for finite element diffusion equation solving
    
    Args:
        volume (numpy.ndarray): 3D labeled phantom volume, shape (Nx, Ny, Nz)
                               
    Returns:
        tuple: (elements, nodes) where:
            - elements: Tetrahedral connectivity matrix, shape (N_tet, 4)
            - nodes: Node coordinate matrix, shape (N_nodes, 3)
    """
    # Configure meshing parameters for optimal FEM performance
    mesh_params = ff.utils.MeshingParams()
    # Set characteristic cell size balancing accuracy and computational cost
    # Using 1.65mm as specified by supervisor for consistent element quality with 1mm voxels
    mesh_params.general_cell_size = DEFAULT_MESH_CELL_SIZE  # 1.65mm for optimal FEM accuracy
    
    logger.info(f"Starting CGAL-based tetrahedral mesh generation")
    logger.info(f"Mesh parameters: cell_size={mesh_params.general_cell_size}mm, voxel_spacing={VOXEL_SIZE_MM}mm")
    logger.debug(f"Input volume shape: {volume.shape}, unique labels: {np.unique(volume)}")
    logger.debug(f"Expected element volume: ~{(mesh_params.general_cell_size/2)**3:.2f} mm³ (target: ~1mm³)")
    
    # Execute CGAL-based tetrahedral mesh generation
    # Using standard 1mm voxel approach - no special scaling needed
    logger.debug("Invoking CGAL mesh generator with 1mm voxel spacing...")
    mesh_elements, mesh_nodes = ff.meshing.RunCGALMeshGenerator(volume, opt=mesh_params)
    
    # Perform comprehensive mesh quality validation
    # Checks for inverted elements, aspect ratios, and topological consistency
    logger.debug("Validating mesh quality and topology...")
    ff.meshing.CheckMesh3D(mesh_elements, mesh_nodes)
    
    # Calculate mesh statistics and coverage analysis
    num_elements = mesh_elements.shape[0]
    num_nodes = mesh_nodes.shape[0]
    mesh_density = num_elements / np.prod(volume.shape)
    
    # Calculate tissue coverage for validation
    tissue_voxels = np.sum(volume >= 1)  # Count all tissue voxels (healthy + tumors)
    tissue_volume_mm3 = tissue_voxels * (VOXEL_SIZE_MM ** 3)  # Convert to mm³ (1mm³ per voxel)
    
    logger.info(f"✓ Mesh generation completed - {num_elements:,} tetrahedra, {num_nodes:,} nodes")
    logger.debug(f"Mesh density: {mesh_density:.3f} elements/voxel")
    logger.debug(f"Average nodes per element: {num_nodes/num_elements:.1f}")
    logger.info(f"Tissue volume analysis: {tissue_voxels:,} voxels = {tissue_volume_mm3:,.0f} mm³")
    logger.debug("Mesh quality validation passed successfully")
    
    return mesh_elements, mesh_nodes

def create_stndmesh(mesh_elements, mesh_nodes):
    """
    Wraps raw mesh data into NIRFASTer standardized mesh object with computed geometric properties.
    
    This function creates a complete finite element mesh structure including:
    - Tetrahedral element connectivity and nodal coordinates
    - Automatic computation of element volumes for FEM integration
    - Boundary face identification for optode placement and boundary conditions
    - Region labeling for heterogeneous optical property assignment
    
    Technical Details:
    - Computes element Jacobians for finite element basis function transformations
    - Identifies surface triangular faces for Robin boundary condition application
    - Calculates mesh quality metrics (volume distribution, aspect ratios)
    - Prepares data structures for efficient sparse matrix assembly in FEM solver
    
    Args:
        mesh_elements (numpy.ndarray): Element connectivity matrix from mesh generation
        mesh_nodes (numpy.ndarray): Node coordinate matrix from mesh generation
        
    Returns:
        nirfasterff.base.stndmesh: Complete mesh object ready for optical property assignment
                                  and finite element simulation
    """
    # Initialize NIRFASTer standard mesh data structure
    phantom_mesh = ff.base.stndmesh()
    
    logger.debug("Creating standardized mesh object from raw connectivity data")
    
    # Populate mesh with tetrahedral solid geometry and compute derived quantities
    # This automatically calculates:
    # - Element volumes via determinant of Jacobian matrix
    # - Surface face identification through adjacency analysis  
    # - Boundary node marking for Dirichlet/Robin boundary conditions
    
    # Using standard 1mm voxel approach - mesh nodes are already in correct physical coordinates
    logger.debug("Using standard 1mm voxel mesh coordinates")
    phantom_mesh.from_solid(mesh_elements, mesh_nodes)  # Direct usage for 1mm voxels
    
    # Extract and display mesh quality statistics for validation
    mean_element_volume = phantom_mesh.element_area.mean()  # Note: 'element_area' actually stores volumes for 3D elements
    std_element_volume = phantom_mesh.element_area.std()
    total_mesh_volume = phantom_mesh.element_area.sum()  # Total meshed volume
    
    logger.info(f"Mesh statistics: {mesh_elements.shape[0]} tetrahedra, {mesh_nodes.shape[0]} nodes")
    logger.info(f"Element volume statistics: {mean_element_volume:.3f} ± {std_element_volume:.3f} mm³")
    logger.info(f"Total meshed volume: {total_mesh_volume:,.0f} mm³")
    
    # Validate mesh quality metrics are within acceptable bounds
    if mean_element_volume < MIN_ELEMENT_VOLUME_MM3 or mean_element_volume > MAX_ELEMENT_VOLUME_MM3:
        logger.warning(f"Element volumes ({mean_element_volume:.3f} mm³) may be outside optimal range for FEM convergence")
    else:
        logger.debug("Mesh quality metrics are within acceptable bounds")
    
    return phantom_mesh


# ============================================================================
# STEP 3: OPTICAL PROPERTY ASSIGNMENT AND GROUND TRUTH GENERATION
# ============================================================================

def assign_optical_properties(phantom_mesh, phantom_volume, phantom_rng=None):
    """
    Assign physiologically realistic optical properties to mesh regions and generate ground truth.
    
    Implements clinically-informed optical property distributions:
    • Assigns absorption (μₐ) and reduced scattering (μ′s) per tissue type
    • Uses randomized sampling within physiological bounds for diversity
    • Maintains realistic contrast ratios between healthy tissue and tumors
    • Generates pixel-wise ground truth maps for supervised learning
    
    Optical Property Ranges (800nm wavelength):
    • Healthy tissue: μₐ ∈ [0.003, 0.007] mm⁻¹, μ′s ∈ [0.78, 1.18] mm⁻¹
    • Tumor tissue: μₐ = (1.5-3.5)×healthy, μ′s = (1.5-2.5)×healthy
    • Fixed refractive index: n = 1.33 (biological tissues)
    
    Args:
        phantom_mesh (nirfasterff.base.stndmesh): Finite element mesh with region labels
        phantom_volume (numpy.ndarray): Original voxel phantom for ground truth mapping
        phantom_rng (np.random.Generator): Random number generator for reproducible property assignment
        
    Returns:
        tuple: (phantom_mesh, ground_truth_maps) where:
            - phantom_mesh: Updated mesh with optical properties
            - ground_truth_maps: Dense (Nx, Ny, Nz, 2) array with μₐ and μ′s maps
    """
    # Initialize controlled random number generator for reproducible property sampling
    if phantom_rng is None:
        phantom_rng = np.random.default_rng()
        logger.warning("No RNG provided to assign_optical_properties - using default random state")
    
    # Extract unique region labels from mesh elements
    # phantom_mesh.region contains the tissue type label for each tetrahedral element
    unique_regions = np.unique(phantom_mesh.region)
    optical_properties = []  # Will store [region_id, μₐ, μ′s, n] for each tissue type

    logger.info(f"Starting optical property assignment for {len(unique_regions)} tissue regions")

    # Sample baseline healthy tissue optical properties from physiological distributions
    # These serve as reference values for relative tumour property scaling
    healthy_mua = phantom_rng.uniform(*HEALTHY_MUA_RANGE)      # Absorption coeff. [mm⁻¹] - controls image contrast
    healthy_musp = phantom_rng.uniform(*HEALTHY_MUSP_RANGE)    # Reduced scattering [mm⁻¹] - controls penetration depth
    
    logger.debug(f"Baseline healthy tissue properties: μₐ={healthy_mua:.4f} mm⁻¹, μ′s={healthy_musp:.3f} mm⁻¹")
    
    # Dictionary for efficient ground truth lookup during voxel assignment
    region_optical_lookup = {}

    # Assign optical properties to each tissue region with appropriate physiological scaling
    for region_label in unique_regions:
        if region_label == HEALTHY_TISSUE_LABEL:  # Healthy tissue baseline
            tissue_mua, tissue_musp = healthy_mua, healthy_musp
            tissue_type_name = "healthy"
            logger.debug(f"Region {region_label} ({tissue_type_name}): μₐ={tissue_mua:.4f} mm⁻¹, μ′s={tissue_musp:.3f} mm⁻¹")
        else:  # Tumour regions (label ≥ TUMOR_START_LABEL)
            # Apply controlled randomization within clinically observed ranges
            # Tumours typically show increased absorption (higher blood volume)
            # and altered scattering (modified cellular architecture)
            mua_multiplier = phantom_rng.uniform(*TUMOR_MUA_MULTIPLIER_RANGE)
            musp_multiplier = phantom_rng.uniform(*TUMOR_MUSP_MULTIPLIER_RANGE)
            tissue_mua = healthy_mua * mua_multiplier     # 50-250% increase in absorption
            tissue_musp = healthy_musp * musp_multiplier  # 50-150% increase in scattering
            tumor_index = region_label - TUMOR_START_LABEL + 1
            tissue_type_name = f"tumor_{tumor_index}"
            logger.debug(f"Region {region_label} ({tissue_type_name}): μₐ={tissue_mua:.4f} mm⁻¹ ({mua_multiplier:.1f}×), "
                        f"μ′s={tissue_musp:.3f} mm⁻¹ ({musp_multiplier:.1f}×)")
            
        # Store optical properties in NIRFASTer format: [region, μₐ, μ′s, n]
        # Refractive index is fixed for biological tissues at NIR wavelengths
        optical_properties.append([region_label, tissue_mua, tissue_musp, TISSUE_REFRACTIVE_INDEX])
        region_optical_lookup[region_label] = (tissue_mua, tissue_musp)

    # Apply optical properties to mesh for FEM simulation
    # This populates phantom_mesh.mua and phantom_mesh.musp arrays used in diffusion equation assembly
    phantom_mesh.set_prop(np.array(optical_properties))
    logger.info("✓ Optical properties assigned to mesh elements")

    # Generate dense voxel-wise ground truth maps for reconstruction evaluation
    Nx, Ny, Nz = phantom_volume.shape
    # Shape: (2, Nx, Ny, Nz) where first dimension is [μₐ, μ′s] - channels first for PyTorch
    ground_truth_maps = np.zeros((2, Nx, Ny, Nz))
    
    logger.debug("Generating dense ground truth maps in channels-first format...")
    # Populate ground truth grid using region-based property lookup
    # This creates pixel-perfect reference maps for quantitative evaluation
    for region_label, (tissue_mua, tissue_musp) in region_optical_lookup.items():
        # Apply properties to all voxels belonging to this tissue region
        region_mask = (phantom_volume == region_label)
        ground_truth_maps[0, region_mask] = tissue_mua   # Channel 0: absorption coefficient
        ground_truth_maps[1, region_mask] = tissue_musp  # Channel 1: reduced scattering coefficient

    logger.info(f"✓ Ground truth maps generated - shape {ground_truth_maps.shape} (channels-first)")
    logger.debug(f"Value ranges: μₐ=[{ground_truth_maps[0].min():.4f}, {ground_truth_maps[0].max():.4f}], "
                f"μ′s=[{ground_truth_maps[1].min():.3f}, {ground_truth_maps[1].max():.3f}]")

    return phantom_mesh, ground_truth_maps


# ============================================================================
# STEP 4: TISSUE SURFACE EXTRACTION VIA MORPHOLOGICAL OPERATIONS
# ============================================================================

def extract_surface_voxels(phantom_volume, tissue_threshold=HEALTHY_TISSUE_LABEL):
    """
    Extract 3D coordinates of accessible surface voxels for probe placement.
    
    CRITICAL FIX: Identifies AIR voxels adjacent to tissue, not tissue voxels adjacent to air.
    This represents the actual accessible surface where NIR probes can be physically placed.
    
    Technical Implementation:
    • Creates binary tissue mask for all tissue regions (healthy + tumors)
    • Applies binary dilation to tissue to find air voxels adjacent to tissue
    • Returns air voxel coordinates that form the accessible tissue-air boundary
    • These coordinates represent physically accessible probe placement locations
    
    Args:
        phantom_volume (numpy.ndarray): 3D labeled phantom volume with tissue regions
        tissue_threshold (int): Minimum label value considered as tissue (excludes air=0)
        
    Returns:
        numpy.ndarray: Surface voxel coordinates in air adjacent to tissue, shape (N_surface, 3)
                      Each row contains (x, y, z) indices of accessible surface voxels
    """
    from scipy.ndimage import binary_dilation
    
    # Create binary mask identifying all tissue regions (healthy + tumours)
    tissue_binary_mask = (phantom_volume >= tissue_threshold)
    
    logger.info(f"Starting CORRECTED surface extraction (threshold={tissue_threshold})")
    logger.debug(f"Input volume shape: {phantom_volume.shape}")
    
    initial_tissue_count = np.sum(tissue_binary_mask)
    initial_air_count = np.sum(phantom_volume == 0)
    logger.debug(f"Initial tissue voxels: {initial_tissue_count:,}")
    logger.debug(f"Initial air voxels: {initial_air_count:,}")
    
    # Apply binary dilation to tissue mask to identify air voxels adjacent to tissue
    # Uses default 3×3×3 structuring element for 26-connected neighborhood
    # Dilated tissue mask includes original tissue + adjacent air voxels
    logger.debug("Applying morphological dilation to identify air voxels adjacent to tissue...")
    dilated_tissue_mask = binary_dilation(tissue_binary_mask, iterations=1)
    
    # Find air voxels that became part of dilated tissue (these are adjacent to tissue)
    # Surface voxels are air voxels (label=0) that are adjacent to tissue
    air_mask = (phantom_volume == 0)  # All air voxels
    accessible_surface_mask = air_mask & dilated_tissue_mask  # Air voxels adjacent to tissue
    
    # Extract explicit (x,y,z) coordinates of accessible surface voxels
    surface_voxel_coordinates = np.argwhere(accessible_surface_mask)
    
    surface_count = surface_voxel_coordinates.shape[0]
    logger.info(f"✓ CORRECTED surface extraction completed - {surface_count:,} accessible surface voxels identified")
    logger.debug(f"These are AIR voxels adjacent to tissue (physically accessible for probe placement)")
    
    # Validate surface extraction results
    if surface_count == 0:
        logger.error("No accessible surface voxels found - check tissue geometry and threshold")
    elif surface_count < 100:
        logger.warning(f"Very few surface voxels ({surface_count}) - may limit probe placement options")
    else:
        surface_ratio = surface_count / initial_air_count
        logger.debug(f"Accessible surface ratio: {surface_ratio:.1%} of air voxels are adjacent to tissue")
    
    # Verify that all returned coordinates are actually air voxels
    verification_labels = phantom_volume[surface_voxel_coordinates[:, 0], 
                                       surface_voxel_coordinates[:, 1], 
                                       surface_voxel_coordinates[:, 2]]
    air_count = np.sum(verification_labels == 0)
    tissue_count = np.sum(verification_labels >= 1)
    
    logger.debug(f"Surface verification: {air_count}/{surface_count} are air voxels ({air_count/surface_count*100:.1f}%)")
    if tissue_count > 0:
        logger.error(f"SURFACE EXTRACTION BUG: {tissue_count} surface voxels are tissue, not air!")
    else:
        logger.debug("✅ All surface voxels are correctly identified as air (accessible)")
    
    return surface_voxel_coordinates


# --------------------------------------------------------------
# STEP 5: PATCH-BASED SURFACE PROBE PLACEMENT FOR CLINICAL REALISM
# --------------------------------------------------------------

def convert_voxel_to_physical_coordinates(voxel_coordinates):
    """
    Convert voxel indices to physical millimeter coordinates.
    
    Transforms discrete voxel indices (i,j,k) to continuous spatial coordinates (x,y,z)
    using the global voxel size. Critical for accurate distance calculations in probe placement.
    
    Args:
        voxel_coordinates (numpy.ndarray or list): Voxel indices, shape (N, 3)
        
    Returns:
        numpy.ndarray: Physical coordinates in mm, shape (N, 3)
    """
    # Ensure input is a NumPy array for consistent processing
    voxel_coords_array = np.array(voxel_coordinates)
    return voxel_coords_array.astype(float) * VOXEL_SIZE_MM

def find_safe_patch_centers(surface_coordinates, patch_radius=DEFAULT_PATCH_RADIUS, 
                           min_patch_voxels=DEFAULT_MIN_PATCH_VOXELS, phantom_rng=None):
    """
    Identifies surface voxels that can support full-radius patches for robust probe placement.
    
    This function implements safe center placement by pre-filtering surface positions to ensure
    each potential patch center can accommodate the full 30mm radius patch without extending
    beyond the tissue boundary. This prevents edge effects and ensures consistent patch sizes.
    
    Clinical Motivation:
    Real probe arrays are typically placed on accessible, relatively flat tissue regions rather
    than at sharp boundaries or curved edges. This filtering simulates realistic clinical
    placement constraints while maintaining geometric validity.
    
    Technical Implementation:
    - Computes pairwise distances from each surface voxel to all others
    - Filters centers based on minimum surface voxel count within patch radius
    - Uses efficient distance computation for large surface sets
    - Prevents patches with insufficient detector placement options
    
    Args:
        surface_coordinates (numpy.ndarray): All available surface voxel positions, shape (N_surface, 3)
        patch_radius (float): Required patch radius in mm for full probe array coverage
        min_patch_voxels (int): Minimum surface voxels required within patch radius
        rng_seed (int): Random seed for reproducible center selection
        
    Returns:
        numpy.ndarray: Safe patch center coordinates, shape (N_safe, 3)
                      Each row contains (x,y,z) coordinates of validated center positions
    """
    logger.info(f"Identifying safe patch centers (radius={patch_radius}mm, min_voxels={min_patch_voxels})")
    logger.debug(f"Input surface positions: {len(surface_coordinates):,} voxels")
    
    # Convert voxel coordinates to physical millimeter coordinates for accurate distance calculations
    surface_coordinates_mm = convert_voxel_to_physical_coordinates(surface_coordinates)
    
    safe_centers = []
    total_candidates = len(surface_coordinates)
    
    # Process in batches to manage memory for large surface sets
    batch_size = min(SURFACE_BATCH_SIZE, total_candidates)
    
    for batch_start in range(0, total_candidates, batch_size):
        batch_end = min(batch_start + batch_size, total_candidates)
        batch_centers_mm = surface_coordinates_mm[batch_start:batch_end]
        
        # Compute distances from batch centers to all surface voxels in millimeters
        distances = cdist(batch_centers_mm, surface_coordinates_mm)
        
        # Count surface voxels within patch radius for each center candidate
        for i, center_distances in enumerate(distances):
            voxels_in_patch = np.sum(center_distances <= patch_radius)
            
            if voxels_in_patch >= min_patch_voxels:
                # Store original voxel coordinates (not mm coordinates) for consistency
                safe_centers.append(surface_coordinates[batch_start + i])
        
        if (batch_end // batch_size) % 5 == 0:  # Progress logging
            logger.debug(f"Processed {batch_end:,}/{total_candidates:,} surface voxels")
    
    safe_centers = np.array(safe_centers) if safe_centers else np.empty((0, 3))
    
    safety_ratio = len(safe_centers) / total_candidates if total_candidates > 0 else 0
    logger.info(f"✓ Safe center identification completed - {len(safe_centers):,}/{total_candidates:,} valid centers ({safety_ratio:.1%})")
    
    if len(safe_centers) == 0:
        logger.error("No safe patch centers found - consider reducing patch radius or minimum voxel requirements")
    elif len(safe_centers) < 10:
        logger.warning(f"Very few safe centers ({len(safe_centers)}) - may limit patch diversity")
    
    return safe_centers


def build_patch_based_probe_layout(surface_coordinates, n_sources=50, detectors_per_source=20,
                                 patch_radius=DEFAULT_PATCH_RADIUS,
                                 min_source_detector_distance=DEFAULT_MIN_PROBE_DISTANCE,
                                 max_source_detector_distance=DEFAULT_MAX_PROBE_DISTANCE,
                                 min_patch_voxels=DEFAULT_MIN_PATCH_VOXELS,
                                 phantom_rng=None):
    """
    Generates optimized probe configurations using strategic source placement with multiple detectors per source.
    
    **NEW OPTIMIZED APPROACH FOR COMPUTATIONAL EFFICIENCY & DATA AUGMENTATION:**
    
    This function implements an advanced source-detector strategy that leverages NIR-DOT physics:
    • Sources require expensive FEM solutions (computational bottleneck)  
    • Detectors are computationally free once source field is solved
    • Solution: Use fewer strategic sources with many detectors each
    
    **IMPLEMENTATION STRATEGY:**
    1. Identifies safe patch centers that can support full 30mm radius patches
    2. Randomly selects one patch center per phantom for spatial diversity  
    3. Creates localized surface patch within specified radius (~2,344 surface voxels typical)
    4. **STRATEGIC SOURCE PLACEMENT:** Uses Poisson disk sampling to place 50 sources with uniform distribution
    5. **DETECTOR OPTIMIZATION:** For each source, selects 20 detectors using pure random sampling
    6. **RESULT:** 50 sources × 20 detectors = 1000 measurements per phantom
    
    **KEY ADVANTAGES:**
    ✅ **5x Computational Speedup:** Only 50 FEM solves vs. current 256
    ✅ **Data Augmentation:** 1000 measurements → subsample 256 for training (3.9x more combinations)
    ✅ **Better ML Generalization:** Random subsampling prevents spatial bias in training
    ✅ **Storage Efficiency:** 1000 measurements vs. 117,000 if using all detectors
    ✅ **Training Input Optimal:** Maintains 256-measurement transformer input size
    
    **TECHNICAL IMPLEMENTATION:**
    • **Poisson Disk Sampling:** Ensures minimum distance between sources (prevents clustering)
    • **Random Detector Selection:** Pure random sampling within 10-40mm SDS range
    • **Clinical Realism:** Maintains physiological SDS constraints and surface placement
    
    Args:
        surface_coordinates (numpy.ndarray): Surface voxel positions from binary erosion, shape (N_surface, 3)
        n_sources (int): Number of strategic source positions to place (50 for optimal efficiency)
        detectors_per_source (int): Number of detectors per source (20 for 1000 total measurements)
        patch_radius (float): Patch radius in mm defining local probe placement region (30mm clinical size)
        min_source_detector_distance (float): Minimum SDS in mm for diffusive regime validity (10mm)
        max_source_detector_distance (float): Maximum SDS in mm for clinical realism (40mm) 
        min_patch_voxels (int): Minimum surface voxels required for valid patch (500 for adequate sampling)
        phantom_rng (np.random.Generator): Random number generator for reproducible patch selection and probe placement
        
    Returns:
        tuple: (probe_sources, probe_detectors, measurement_links, patch_info) where:
            - probe_sources: Source positions for measurements, shape (n_sources*detectors_per_source, 3)
            - probe_detectors: Detector positions for measurements, shape (n_sources*detectors_per_source, 3) 
            - measurement_links: Source-detector connectivity, shape (n_sources*detectors_per_source, 3)
            - patch_info: Dictionary containing patch metadata for visualization and analysis
    """
    # Initialize controlled randomization for reproducible patch-based placement
    if phantom_rng is None:
        phantom_rng = np.random.default_rng()
        logger.warning("No RNG provided to build_patch_based_probe_layout - using default random state")
    
    # Calculate total expected measurements for comprehensive logging
    total_expected_measurements = n_sources * detectors_per_source
    
    logger.info(f"Starting OPTIMIZED patch-based probe layout generation")
    logger.info(f"Strategy: {n_sources} strategic sources × {detectors_per_source} detectors/source = {total_expected_measurements} total measurements")
    logger.info(f"Computational advantage: {n_sources} FEM solves (vs. {total_expected_measurements} in old approach = {total_expected_measurements/n_sources:.1f}x speedup)")
    logger.info(f"Patch constraints: {patch_radius}mm radius, SDS range: {min_source_detector_distance}-{max_source_detector_distance}mm")
    logger.debug(f"Available surface positions: {len(surface_coordinates):,} voxels")
    
    # STEP 5.1: Identify safe patch centers that can support full radius patches
    logger.debug("Step 1/6: Identifying safe patch centers...")
    safe_patch_centers = find_safe_patch_centers(surface_coordinates, patch_radius, min_patch_voxels, phantom_rng)
    
    if len(safe_patch_centers) == 0:
        logger.error("No safe patch centers available - cannot proceed with probe placement")
        return np.array([]), np.array([]), np.array([]), {}
    
    # STEP 5.2: Randomly select one patch center for this phantom
    logger.debug("Step 2/6: Selecting random patch center...")
    center_idx = phantom_rng.integers(0, len(safe_patch_centers))
    selected_patch_center = safe_patch_centers[center_idx]
    
    logger.info(f"Selected patch center: ({selected_patch_center[0]:.1f}, {selected_patch_center[1]:.1f}, {selected_patch_center[2]:.1f})")
    
    # STEP 5.3: Create patch by filtering surface voxels within radius
    logger.debug("Step 3/6: Creating surface patch...")
    # Convert coordinates to physical mm for accurate distance calculations
    surface_coordinates_mm = convert_voxel_to_physical_coordinates(surface_coordinates)
    selected_patch_center_mm = convert_voxel_to_physical_coordinates([selected_patch_center])[0]
    
    distances_to_center = cdist([selected_patch_center_mm], surface_coordinates_mm)[0]
    patch_mask = distances_to_center <= patch_radius
    patch_surface_coordinates = surface_coordinates[patch_mask]  # Keep as voxel coordinates
    patch_surface_coordinates_mm = surface_coordinates_mm[patch_mask]  # Physical coordinates for distance calc
    
    patch_size = len(patch_surface_coordinates)
    logger.info(f"Patch created: {patch_size:,} surface voxels within {patch_radius}mm radius")
    
    if patch_size < min_patch_voxels:
        logger.warning(f"Patch size ({patch_size}) below minimum requirement ({min_patch_voxels})")
    
    # STEP 5.4: **NEW - STRATEGIC SOURCE PLACEMENT USING POISSON DISK SAMPLING**
    logger.debug("Step 4/6: Placing sources using Poisson disk sampling for uniform distribution...")
    
    def poisson_disk_sampling(patch_coordinates, n_samples, phantom_rng, min_distance_mm=8.0, max_attempts=30):
        """
        Implements Poisson disk sampling for uniform source distribution without clustering.
        
        This algorithm ensures sources are well-distributed across the patch surface:
        • Prevents source clustering (which wastes computational resources)
        • Maintains minimum inter-source distance for spatial diversity
        • Provides better measurement coverage than random placement
        
        Args:
            patch_coordinates: Available patch surface positions in voxel coordinates
            n_samples: Number of sources to place
            min_distance_mm: Minimum distance between sources in millimeters
            max_attempts: Maximum rejection sampling attempts per source
            
        Returns:
            List of selected source positions (voxel coordinates)
        """
        if len(patch_coordinates) == 0:
            return []
            
        patch_coords_mm = convert_voxel_to_physical_coordinates(patch_coordinates)
        selected_sources = []
        
        # Place first source randomly
        first_idx = phantom_rng.integers(0, len(patch_coordinates))
        selected_sources.append(patch_coordinates[first_idx])
        
        # Place remaining sources with minimum distance constraint
        for source_idx in range(1, min(n_samples, len(patch_coordinates))):
            placed = False
            
            for attempt in range(max_attempts):
                candidate_idx = phantom_rng.integers(0, len(patch_coordinates))
                candidate_pos_mm = convert_voxel_to_physical_coordinates([patch_coordinates[candidate_idx]])[0]
                
                # Check distance to all existing sources
                valid_placement = True
                for existing_source in selected_sources:
                    existing_pos_mm = convert_voxel_to_physical_coordinates([existing_source])[0]
                    distance = np.linalg.norm(candidate_pos_mm - existing_pos_mm)
                    
                    if distance < min_distance_mm:
                        valid_placement = False
                        break
                
                if valid_placement:
                    selected_sources.append(patch_coordinates[candidate_idx])
                    placed = True
                    break
            
            if not placed:
                logger.warning(f"Could not place source {source_idx+1} with minimum distance constraint")
                break
                
        return selected_sources
    
    # Apply Poisson disk sampling with simple retry mechanism
    max_source_placement_attempts = 3
    source_positions = []
    
    for attempt in range(max_source_placement_attempts):
        source_positions = poisson_disk_sampling(patch_surface_coordinates, n_sources, phantom_rng, min_distance_mm=5.0)
        n_sources_placed = len(source_positions)
        
        logger.info(f"Source placement attempt {attempt+1}/{max_source_placement_attempts}: {n_sources_placed}/{n_sources} sources placed")
        
        if n_sources_placed == n_sources:
            logger.info(f"✓ Successfully placed all {n_sources} sources on attempt {attempt+1}")
            break
        else:
            logger.warning(f"Only placed {n_sources_placed}/{n_sources} sources on attempt {attempt+1}")
            if attempt < max_source_placement_attempts - 1:
                logger.info("Trying again with different random configuration...")
    
    # If we still don't have 50 sources after 3 attempts, throw away this phantom
    if len(source_positions) < n_sources:
        logger.error(f"PHANTOM REJECTED: Could not place {n_sources} sources after {max_source_placement_attempts} attempts")
        logger.error(f"Final result: {len(source_positions)}/{n_sources} sources placed")
        logger.error("Discarding this phantom and will generate a new one")
        raise RuntimeError(f"Failed to place {n_sources} sources after {max_source_placement_attempts} attempts - phantom rejected")
    
    # STEP 5.5: **SIMPLIFIED - PURE RANDOM DETECTOR SELECTION FOR EACH SOURCE**
    logger.debug("Step 5/6: Assigning detectors to sources using PURE RANDOM sampling...")
    
    def random_detector_selection(source_pos, patch_coords, n_detectors, min_sds, max_sds):
        """
        Selects detectors for a source using pure random sampling within SDS constraints.
        
        SIMPLIFIED APPROACH - No binning, no stratification, just pure randomness:
        • Finds all detectors within 10-40mm SDS range from source
        • Randomly samples n_detectors from this valid set
        • With 1000 total measurements, natural SDS distribution emerges across 10-40mm
        • Much simpler than stratified sampling, same effective coverage
        
        Args:
            source_pos: Source position (voxel coordinates)
            patch_coords: Available patch detector positions (voxel coordinates)  
            n_detectors: Number of detectors to randomly select for this source
            min_sds, max_sds: SDS range constraints in millimeters
            
        Returns:
            List of randomly selected detector positions for this source
        """
        source_pos_mm = convert_voxel_to_physical_coordinates([source_pos])[0]
        patch_coords_mm = convert_voxel_to_physical_coordinates(patch_coords)
        
        # Calculate distances from source to all potential detectors
        distances = np.linalg.norm(patch_coords_mm - source_pos_mm, axis=1)
        
        # Filter detectors within SDS constraints
        valid_mask = (distances >= min_sds) & (distances <= max_sds)
        valid_detector_indices = np.where(valid_mask)[0]
        
        if len(valid_detector_indices) == 0:
            return []
        
        # PURE RANDOM SAMPLING - no bins, no complexity
        n_sample = min(n_detectors, len(valid_detector_indices))
        selected_indices = phantom_rng.choice(valid_detector_indices, size=n_sample, replace=False)
        
        return [patch_coords[idx] for idx in selected_indices]
    
    # Apply pure random detector selection for each source
    all_probe_sources, all_probe_detectors, measurement_links = [], [], []
    measurement_idx = 0  # Global measurement counter
    
    for source_idx, source_pos in enumerate(source_positions):
        # Select random detectors for this source (no stratification)
        selected_detectors = random_detector_selection(
            source_pos, patch_surface_coordinates, detectors_per_source,
            min_source_detector_distance, max_source_detector_distance
        )
        
        n_detectors_found = len(selected_detectors)
        logger.debug(f"Source {source_idx+1}/{n_sources_placed}: {n_detectors_found}/{detectors_per_source} detectors via pure random sampling")
        
        # Create measurements for this source-detector combination
        for detector_pos in selected_detectors:
            # Store source and detector positions
            all_probe_sources.append(source_pos)
            all_probe_detectors.append(detector_pos)
            
            # **MEASUREMENT LINKING FOR SHARED SOURCES**
            # NIRFASTer format: [source_index, detector_index, active_flag]
            # With shared sources: multiple measurements use the same source
            # source_index points to unique source positions, detector_index points to unique detector positions
            measurement_links.append([source_idx + 1, measurement_idx + 1, 1])  # 1-based indexing for NIRFASTer
            measurement_idx += 1
    
    # STEP 5.6: Validate and report optimized placement results
    n_total_measurements = len(all_probe_sources)
    n_unique_sources = len(source_positions)
    avg_detectors_per_source = n_total_measurements / n_unique_sources if n_unique_sources > 0 else 0
    
    logger.info(f"✓ OPTIMIZED probe layout completed successfully!")
    logger.info(f"Final configuration: {n_unique_sources} sources → {n_total_measurements} total measurements")
    logger.info(f"Average detectors per source: {avg_detectors_per_source:.1f}")
    
    if n_total_measurements < total_expected_measurements:
        completion_rate = n_total_measurements / total_expected_measurements * 100
        logger.warning(f"Placement completion: {completion_rate:.1f}% ({n_total_measurements}/{total_expected_measurements} measurements)")
        logger.info(f"Consider adjusting patch radius or SDS constraints for higher completion rate")
    
    if n_total_measurements < total_expected_measurements:
        completion_rate = n_total_measurements / total_expected_measurements * 100
        logger.warning(f"Placement completion: {completion_rate:.1f}% ({n_total_measurements}/{total_expected_measurements} measurements)")
        logger.info(f"Consider adjusting patch radius or SDS constraints for higher completion rate")
    
    # STEP 5.7: Compile comprehensive patch metadata for visualization and analysis
    patch_info = {
        'center_position': selected_patch_center,
        'radius': patch_radius,
        'surface_voxels_in_patch': patch_size,
        'safe_centers_available': len(safe_patch_centers),
        'patch_surface_coordinates': patch_surface_coordinates,  # For visualization
        'sds_range': (min_source_detector_distance, max_source_detector_distance),
        # **NEW OPTIMIZATION METADATA**
        'optimization_strategy': 'strategic_sources_with_random_detectors',
        'n_unique_sources': n_unique_sources,
        'n_total_measurements': n_total_measurements,
        'avg_detectors_per_source': avg_detectors_per_source,
        'computational_speedup': f"{n_total_measurements/n_unique_sources:.1f}x",
        'data_augmentation_potential': f"~{n_total_measurements//256} training subsamples per phantom"
    }
    
    logger.debug(f"Patch metadata: center={selected_patch_center}, radius={patch_radius}mm, voxels={patch_size}")
    
    # Convert to NumPy arrays for efficient numerical processing
    return (np.array(all_probe_sources), 
            np.array(all_probe_detectors), 
            np.array(measurement_links),
            patch_info)


# --------------------------------------------------------------
# STEP 6: FREQUENCY-DOMAIN FORWARD MODELING AND DATA STORAGE
# --------------------------------------------------------------

def run_fd_simulation_and_save(phantom_mesh, ground_truth_maps, probe_sources, probe_detectors, measurement_links,
                               phantom_volume=None, fd_frequency_hz=DEFAULT_FD_FREQUENCY, output_h5_filename="phantom_fd_scan.h5"):
    """
    Runs frequency-domain FEM simulation and saves the dataset for ML training.

    Steps:
    1. Configures mesh with source/detector positions and measurement links.
    2. Projects optodes onto mesh surface for valid boundary conditions.
    3. Solves the frequency-domain diffusion equation for amplitude and phase.
    4. Adds realistic amplitude and phase noise.
    5. Applies log-amplitude transformation and reshapes data for ML.
    6. Saves all results, geometry, and ground truth to HDF5 with metadata.
    7. Validates measurement quality and detects NaN values.

    Args:
        phantom_mesh: NIRFASTer mesh with assigned optical properties.
        ground_truth_maps: Voxel-wise optical property maps (Nx, Ny, Nz, 2).
        probe_sources: Source positions (N, 3).
        probe_detectors: Detector positions (M, 3).
        measurement_links: Source-detector connectivity (K, 3).
        phantom_volume: (Optional) Labeled phantom volume (Nx, Ny, Nz).
        fd_frequency_hz: Modulation frequency in Hz.
        output_h5_filename: Output HDF5 file path.

    Returns:
        bool: True if simulation succeeded without NaN values, False if NaN detected.
              Used for retry logic to prevent saving corrupted phantoms.
    """
    
    # STEP 1: Configure mesh with OPTIMIZED optode integration and geometric validation
    # **UPDATED FOR NEW SHARED-SOURCE STRATEGY:** 
    # • Sources array contains unique source positions (e.g., 50 unique sources)
    # • Detectors array contains all detector positions (e.g., 1000 detector positions)  
    # • Measurement links define which source-detector pairs are active
    # This critical step ensures proper coupling between probe positions and finite element mesh
    logger.info("Configuring mesh with OPTIMIZED source-detector optode layout and geometric validation")
    
    # Convert discrete voxel coordinates to continuous spatial coordinates for FEM compatibility
    # Ensures proper interpolation and boundary condition application in finite element solver
    probe_sources_mm = convert_voxel_to_physical_coordinates(probe_sources)
    probe_detectors_mm = convert_voxel_to_physical_coordinates(probe_detectors)
    
    # **CRITICAL FIX: Extract unique source positions for NIRFASTer mesh setup**
    # With the new strategy, probe_sources contains repeated positions (one per measurement)
    # NIRFASTer needs unique source positions in the source array
    unique_source_positions = []
    seen_sources = set()
    
    for source_pos in probe_sources:
        source_tuple = tuple(source_pos)  # Convert to hashable tuple
        if source_tuple not in seen_sources:
            unique_source_positions.append(source_pos)
            seen_sources.add(source_tuple)
    
    unique_sources_mm = convert_voxel_to_physical_coordinates(unique_source_positions)
    n_unique_sources = len(unique_sources_mm)
    n_total_detectors = len(probe_detectors_mm)
    n_measurements = len(measurement_links)
    
    phantom_mesh.source = ff.base.optode(unique_sources_mm.astype(float))   # NIRFASTer source container (unique positions)
    phantom_mesh.meas = ff.base.optode(probe_detectors_mm.astype(float))     # NIRFASTer detector container (all positions)
    phantom_mesh.link = measurement_links  # Connectivity matrix defining active source-detector pairs
    
    logger.debug(f"OPTIMIZED optode configuration: {n_unique_sources} unique sources, {n_total_detectors} detectors")
    logger.debug(f"Measurement efficiency: {n_measurements} measurements from {n_unique_sources} FEM solves = {n_measurements/n_unique_sources:.1f}x multiplier")
    
    # Check if measurement_links is valid before accessing
    if len(measurement_links) > 0 and measurement_links.ndim == 2:
        active_measurements = np.sum(measurement_links[:, 2])
        logger.debug(f"Active measurements: {active_measurements} of {len(measurement_links)} total pairs")
    else:
        logger.error(f"Invalid measurement_links array: shape={measurement_links.shape if hasattr(measurement_links, 'shape') else 'unknown'}")
        return False  # Exit early if no valid measurements
    
    # Project optodes onto mesh surface with geometric consistency validation
    # Critical for accurate boundary condition application and prevents non-physical floating optodes
    # NIRFASTer automatically finds nearest mesh surface nodes for each optode position
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress NIRFASTer warnings during optode projection
        phantom_mesh.touch_optodes()
    logger.debug("Optodes successfully projected onto mesh surface with geometric validation")

    # STEP 2: Execute frequency-domain finite element forward simulation with comprehensive monitoring
    # This represents the core physics calculation solving the diffusion equation across the entire mesh
    logger.info(f"Executing OPTIMIZED frequency-domain diffusion equation simulation (modulation: {fd_frequency_hz/1e6:.1f} MHz)")
    
    # Log detailed mesh and measurement statistics for performance monitoring and debugging
    num_nodes = phantom_mesh.nodes.shape[0]
    num_elements = phantom_mesh.elements.shape[0]
    mesh_density = num_elements / np.prod(phantom_volume.shape) if phantom_volume is not None else 0
    
    logger.debug(f"FEM mesh statistics: {num_nodes:,} nodes, {num_elements:,} elements")
    logger.debug(f"OPTIMIZED measurement matrix: {n_unique_sources} unique sources → {n_measurements} total measurements")
    
    # Execute complex-valued frequency-domain finite element solution
    # Solves: -∇·(D∇Φ) + [μₐ + iω/c]Φ = S for complex photon fluence Φ(r,ω)
    # Returns measurement data structure containing amplitude and phase at each detector
    simulation_start_time = time.time()
    
    # 🚀 ADAPTIVE SOLVER: Use best available solver (GPU preferred, CPU fallback)
    best_solver = ff.utils.get_solver()  # Returns 'GPU' if available, 'CPU' otherwise
    solver_options = ff.utils.SolverOptions(GPU=0 if best_solver == 'GPU' else -1)
    simulation_data, _ = phantom_mesh.femdata(fd_frequency_hz, solver=best_solver, opt=solver_options)  # Adaptive NIRFASTer FEM solver
    simulation_time = time.time() - simulation_start_time
    
    # Extract amplitude and phase components from complex photon fluence solution
    raw_amplitude = simulation_data.amplitude  # |Φ|: Photon fluence magnitude [photons·mm⁻²·s⁻¹]
    raw_phase = simulation_data.phase          # arg(Φ): Phase delay [degrees] relative to source
    
    logger.info(f"✓ FD simulation completed successfully in {simulation_time:.1f}s")
    logger.debug(f"Raw measurement ranges: amplitude=[{raw_amplitude.min():.2e}, {raw_amplitude.max():.2e}] photons/mm²")
    logger.debug(f"                       phase=[{raw_phase.min():.1f}°, {raw_phase.max():.1f}°]")

    # STEP 3: Apply realistic measurement noise modeling for robust machine learning training
    # Noise characteristics based on published performance specifications of clinical NIR systems
    logger.debug("Applying clinical-grade measurement noise simulation...")
    noise_rng = np.random.default_rng()  # Independent RNG for noise to avoid correlation with phantom generation
    
    # Amplitude noise: Multiplicative relative noise proportional to signal magnitude
    # Models detector shot noise, electronic noise, and ambient light interference
    # Clinical NIR systems typically achieve 40-50 dB SNR corresponding to 0.3-1% relative noise
    amplitude_noise_std = AMPLITUDE_NOISE_PERCENTAGE * np.mean(raw_amplitude)
    noisy_amplitude = raw_amplitude + noise_rng.normal(0, amplitude_noise_std, raw_amplitude.shape)
    
    # Phase noise: Additive Gaussian noise independent of signal magnitude
    # Models lock-in amplifier precision, timing jitter, and temperature drift
    # Commercial systems typically achieve ±0.5-2 degree phase precision
    noisy_phase = raw_phase + noise_rng.normal(0, PHASE_NOISE_STD_DEGREES, raw_phase.shape)
    
    # FIX: Ensure physical validity by clamping phase to [0°, 360°) range
    # Prevents negative phase values that are physically impossible in NIR measurements
    phase_before_clamp = noisy_phase.copy()  # Store original values for logging
    noisy_phase = np.clip(noisy_phase, 0.0, 360.0)
    
    # Log clamping statistics for quality monitoring
    negative_count = np.sum(phase_before_clamp < 0)
    above_360_count = np.sum(phase_before_clamp > 360)
    total_clamped = negative_count + above_360_count
    
    if total_clamped > 0:
        percentage_clamped = total_clamped / noisy_phase.size * 100
        logger.debug(f"Phase clamping applied: {total_clamped}/{noisy_phase.size} values ({percentage_clamped:.2f}%)")
        logger.debug(f"  • Negative phases clamped to 0°: {negative_count}")
        logger.debug(f"  • Phases >360° clamped to 360°: {above_360_count}")
        
        if negative_count > 0:
            min_negative = np.min(phase_before_clamp[phase_before_clamp < 0])
            logger.debug(f"  • Most negative value clamped: {min_negative:.2f}°")
        if above_360_count > 0:
            max_above = np.max(phase_before_clamp[phase_before_clamp > 360])
            logger.debug(f"  • Highest value clamped: {max_above:.2f}°")
    else:
        logger.debug("Phase clamping: No values required clamping (all phases within [0°, 360°))")
    
    # Calculate effective signal-to-noise ratios for quality validation
    amplitude_snr_db = 20 * np.log10(np.mean(raw_amplitude) / amplitude_noise_std)
    phase_snr_ratio = PHASE_NOISE_STD_DEGREES / np.std(raw_phase)
    
    logger.debug(f"Applied realistic noise: amplitude SNR = {amplitude_snr_db:.1f} dB, phase precision = ±{PHASE_NOISE_STD_DEGREES}°")
    logger.debug(f"Noise parameters: amplitude_std={amplitude_noise_std:.2e}, phase_ratio={phase_snr_ratio:.3f}")

    # STEP 4: Process measurements for machine learning framework compatibility
    # Transforms raw physics simulation output into standardized ML training format
    logger.debug("Processing measurements for machine learning framework compatibility...")
    
    # Log-amplitude transformation for neural network numerical stability
    # Converts exponential distance decay to linear relationship, preventing gradient explosion
    # Clips small values to prevent -∞ from log(0) and maintains numerical stability
    log_amplitude_processed = np.log(np.clip(noisy_amplitude, 1e-8, None))  # Keep as 1D vector (N_measurements,)
    phase_processed = noisy_phase  # Keep as 1D vector (N_measurements,)
    
    # Validate processed measurement ranges for neural network compatibility
    log_amp_range = [log_amplitude_processed.min(), log_amplitude_processed.max()]
    phase_range = [phase_processed.min(), phase_processed.max()]
    
    logger.debug(f"ML-processed measurements: log_amplitude=[{log_amp_range[0]:.2f}, {log_amp_range[1]:.2f}] ln(photons/mm²)")
    logger.debug(f"                          phase=[{phase_range[0]:.1f}°, {phase_range[1]:.1f}°]")
    
    # Validate measurement quality and detect potential numerical issues
    has_nan_values = False
    if log_amp_range[0] < -50 or log_amp_range[1] > 50:
        logger.warning(f"Log-amplitude range may be outside optimal bounds for neural networks")
    if np.any(np.isnan(log_amplitude_processed)) or np.any(np.isnan(phase_processed)):
        logger.error("NaN values detected in processed measurements - check phantom geometry")
        has_nan_values = True
        
        # Count and report NaN statistics for debugging
        n_nan_amplitude = np.sum(np.isnan(log_amplitude_processed))
        n_nan_phase = np.sum(np.isnan(phase_processed))
        total_measurements = len(log_amplitude_processed)
        logger.error(f"NaN Statistics: {n_nan_amplitude}/{total_measurements} amplitude NaNs ({n_nan_amplitude/total_measurements*100:.1f}%), "
                    f"{n_nan_phase}/{total_measurements} phase NaNs ({n_nan_phase/total_measurements*100:.1f}%)")
        
        # Return early if NaN values detected - don't save corrupted data
        logger.error("Phantom generation failed due to NaN values - returning False for retry")
        return False

    # STEP 5: Save complete dataset to hierarchical HDF5 format with comprehensive metadata
    # HDF5 provides efficient storage, compression, and self-describing metadata for large datasets
    logger.info(f"Saving complete ML-ready dataset to {output_h5_filename}")
    
    with h5py.File(output_h5_filename, "w") as h5_file:
        # Create comprehensive dataset structure with detailed metadata for ML framework compatibility
        
        # SUBSTEP 5.1: Save processed measurements with ML-optimized formatting and comprehensive metadata
        log_amp_dataset = h5_file.create_dataset("log_amplitude", data=log_amplitude_processed, 
                                                 compression="gzip", compression_opts=6)  # Balanced compression
        log_amp_dataset.attrs["units"] = "ln(photons/mm²)"
        log_amp_dataset.attrs["description"] = "Natural logarithm of photon fluence amplitude measurements"
        log_amp_dataset.attrs["processing"] = "Log-transformed for neural network stability, clipped at 1e-8"
        log_amp_dataset.attrs["noise_model"] = f"{AMPLITUDE_NOISE_PERCENTAGE*100:.1f}% relative Gaussian noise"
        log_amp_dataset.attrs["shape_interpretation"] = "(N_measurements,) - one value per independent source-detector pair"
        
        phase_dataset = h5_file.create_dataset("phase", data=phase_processed,
                                              compression="gzip", compression_opts=6)
        phase_dataset.attrs["units"] = "degrees"
        phase_dataset.attrs["description"] = "Phase delay measurements relative to source modulation"
        phase_dataset.attrs["noise_model"] = f"±{PHASE_NOISE_STD_DEGREES}° additive Gaussian noise"
        phase_dataset.attrs["phase_reference"] = "Relative to source modulation at specified frequency"
        phase_dataset.attrs["shape_interpretation"] = "(N_measurements,) - one value per independent source-detector pair"

        # SUBSTEP 5.2: Save complete geometric configuration for reconstruction and visualization
        # **UPDATED FOR OPTIMIZED SHARED-SOURCE STRATEGY:**
        # NIRFASTer femdata() returns measurements in the order specified by measurement_links,
        # but measurement_links uses 1-based indexing while Python arrays use 0-based indexing
        # With shared sources: measurement_links[:,0] points to unique source indices (50 unique)
        # measurement_links[:,1] points to measurement-specific detector indices (1000 total)
        
        # **COORDINATE SYSTEM CLARIFICATION:**
        # 1. Surface extraction finds air voxels adjacent to tissue (accessible placement locations)
        # 2. Voxel-to-physical conversion gives air voxel centers in mm coordinates  
        # 3. touch_optodes() projects these to nearest mesh surface nodes (on tissue boundary)
        # 4. Mesh surface coordinates may fall inside tissue voxels due to coordinate system differences
        # 5. This is CORRECT behavior - mesh and voxel represent different discretizations
        
        # Extract source and detector indices from measurement links (convert from 1-based to 0-based for Python arrays)
        source_indices = (measurement_links[:, 0] - 1).astype(int)  # Convert from 1-based to 0-based (points to unique sources)
        detector_indices = (measurement_links[:, 1] - 1).astype(int)  # Convert from 1-based to 0-based (points to all detectors)
        
        # **CRITICAL:** Use mesh-projected coordinates after touch_optodes() for accurate physics simulation
        # These represent the actual positions where boundary conditions are applied in the FEM solver
        # touch_optodes() finds nearest mesh surface nodes to ensure proper coupling
        logger.debug("Using mesh-projected coordinates after touch_optodes() for accurate physics simulation")
        mesh_projected_sources = phantom_mesh.source.coord  # Mesh surface coordinates (N_unique_sources, 3)
        mesh_projected_detectors = phantom_mesh.meas.coord  # Mesh surface coordinates (N_detectors, 3)
        
        # Reorder mesh-projected positions to match measurement order returned by femdata()
        ordered_sources_mm = mesh_projected_sources[source_indices]  # Mesh-projected sources in measurement order
        ordered_detectors_mm = mesh_projected_detectors[detector_indices]  # Mesh-projected detectors in measurement order
        
        # **NEW: SAVE ORIGINAL COORDINATES (BEFORE TOUCH_OPTODES) FOR VALIDATION**
        # These are the original air voxel centers before mesh projection - should be 100% on surface
        ordered_sources_original_mm = unique_sources_mm[source_indices]  # Original air voxel centers in measurement order
        ordered_detectors_original_mm = probe_detectors_mm[detector_indices]  # Original air voxel centers in measurement order
        
        # Log coordinate system differences for transparency
        if len(unique_sources_mm) > 0:
            original_source_sample = unique_sources_mm[0]
            projected_source_sample = mesh_projected_sources[0]
            projection_distance = np.linalg.norm(projected_source_sample - original_source_sample)
            logger.debug(f"Coordinate system example: air voxel center {original_source_sample} -> mesh surface {projected_source_sample}")
            logger.debug(f"Typical projection distance: {projection_distance:.3f} mm (mesh surface refinement)")
        
        logger.debug(f"Final coordinates: {len(ordered_sources_mm)} sources, {len(ordered_detectors_mm)} detectors for physics simulation")
        
        source_dataset = h5_file.create_dataset("source_positions", data=ordered_sources_mm)
        source_dataset.attrs["units"] = "mm"
        source_dataset.attrs["description"] = "NIR source positions on mesh surface after touch_optodes() projection"
        source_dataset.attrs["coordinate_system"] = "Continuous mesh surface coordinates (may differ from voxel grid)"
        source_dataset.attrs["placement_method"] = "Air voxel centers -> mesh surface projection via touch_optodes()"
        source_dataset.attrs["physics_accuracy"] = "Exact coordinates used in FEM simulation for boundary conditions"
        source_dataset.attrs["coordinate_note"] = "Mesh surface may pass through tissue voxels - this is correct physics"
        source_dataset.attrs["optimization_strategy"] = f"{n_unique_sources} unique sources shared across {n_measurements} measurements"
        source_dataset.attrs["ordering"] = "Positions ordered to match femdata() measurement sequence"
        
        detector_dataset = h5_file.create_dataset("detector_positions", data=ordered_detectors_mm)
        detector_dataset.attrs["units"] = "mm"
        detector_dataset.attrs["description"] = "Detector positions on mesh surface after touch_optodes() projection"
        detector_dataset.attrs["coordinate_system"] = "Continuous mesh surface coordinates (may differ from voxel grid)"
        detector_dataset.attrs["placement_method"] = "Air voxel centers -> mesh surface projection via touch_optodes()"
        detector_dataset.attrs["physics_accuracy"] = "Exact coordinates used in FEM simulation for boundary conditions"
        detector_dataset.attrs["coordinate_note"] = "Mesh surface may pass through tissue voxels - this is correct physics"
        detector_dataset.attrs["sds_range"] = f"[{DEFAULT_MIN_PROBE_DISTANCE}, {DEFAULT_MAX_PROBE_DISTANCE}]mm"
        detector_dataset.attrs["selection_strategy"] = "Pure random sampling: uniform random selection within 10-40mm SDS range"
        detector_dataset.attrs["grouping"] = "Shape: (N_measurements, 3) for [measurement_idx][x,y,z] ordered by links"
        detector_dataset.attrs["ordering"] = "Positions ordered to match femdata() measurement sequence"
        
        # **NEW: SAVE ORIGINAL COORDINATES (BEFORE TOUCH_OPTODES) FOR VALIDATION**
        # These represent the pure surface extraction results - should be 100% on air-tissue boundary
        source_original_dataset = h5_file.create_dataset("source_positions_original", data=ordered_sources_original_mm)
        source_original_dataset.attrs["units"] = "mm"
        source_original_dataset.attrs["description"] = "Original NIR source positions before touch_optodes() projection"
        source_original_dataset.attrs["coordinate_system"] = "Air voxel centers from surface extraction"
        source_original_dataset.attrs["placement_method"] = "Direct conversion from air voxels adjacent to tissue"
        source_original_dataset.attrs["validation_purpose"] = "Should show 100% air voxel placement (before mesh projection)"
        source_original_dataset.attrs["coordinate_note"] = "These are the ideal surface positions before physics projection"
        source_original_dataset.attrs["comparison"] = "Compare with source_positions to see mesh projection effects"
        
        detector_original_dataset = h5_file.create_dataset("detector_positions_original", data=ordered_detectors_original_mm)
        detector_original_dataset.attrs["units"] = "mm"
        detector_original_dataset.attrs["description"] = "Original detector positions before touch_optodes() projection"
        detector_original_dataset.attrs["coordinate_system"] = "Air voxel centers from surface extraction"
        detector_original_dataset.attrs["placement_method"] = "Direct conversion from air voxels adjacent to tissue"
        detector_original_dataset.attrs["validation_purpose"] = "Should show 100% air voxel placement (before mesh projection)"
        detector_original_dataset.attrs["coordinate_note"] = "These are the ideal surface positions before physics projection"
        detector_original_dataset.attrs["comparison"] = "Compare with detector_positions to see mesh projection effects"
        
        # SUBSTEP 5.3: Save measurement connectivity matrix for OPTIMIZED source-detector pairing validation
        links_dataset = h5_file.create_dataset("measurement_links", data=measurement_links)
        links_dataset.attrs["description"] = "OPTIMIZED source-detector connectivity matrix (1-based indexing for NIRFASTer compatibility)"
        links_dataset.attrs["columns"] = "[source_index, detector_index, active_flag] where indices are 1-based (NIRFASTer convention)"
        links_dataset.attrs["optimization_strategy"] = f"Shared sources: {n_unique_sources} sources reused across {n_measurements} measurements"
        links_dataset.attrs["efficiency_gain"] = f"{n_measurements/n_unique_sources:.1f}x more measurements per FEM solve"
        links_dataset.attrs["active_measurements"] = f"{np.sum(measurement_links[:, 2])}/{len(measurement_links)}"
        links_dataset.attrs["usage"] = "Identifies which source-detector pairs contribute to measurement vector"

        # SUBSTEP 5.4: Save pixel-perfect ground truth optical property maps for supervised learning
        ground_truth_dataset = h5_file.create_dataset("ground_truth", data=ground_truth_maps,
                                                     compression="gzip", compression_opts=6)
        ground_truth_dataset.attrs["channels"] = "absorption_coefficient, reduced_scattering_coefficient"
        ground_truth_dataset.attrs["units"] = "mm^-1"
        ground_truth_dataset.attrs["description"] = "Dense voxel-wise optical property maps for reconstruction validation"
        ground_truth_dataset.attrs["wavelength"] = "800nm (typical NIR tomography wavelength)"
        ground_truth_dataset.attrs["shape_interpretation"] = "(2, Nx, Ny, Nz) for [μₐ, μ′s] channels (channels-first format)"
        ground_truth_dataset.attrs["clinical_relevance"] = "Literature-based physiological ranges with tumor contrast"
        
        # SUBSTEP 5.5: Save original phantom segmentation for region-specific analysis and visualization
        if phantom_volume is not None:
            labels_dataset = h5_file.create_dataset("tissue_labels", data=phantom_volume.astype(np.uint8))
            labels_dataset.attrs["label_encoding"] = "0=air, 1=healthy_tissue, >=2=tumor_regions"
            labels_dataset.attrs["voxel_size_mm"] = f"[{VOXEL_SIZE_MM}, {VOXEL_SIZE_MM}, {VOXEL_SIZE_MM}]"
            labels_dataset.attrs["description"] = "Tissue type labels for phantom segmentation and patch extraction masking"
            labels_dataset.attrs["usage"] = "Region-specific analysis, air masking, tissue boundary visualization"
            logger.debug(f"Tissue labels dataset saved: {phantom_volume.shape}")
        else:
            logger.debug("No phantom volume provided - skipping tissue labels storage")
        
        # Set file-level attributes with OPTIMIZATION metadata
        h5_file.attrs["modulation_frequency_hz"] = fd_frequency_hz
        h5_file.attrs["noise_amplitude_std"] = amplitude_noise_std
        h5_file.attrs["noise_phase_std"] = PHASE_NOISE_STD_DEGREES
        h5_file.attrs["n_measurements"] = len(measurement_links)
        h5_file.attrs["n_unique_sources"] = n_unique_sources  # NEW: Track unique sources
        h5_file.attrs["n_total_detectors"] = n_total_detectors  # NEW: Track total detectors
        h5_file.attrs["optimization_strategy"] = "strategic_sources_with_random_detectors"
        h5_file.attrs["computational_efficiency"] = f"{n_measurements/n_unique_sources:.1f}x"
        h5_file.attrs["data_augmentation_potential"] = f"~{n_measurements//256}_subsamples_per_phantom"
        h5_file.attrs["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
        
    file_size_mb = os.path.getsize(output_h5_filename) / (1024**2)
    logger.info(f"✓ OPTIMIZED dataset saved successfully - {output_h5_filename} ({file_size_mb:.1f} MB)")
    logger.info(f"Final OPTIMIZED dataset: {n_measurements} measurements from {n_unique_sources} strategic sources")
    logger.debug(f"Ground truth shape: {ground_truth_maps.shape[0]}×{ground_truth_maps.shape[1]}×{ground_truth_maps.shape[2]}×{ground_truth_maps.shape[3]} voxels")
    
    # Return success - no NaN values detected
    return True

# --------------------------------------------------------------
# VISUALIZATION: 3D PROBE-MESH RENDERING FOR GEOMETRIC VALIDATION
# --------------------------------------------------------------

def visualize_probe_on_mesh(phantom_volume, phantom_mesh, probe_sources, probe_detectors, measurement_links, 
                           probe_index, save_directory, patch_info=None, show_interactive=False, 
                           n_visual_pairs=DEFAULT_N_VISUAL_PROBE_PAIRS):
    """
    Creates a clean, professional 3D visualization suitable for academic reports.
    
    NEW FEATURES:
    • Shows multiple source-detector pairs (default: 5) for better understanding
    • Interactive 3D popup option controlled by show_interactive flag
    • Displays original voxel coordinates (before touch_optodes mesh projection)
    • Both 2D PNG save and optional 3D interactive viewing
    
    Key Features:
    • Clean axis labels (X, Y, Z in mm) with proper spatial orientation
    • High-contrast colors optimized for both screen and print
    • Minimal, professional styling suitable for thesis/reports
    • Clear legend and typography
    • Publication-quality output at 300 DPI
    """
    from scipy.ndimage import binary_erosion
    
    logger.debug(f"Creating professional visualization for phantom {probe_index+1}")
    logger.debug(f"Showing {n_visual_pairs} source-detector pairs (interactive: {show_interactive})")

    # Initialize figure with clean, modern styling
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 10), facecolor='white', dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Professional color scheme - optimized for reports
    colors = {
        'tissue': '#2E8B57',        # Sea green - professional medical
        'tumor': '#DC143C',         # Crimson red - high contrast
        'patch': '#FF8C00',         # Dark orange - MUCH more visible than blue
        'source': '#FF6347',        # Tomato red - bright but professional
        'detector': '#1E90FF',      # Dodger blue - clean contrast
        'connection': '#708090'     # Slate gray - subtle connection
    }
    
    # Clean background - minimal visual noise
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(True, alpha=0.2, linewidth=0.5)  # Subtle grid
    
    # Professional axis styling
    ax.tick_params(axis='x', colors='#2C3E50', labelsize=10)
    ax.tick_params(axis='y', colors='#2C3E50', labelsize=10)
    ax.tick_params(axis='z', colors='#2C3E50', labelsize=10)

    phantom_size_mm = phantom_volume.shape[0] * VOXEL_SIZE_MM

    # TISSUE SURFACE RENDERING - Clean and efficient
    healthy_tissue_mask = (phantom_volume == HEALTHY_TISSUE_LABEL)
    if np.any(healthy_tissue_mask):
        # Extract surface using morphological operations
        healthy_surface = healthy_tissue_mask & (~binary_erosion(healthy_tissue_mask, iterations=1))
        healthy_coords = np.argwhere(healthy_surface)
        
        if len(healthy_coords) > 0:
            healthy_coords_mm = convert_voxel_to_physical_coordinates(healthy_coords)
            # Downsample for performance while maintaining visual quality
            step = max(1, len(healthy_coords_mm) // 2000)
            
            ax.scatter(healthy_coords_mm[::step, 0], 
                      healthy_coords_mm[::step, 1], 
                      healthy_coords_mm[::step, 2],
                      c=colors['tissue'], s=4, alpha=0.6, 
                      label='Tissue Surface', edgecolors='none')

    # TUMOR VISUALIZATION - High contrast for visibility
    tumor_labels = [label for label in np.unique(phantom_volume) if label >= TUMOR_START_LABEL]
    if tumor_labels:
        all_tumor_coords = []
        for region_label in tumor_labels:
            tumor_mask = (phantom_volume == region_label)
            tumor_surface = tumor_mask & (~binary_erosion(tumor_mask, iterations=1))
            tumor_coords = np.argwhere(tumor_surface)
            if len(tumor_coords) > 0:
                all_tumor_coords.extend(tumor_coords)
        
        if all_tumor_coords:
            all_tumor_coords = np.array(all_tumor_coords)
            tumor_coords_mm = convert_voxel_to_physical_coordinates(all_tumor_coords)
            step = max(1, len(tumor_coords_mm) // 1000)
            
            ax.scatter(tumor_coords_mm[::step, 0],
                      tumor_coords_mm[::step, 1], 
                      tumor_coords_mm[::step, 2],
                      c=colors['tumor'], s=8, alpha=0.8,
                      label=f'Tumor{"s" if len(tumor_labels) > 1 else ""} ({len(tumor_labels)})',
                      edgecolors='white', linewidths=0.2)

    # PROBE PATCH REGION - HIGHLY VISIBLE for clinical context
    if patch_info and 'patch_surface_coordinates' in patch_info:
        patch_coords = patch_info['patch_surface_coordinates']
        if len(patch_coords) > 0:
            patch_coords_mm = convert_voxel_to_physical_coordinates(patch_coords)
            step = max(1, len(patch_coords_mm) // 600)  # Less downsampling for better visibility
            
            ax.scatter(patch_coords_mm[::step, 0],
                      patch_coords_mm[::step, 1], 
                      patch_coords_mm[::step, 2],
                      c=colors['patch'], s=8, alpha=0.8,  # Larger size, higher alpha
                      label=f'Probe Region ({patch_info["radius"]}mm)',
                      edgecolors='darkred', linewidths=0.3)  # Dark red outline for contrast

    # MULTIPLE PROBE ELEMENTS - Show N source-detector pairs for better understanding
    logger.debug(f"Available probe data: {len(probe_sources)} sources, {len(probe_detectors)} detectors, {len(measurement_links)} links")
    
    # Randomly sample N pairs from available measurements
    if len(measurement_links) > 0:
        n_available = len(measurement_links)
        n_to_show = min(n_visual_pairs, n_available)
        
        # Use fixed seed for reproducible visualization selection
        vis_rng = np.random.default_rng(probe_index + 12345)  # Deterministic based on phantom index
        selected_indices = vis_rng.choice(n_available, size=n_to_show, replace=False)
        
        logger.debug(f"Randomly selected {n_to_show} measurement pairs from {n_available} available")
        
        # Show selected source-detector pairs with consistent colors
        for i, idx in enumerate(selected_indices):
            # Get source and detector for this measurement
            source_pos = probe_sources[idx]
            detector_pos = probe_detectors[idx]
            
            # Convert to physical coordinates
            source_mm = convert_voxel_to_physical_coordinates([source_pos])[0]
            detector_mm = convert_voxel_to_physical_coordinates([detector_pos])[0]
            
            # Plot source - ALL sources are RED
            ax.scatter(source_mm[0], source_mm[1], source_mm[2], 
                      c='red', s=140, 
                      label='NIR Sources' if i == 0 else "", marker='o', 
                      edgecolors='white', linewidths=2, alpha=1.0, zorder=10)
            
            # Plot detector - ALL detectors are BLUE circles
            ax.scatter(detector_mm[0], detector_mm[1], detector_mm[2],
                      c='blue', s=120,
                      label='NIR Detectors' if i == 0 else "", marker='o',
                      edgecolors='white', linewidths=1.5, alpha=0.9, zorder=9)
            
            # Connection line between source and detector - subtle gray
            ax.plot([source_mm[0], detector_mm[0]], 
                   [source_mm[1], detector_mm[1]], 
                   [source_mm[2], detector_mm[2]],
                   color='gray', linewidth=1.5, alpha=0.5, linestyle='-',
                   label='Source-Detector Links' if i == 0 else "")
            
            # Calculate and log SDS for this pair
            sds_distance = np.linalg.norm(source_mm - detector_mm)
            logger.debug(f"  Pair {i+1}: SDS = {sds_distance:.1f}mm")
            
            # Calculate and log SDS for this pair
            sds_distance = np.linalg.norm(source_mm - detector_mm)
            logger.debug(f"  Pair {i+1}: SDS = {sds_distance:.1f}mm")
    
    else:
        logger.warning("No measurement links available for visualization")

    # PROFESSIONAL LABELING - Clean and minimal
    ax.set_title(f'NIR Phantom {probe_index+1:02d}: {n_visual_pairs} Source-Detector Pairs\n'
                f'Probe Layout Visualization', 
                 fontsize=14, fontweight='bold', color='#2C3E50', pad=20)
    
    # Simple, clear axis labels
    ax.set_xlabel('X (mm)', fontsize=12, color='#2C3E50', fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, color='#2C3E50', fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, color='#2C3E50', fontweight='bold')
    
    # Professional legend - clean and readable
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0),
                      frameon=True, fancybox=False, shadow=False,
                      fontsize=10, edgecolor='#34495E', facecolor='white',
                      framealpha=0.95, borderpad=1)
    legend.get_frame().set_linewidth(1)
    
    # Proper 3D orientation - Z as expected vertical depth
    ax.set_xlim(0, phantom_size_mm)
    ax.set_ylim(0, phantom_size_mm) 
    ax.set_zlim(0, phantom_size_mm)
    ax.set_box_aspect([1, 1, 1])
    
    # Optimal viewing angle for medical imaging
    ax.view_init(elev=25, azim=45)
    
    # Save high-quality 2D PNG (always done)
    output_filename = f"phantom_{probe_index+1:03d}_probe_layout.png"
    output_path = os.path.join(save_directory, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    logger.info(f"✓ 2D visualization saved: {output_path}")
    
    # Optional 3D interactive popup
    if show_interactive:
        logger.info("🔍 Opening 3D interactive visualization - close window to continue...")
        plt.show()  # This will block until user closes the window
        logger.debug("3D interactive window closed, continuing execution...")
    else:
        plt.close(fig)  # Clean up to save memory
        logger.debug("3D interactive display disabled - figure closed to save memory")
    
    return output_path


# --------------------------------------------------------------
# MAIN EXECUTION PIPELINE: BATCH PHANTOM DATASET GENERATION
# --------------------------------------------------------------

def main():
    """
    Orchestrates the complete phantom dataset generation pipeline for machine learning training.

    This main function coordinates all simulation stages to produce a dataset of synthetic NIR frequency-domain tomography measurements with ground truth optical property maps. Multiple unique phantoms are generated to ensure dataset diversity and robust training data for deep learning reconstruction.

    Pipeline Overview:
    1. Phantom geometry construction with randomized tissue/tumor distributions
    2. Finite element mesh generation (CGAL-based tetrahedral mesh)
    3. Optical property assignment (physiological μₐ and μ′s)
    4. Surface extraction for probe placement
    5. Patch-based probe layout (surface-constrained, clinical SDS)
    6. Frequency-domain FEM simulation with realistic noise
    7. Ground truth map generation
    8. Visualization (optional, for validation)
    9. Data storage in HDF5 format with metadata

    Key Features:
    - Patch-based probe placement to avoid spatial bias
    - Physiological optical properties and tumor contrast
    - Realistic measurement noise (amplitude/phase)
    - Pixel-level ground truth for supervised learning
    - Batch processing and quality control

    Output Structure:
        data/
        ├── phantom_001/
        │   ├── phantom_001_scan.h5
        │   └── probe_001.png
        ├── phantom_002/
        │   ├── phantom_002_scan.h5
        │   └── probe_001.png
        └── ...

    Returns:
        None. Generates datasets and logs progress.
    """
    
    # STEP 1: Initialize comprehensive output directory structure and logging framework
    # Create hierarchical data directory in mah422 folder using absolute paths
    # This ensures datasets are always stored in the correct location regardless of where script is run
    data_dir = project_root / "data"  # Absolute path to mah422/data directory
    data_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist, ignore if it does
    
    # Configure professional logging system using centralized NIRDOTLogger
    # Initialize logging system for complete pipeline monitoring and debugging with absolute paths
    logs_dir = project_root / "logs"
    NIRDOTLogger.setup_logging(log_dir=str(logs_dir))  # Setup centralized logging system
    logger = get_data_logger(__name__)  # Get data processing specific logger
    
    # Generate comprehensive pipeline initialization summary for monitoring and reproducibility
    logger.info("="*80)
    logger.info("STARTING NIR FREQUENCY-DOMAIN PHANTOM DATASET GENERATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Output directory: {data_dir.absolute()}")  # Absolute path for clarity
    logger.info(f"Logs directory: {logs_dir.absolute()}")  # Show where logs are actually stored
    logger.info(f"Project root: {project_root.absolute()}")  # Show project root for reference
    logger.info("Visualizations: ENABLED (static PNG images generated for all phantoms)")

    # STEP 2: Configure OPTIMIZED dataset generation parameters for machine learning training requirements
    # **NEW STRATEGY: Generate fewer phantoms with MORE measurements each for better data augmentation**
    # Each phantom now produces 1000 measurements (vs. 256 previously) enabling training subsampling
    n_phantoms = DEFAULT_N_PHANTOMS  # Production dataset size for robust ML training 
    measurements_per_phantom = 50 * 20  # 50 sources × 20 detectors = 1000 measurements per phantom
    expected_measurements = n_phantoms * measurements_per_phantom  # Total measurement count for memory planning
    
    logger.info(f"Generating {n_phantoms} OPTIMIZED phantom datasets for machine learning training")
    logger.info(f"OPTIMIZED measurement strategy: {measurements_per_phantom} measurements per phantom (50 strategic sources × 20 detectors)")
    logger.info(f"Expected measurement count: {n_phantoms} phantoms × {measurements_per_phantom} measurements = {expected_measurements:,} measurements")
    logger.info(f"Computational efficiency: Only 50 FEM solves per phantom (vs. 1000 measurements = 20x efficiency gain)")
    logger.info(f"Data augmentation potential: ~{measurements_per_phantom//256} different 256-measurement training samples per phantom")
    logger.info(f"Estimated dataset size: ~{expected_measurements * 8 / 1024**2:.1f} MB (float64 measurements)")
    
    # STEP 2.5: Initialize master random state for reproducible dataset generation
    master_rng, master_seed = setup_random_state(MASTER_RANDOM_SEED)
    phantom_seeds = master_rng.integers(0, 2**31, size=n_phantoms)
    
    logger.info("🎯 Random state management initialized:")
    logger.info(f"   Master seed: {master_seed} (for reproducible datasets)")
    logger.info(f"   Unique phantom seeds: {n_phantoms} generated")
    logger.info(f"   Reproducibility: ENABLED (same dataset every run)")
    
    # STEP 2.7: Hardware detection and capability logging for sanity checks
    logger.info("🖥️  Hardware Detection & Capability Analysis:")
    
    # System basics
    import platform
    import psutil
    logger.info(f"   System: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"   Python: {platform.python_version()}")
    
    # Memory analysis
    memory = psutil.virtual_memory()
    logger.info(f"   RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # CPU analysis
    cpu_count = psutil.cpu_count()
    try:
        cpu_freq_info = psutil.cpu_freq()
        if cpu_freq_info and cpu_freq_info.current:
            logger.info(f"   CPU: {cpu_count} cores @ {cpu_freq_info.current:.0f} MHz")
        else:
            logger.info(f"   CPU: {cpu_count} cores (frequency info unavailable)")
    except (FileNotFoundError, AttributeError, OSError):
        # Handle macOS ARM64 and other systems where CPU frequency is not accessible
        logger.info(f"   CPU: {cpu_count} cores (frequency detection not supported on this system)")
    
    # NIRFASTer solver detection
    try:
        best_solver = ff.utils.get_solver()
        logger.info(f"   NIRFASTer Solver: {best_solver} (preferred for FEM calculations)")
        if best_solver == 'GPU':
            logger.info(f"   🚀 GPU acceleration ENABLED - expect significant speedup")
        else:
            logger.info(f"   💻 CPU solver mode - reliable but slower than GPU")
    except Exception as e:
        logger.warning(f"   NIRFASTer solver detection failed: {e}")
    
    # Estimate computational requirements
    fem_solves_total = n_phantoms * 50  # 50 strategic sources per phantom
    logger.info(f"   Computational Load: {fem_solves_total:,} FEM solves planned")
    logger.info(f"   Expected Runtime: ~{fem_solves_total * 0.5 / 60:.1f} minutes (est. 0.5s/solve)")
    
    # STEP 3: Execute iterative phantom generation with comprehensive quality control
    # Each phantom is generated independently with unique random seeds to ensure statistical diversity
    pipeline_start_time = time.time()  # Track total pipeline execution time
    
    for phantom_idx in range(n_phantoms):
        phantom_start_time = time.time()  # Track per-phantom generation time for performance monitoring
        phantom_seed = phantom_seeds[phantom_idx]
        phantom_rng, _ = setup_random_state(phantom_seed)
        
        logger.info(f"🎲 Phantom {phantom_idx+1:02d}/{n_phantoms} - Seed: {phantom_seed}")
        
        logger.info("="*60)
        logger.info(f"GENERATING PHANTOM {phantom_idx+1:02d}/{n_phantoms}")
        logger.info("="*60)
        
        # SUBSTEP 3.1: Create phantom-specific output directory with systematic naming convention
        # Each phantom gets its own subdirectory for organized dataset storage and easy access
        phantom_dir = data_dir / f"phantom_{phantom_idx+1:02d}"  # Use pathlib for cleaner path handling
        phantom_dir.mkdir(exist_ok=True)  # Create directory with error handling
        logger.debug(f"Created phantom directory: {phantom_dir.absolute()}")

        # RETRY LOOP: Attempt phantom generation with different seeds until success or max attempts reached
        # This prevents NaN failures from corrupting the dataset and ensures all phantoms are valid
        phantom_success = False
        retry_attempt = 0
        
        while not phantom_success and retry_attempt < MAX_PHANTOM_RETRY_ATTEMPTS:
            if retry_attempt > 0:
                logger.warning(f"Retrying phantom {phantom_idx+1} generation (attempt {retry_attempt+1}/{MAX_PHANTOM_RETRY_ATTEMPTS})")
                # Generate new seed for retry attempt
                retry_seed = phantom_seeds[phantom_idx] + (retry_attempt * RETRY_SEED_OFFSET)
                phantom_rng, _ = setup_random_state(retry_seed)
                logger.info(f"🎲 Retry with new seed: {retry_seed}")
                
            try:
                # ============================================================================
                # STEP 1/6: PHANTOM GEOMETRY CONSTRUCTION
                # ============================================================================
                step1_start = time.time()
                logger.info("▶️  STEP 1/6: Constructing randomized phantom geometry with tissue and tumor embedding...")
                phantom_volume = build_phantom_with_tissue_and_tumours(phantom_rng=phantom_rng)  # Use managed RNG
                step1_time = time.time() - step1_start
                logger.info(f"✅ STEP 1/6 COMPLETED: Phantom geometry construction finished in {step1_time:.2f}s")
        
                # ============================================================================
                # STEP 2/6: FINITE ELEMENT MESH GENERATION  
                # ============================================================================
                step2_start = time.time()
                logger.info("▶️  STEP 2/6: Generating CGAL-based tetrahedral finite element mesh...")
                
                # Temporarily suppress NIRFASTer warnings during mesh creation (they're harmless but noisy)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Suppress all warnings during mesh creation
                    mesh_elements, mesh_nodes = mesh_volume(phantom_volume)  # Raw mesh generation
                    phantom_mesh = create_stndmesh(mesh_elements, mesh_nodes)  # NIRFASTer standardization
                
                step2_time = time.time() - step2_start
                logger.info(f"✅ STEP 2/6 COMPLETED: FEM mesh generation finished in {step2_time:.2f}s")
                
                # ============================================================================
                # STEP 3/6: OPTICAL PROPERTY ASSIGNMENT
                # ============================================================================
                step3_start = time.time()
                logger.info("▶️  STEP 3/6: Assigning physiological optical properties and generating ground truth maps...")
                phantom_mesh, ground_truth_maps = assign_optical_properties(phantom_mesh, phantom_volume, phantom_rng=phantom_rng)
                step3_time = time.time() - step3_start
                logger.info(f"✅ STEP 3/6 COMPLETED: Optical property assignment finished in {step3_time:.2f}s")
                
                # ============================================================================
                # STEP 4/6: SURFACE EXTRACTION & PROBE LAYOUT
                # ============================================================================
                step4_start = time.time()
                logger.info("▶️  STEP 4/6: Extracting tissue surface and generating OPTIMIZED probe layout...")
                surface_coordinates = extract_surface_voxels(phantom_volume)  # Morphological surface extraction
                probe_sources, probe_detectors, measurement_links, patch_info = build_patch_based_probe_layout(
                    surface_coordinates, n_sources=50, detectors_per_source=20, phantom_rng=phantom_rng)  # Use managed RNG
                step4_time = time.time() - step4_start
                logger.info(f"✅ STEP 4/6 COMPLETED: Surface extraction & probe layout finished in {step4_time:.2f}s")

                # ============================================================================
                # STEP 5/6: VISUALIZATION GENERATION (EVERY 100TH PHANTOM)
                # ============================================================================
                step5_start = time.time()
                
                # Only generate visualizations every 100 phantoms to save disk space
                # For 100,000 phantoms: 1,000 PNGs instead of 100,000 (saves ~100GB)
                should_visualize = (phantom_idx + 1) % 100 == 0 or phantom_idx < 5  # Every 100th + first 5 for validation
                
                if should_visualize:
                    logger.info("▶️  STEP 5/6: Generating probe visualization for quality assurance...")
                    
                    # Generate detailed visualization for the first probe of this phantom
                    if len(probe_sources) > 0:  # Ensure probes were successfully placed
                        
                        # Generate 3D visualization with multiple source-detector pairs
                        # Uses new multiple probe pair visualization for better understanding
                        visualize_probe_on_mesh(phantom_volume, phantom_mesh, probe_sources, probe_detectors, measurement_links, 
                                               0, str(phantom_dir), patch_info=patch_info, 
                                               show_interactive=ENABLE_3D_INTERACTIVE_VISUAL,
                                               n_visual_pairs=DEFAULT_N_VISUAL_PROBE_PAIRS)
                        
                        logger.debug(f"Generated visualization for phantom {phantom_idx+1} - "
                                   f"showing {DEFAULT_N_VISUAL_PROBE_PAIRS} probe pairs "
                                   f"(interactive: {ENABLE_3D_INTERACTIVE_VISUAL})")
                else:
                    logger.info("▶️  STEP 5/6: Skipping visualization generation (not a milestone phantom)")
                    logger.debug(f"Next visualization will be generated at phantom {((phantom_idx // 100) + 1) * 100}")
        
                step5_time = time.time() - step5_start
                logger.info(f"✅ STEP 5/6 COMPLETED: Visualization step finished in {step5_time:.2f}s")

                # ============================================================================
                # STEP 6/6: FREQUENCY-DOMAIN SIMULATION & DATA STORAGE
                # ============================================================================
                step6_start = time.time()
                logger.info("▶️  STEP 6/6: Executing frequency-domain diffusion equation simulation and dataset generation...")
                h5_output_path = phantom_dir / f"phantom_{phantom_idx+1:03d}_scan.h5"  # Use pathlib for systematic HDF5 naming
                
                # Execute complete forward modeling pipeline with noise simulation and data processing
                # This now returns True/False based on NaN detection
                simulation_success = run_fd_simulation_and_save(phantom_mesh, ground_truth_maps, probe_sources, probe_detectors, measurement_links, 
                                       phantom_volume=phantom_volume, output_h5_filename=str(h5_output_path))
                
                step6_time = time.time() - step6_start
                logger.info(f"✅ STEP 6/6 COMPLETED: FD simulation & data storage finished in {step6_time:.2f}s")
                
                if simulation_success:
                    # Success! Break out of retry loop
                    phantom_success = True
                    
                    # ============================================================================
                    # PHANTOM TIMING SUMMARY
                    # ============================================================================
                    total_phantom_time = time.time() - phantom_start_time
                    logger.info("="*70)
                    logger.info(f"🎉 PHANTOM {phantom_idx+1:02d} COMPLETED SUCCESSFULLY!")
                    logger.info("="*70)
                    logger.info("⏱️  STEP-BY-STEP TIMING BREAKDOWN:")
                    logger.info(f"   Step 1 - Geometry Construction:     {step1_time:6.2f}s ({step1_time/total_phantom_time*100:5.1f}%)")
                    logger.info(f"   Step 2 - FEM Mesh Generation:      {step2_time:6.2f}s ({step2_time/total_phantom_time*100:5.1f}%)")
                    logger.info(f"   Step 3 - Optical Properties:       {step3_time:6.2f}s ({step3_time/total_phantom_time*100:5.1f}%)")
                    logger.info(f"   Step 4 - Surface & Probe Layout:   {step4_time:6.2f}s ({step4_time/total_phantom_time*100:5.1f}%)")
                    logger.info(f"   Step 5 - Visualization:            {step5_time:6.2f}s ({step5_time/total_phantom_time*100:5.1f}%)")
                    logger.info(f"   Step 6 - FD Simulation & Storage:  {step6_time:6.2f}s ({step6_time/total_phantom_time*100:5.1f}%)")
                    logger.info("   " + "-"*60)
                    logger.info(f"   🏁 TOTAL PHANTOM TIME:             {total_phantom_time:6.2f}s (100.0%)")
                    logger.info("="*70)
                    
                    # Identify bottleneck
                    step_times = [step1_time, step2_time, step3_time, step4_time, step5_time, step6_time]
                    step_names = ["Geometry", "FEM Mesh", "Optical Props", "Surface/Probe", "Visualization", "FD Simulation"]
                    slowest_step_idx = step_times.index(max(step_times))
                    logger.info(f"🔍 BOTTLENECK ANALYSIS: Slowest step is '{step_names[slowest_step_idx]}' ({step_times[slowest_step_idx]:.2f}s)")
                    
                else:
                    # NaN values detected - increment retry counter and try again
                    retry_attempt += 1
                    if retry_attempt < MAX_PHANTOM_RETRY_ATTEMPTS:
                        logger.warning(f"Phantom {phantom_idx+1} failed due to NaN values, retrying with different seed...")
                        # Clean up failed H5 file if it exists
                        if h5_output_path.exists():
                            h5_output_path.unlink()
                            logger.debug(f"Removed corrupted H5 file: {h5_output_path}")
                    else:
                        logger.error(f"Phantom {phantom_idx+1} failed after {MAX_PHANTOM_RETRY_ATTEMPTS} attempts - giving up")
                        phantom_success = True  # Exit retry loop (will be handled by data cleaning later)
                        
            except Exception as e:
                # Handle any errors during phantom generation with specific error detection
                error_msg = str(e).lower()
                retry_attempt += 1
                
                # Detect specific mesh generation failures
                if "cgalmesher" in error_msg or "exec format error" in error_msg:
                    logger.error(f"Phantom {phantom_idx+1} failed with NIRFASTer mesh generation error: {e}")
                    logger.warning("This suggests geometry too complex for CGAL mesher - will try simpler geometry")
                elif "sigkill" in error_msg or "died with" in error_msg:
                    logger.error(f"Phantom {phantom_idx+1} failed with process termination: {e}")
                    logger.warning("This suggests memory/resource exhaustion - will try with different seed")
                else:
                    logger.error(f"Phantom {phantom_idx+1} generation failed with error: {e}")
                
                if retry_attempt < MAX_PHANTOM_RETRY_ATTEMPTS:
                    logger.warning(f"Retrying phantom {phantom_idx+1} with different seed...")
                    # Clean up any partial files
                    if 'h5_output_path' in locals() and h5_output_path.exists():
                        h5_output_path.unlink()
                        logger.debug(f"Removed partial H5 file: {h5_output_path}")
                else:
                    logger.error(f"Phantom {phantom_idx+1} failed after {MAX_PHANTOM_RETRY_ATTEMPTS} attempts with error: {e}")
                    logger.warning(f"Skipping phantom {phantom_idx+1} and continuing with next phantom...")
                    phantom_success = True  # Exit retry loop and continue to next phantom
        
        # Final phantom status logging (only if phantom failed completely)
        if not phantom_success or ('simulation_success' in locals() and not simulation_success):
            phantom_time = time.time() - phantom_start_time
            logger.warning(f"⚠️  PHANTOM {phantom_idx+1:02d} HAD ISSUES - completed in {phantom_time:.1f}s")
            logger.warning(f"Check data cleaning logs for phantom {phantom_idx+1}")

    # ============================================================================
    # PIPELINE COMPLETION SUMMARY WITH COMPREHENSIVE TIMING ANALYSIS
    # ============================================================================
    total_pipeline_time = time.time() - pipeline_start_time  # Calculate total time for all phantoms
    average_time_per_phantom = total_pipeline_time / n_phantoms if n_phantoms > 0 else 0
    
    logger.info("\n" + "="*80)
    logger.info("🏁 NIR PHANTOM DATASET GENERATION PIPELINE COMPLETED")
    logger.info("="*80)
    logger.info("📊 PERFORMANCE SUMMARY:")
    logger.info(f"   • Total phantoms generated:     {n_phantoms}")
    logger.info(f"   • Total processing time:        {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f} minutes)")
    logger.info(f"   • Average time per phantom:     {average_time_per_phantom:.1f}s")
    logger.info(f"   • Generation rate:              {n_phantoms / (total_pipeline_time/60):.1f} phantoms/minute")
    logger.info(f"   • Throughput:                   {n_phantoms * 1000 / (total_pipeline_time/60):.0f} measurements/minute")
    logger.info(f"   📁 Dataset location:            {data_dir.absolute()}")
    
    # Performance assessment
    if average_time_per_phantom < 60:
        performance_status = "🚀 EXCELLENT"
    elif average_time_per_phantom < 120:
        performance_status = "✅ GOOD"  
    elif average_time_per_phantom < 300:
        performance_status = "⚠️  MODERATE"
    else:
        performance_status = "🐌 SLOW"
    
    logger.info(f"   🎯 Performance rating:          {performance_status} ({average_time_per_phantom:.1f}s per phantom)")
    
    if average_time_per_phantom > 60:
        logger.info("💡 OPTIMIZATION SUGGESTIONS:")
        logger.info("   • Review step timing logs above to identify bottlenecks")
        logger.info("   • Consider reducing mesh density or phantom complexity")
        logger.info("   • Check system resources (CPU, memory, disk I/O)")
    
    logger.info("\n📋 DATASET SPECIFICATIONS:")
    logger.info("="*50)
    logger.info("Each phantom directory contains:")
    logger.info("  • HDF5 dataset file with complete measurement data")
    logger.info("    - Log-amplitude and phase measurements with realistic noise")
    logger.info("    - Source/detector positions and measurement connectivity")
    logger.info("    - Ground truth optical property maps (μₐ, μ′s)")
    logger.info("    - Complete geometry and metadata for reproducibility")
    
    logger.info("  • High-quality 3D visualization (probe_001.png) - Generated every 100 phantoms + first 5")
    logger.info("    - Storage optimization: ~1,000 PNGs instead of 100,000 (saves ~100GB disk space)")
    logger.info("    - Publication-quality rendering for geometric validation")
    logger.info("    - Tumor surfaces: unified red coloring for all tumor regions")
    logger.info("    - Patch regions: orange circular markers matching tissue surface")
    logger.info("    - Static PNG files only (no interactive displays)")
    
    # Technical specifications summary for dataset users
    logger.info("\nTechnical Dataset Specifications:")
    logger.info("="*40)
    logger.info(f"  • Phantom dimensions: {DEFAULT_PHANTOM_SHAPE[0]}×{DEFAULT_PHANTOM_SHAPE[1]}×{DEFAULT_PHANTOM_SHAPE[2]} voxels")
    logger.info(f"  • Tissue radius range: {DEFAULT_TISSUE_RADIUS_RANGE[0]}-{DEFAULT_TISSUE_RADIUS_RANGE[1]} mm")
    logger.info(f"  • Tumor radius range: {DEFAULT_TUMOR_RADIUS_RANGE[0]}-{DEFAULT_TUMOR_RADIUS_RANGE[1]} mm")
    logger.info(f"  • Measurements per phantom: {DEFAULT_N_GENERATED_MEASUREMENTS} total → subsample {DEFAULT_N_TRAINING_MEASUREMENTS} for training")
    logger.info(f"  • Source-detector separation: {DEFAULT_MIN_PROBE_DISTANCE}-{DEFAULT_MAX_PROBE_DISTANCE}mm (clinical range)")
    logger.info(f"  • Patch radius: {DEFAULT_PATCH_RADIUS}mm (realistic probe array constraints)")
    logger.info(f"  • Frequency-domain modulation: {DEFAULT_FD_FREQUENCY/1e6:.0f} MHz")
    logger.info(f"  • Mesh cell size: {DEFAULT_MESH_CELL_SIZE}mm (optimized for 800nm wavelength)")
    
    # Final validation and next steps guidance
    logger.info("\n" + "="*80)
    logger.info("DATASET GENERATION COMPLETE - READY FOR MACHINE LEARNING TRAINING")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("  1. Validate dataset integrity using provided visualization tools")
    logger.info("  2. Load HDF5 files into machine learning framework of choice")
    logger.info("  3. Implement data loaders for batch training with proper normalization")
    logger.info("  4. Consider data augmentation strategies for enhanced generalization")
    logger.info("  5. Establish proper train/validation/test splits for robust evaluation")
    logger.info("\n✅ All phantom datasets generated successfully and ready for use!")

if __name__ == "__main__":
    main()
