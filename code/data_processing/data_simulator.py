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
• 2mm voxel size for clinically realistic phantom dimensions (128×128×128mm)

Author: Max Hart - NIR Tomography Research
Version: 2.1 - Clinical Scale Phantom Generation (2mm voxels)
"""

# System path configuration for NIRFASTer-FF library access
import sys
import os
from pathlib import Path

# Get the project root directory (mah422) - works regardless of where script is run from
project_root = Path(__file__).parent.parent.parent  # Go up to mah422 directory

# Add NIRFASTer-FF to path dynamically
nirfaster_path = project_root / "nirfaster-FF"
if nirfaster_path.exists():
    sys.path.append(str(nirfaster_path))
else:
    # Fallback for development environment
    sys.path.append("/Users/maxhart/Documents/MSc_AI_ML/Dissertation/mah422/nirfaster-FF")

# Core libraries for numerical computing and visualization
import numpy as np                               # Primary array operations and linear algebra
import nirfasterff as ff                        # type: ignore # NIR light transport modeling via finite elements
import matplotlib.pyplot as plt                 # 2D/3D visualization and plotting
from mpl_toolkits.mplot3d import Axes3D         # 3D scatter plot capabilities

# Data storage and system operations
import h5py                                      # HDF5 hierarchical data format for large datasets
import os                                        # Operating system interface for directory management
import logging                                   # Professional logging system for pipeline monitoring
import time                                      # Time utilities for performance monitoring

# Specialized scipy modules for morphological and spatial operations
from scipy.ndimage import binary_erosion        # Morphological operation for surface extraction
from scipy.spatial.distance import cdist        # Efficient pairwise distance computation

# Import centralized logging configuration
sys.path.append(str(project_root / "code"))  # Add code directory to path using absolute path
from utils.logging_config import get_data_logger, NIRDOTLogger

# Constants for phantom generation
DEFAULT_PHANTOM_SHAPE = (64, 64, 64)        # Default cubic phantom dimensions in voxels (power of 2)
DEFAULT_TISSUE_RADIUS_RANGE = (26, 30)      # Healthy tissue ellipsoid semi-axis range (52-60mm with 2mm voxels)
DEFAULT_TUMOR_RADIUS_RANGE = (5, 10)        # Tumor ellipsoid semi-axis range (10-20mm with 2mm voxels)
DEFAULT_MAX_TUMORS = 5                       # Maximum number of tumors per phantom
DEFAULT_N_MEASUREMENTS = 256                # Number of independent source-detector pairs per phantom
DEFAULT_MIN_PROBE_DISTANCE = 10              # Minimum source-detector separation [mm] for diffusive regime validity
DEFAULT_MAX_PROBE_DISTANCE = 40              # Maximum source-detector separation [mm] for clinical realism
DEFAULT_PATCH_RADIUS = 40                    # Patch radius [mm] for surface probe placement (clinical probe array size)
DEFAULT_MIN_PATCH_VOXELS = 500               # Minimum surface voxels for valid patch placement
DEFAULT_FD_FREQUENCY = 140e6                 # Frequency-domain modulation frequency [Hz]
DEFAULT_MESH_CELL_SIZE = 1.65                # CGAL mesh characteristic cell size [mm] (maintained for high accuracy)
VOXEL_SIZE_MM = 2.0                          # Voxel size in millimeters for spatial calibration (doubled for clinical realism)

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

# Measurement noise parameters (conservative clean values for high SNR)
AMPLITUDE_NOISE_PERCENTAGE = 0.001           # 0.1% relative amplitude noise (ultra-clean, SNR ~60dB) 
PHASE_NOISE_STD_DEGREES = 0.1               # ±0.1° phase noise (precision research systems)

# Tumor placement algorithm parameters
MAX_TUMOR_PLACEMENT_ATTEMPTS = 50            # Maximum iterations for tumor placement rejection sampling
TUMOR_TISSUE_EMBEDDING_THRESHOLD = 0.80      # Required fraction of tumor volume inside tissue (80%)

# Mesh validation parameters
MIN_ELEMENT_VOLUME_MM3 = 0.8                 # Minimum acceptable tetrahedral element volume [mm³] (scaled for 2mm voxels)
MAX_ELEMENT_VOLUME_MM3 = 80.0                # Maximum acceptable tetrahedral element volume [mm³] (scaled for 2mm voxels)

# Surface processing and batch parameters  
SURFACE_BATCH_SIZE = 1000                    # Batch size for safe center identification
DEFAULT_RANDOM_SEED = 42                     # Default random seed for reproducibility

# Initialize logger for the module using centralized logging system
logger = get_data_logger(__name__)


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
                                          rng_seed=None):
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
        rng_seed (int): Random seed for reproducible generation
        
    Returns:
        numpy.ndarray: 3D volume with integer labels, shape (Nx, Ny, Nz)
    """
    # Initialize pseudorandom number generator with specified seed for reproducibility
    rng = np.random.default_rng(rng_seed)
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
    rx = rng.integers(*tissue_radius_range)  # Semi-axis length in x-direction [voxels]
    ry = rng.integers(*tissue_radius_range)  # Semi-axis length in y-direction [voxels] 
    rz = rng.integers(*tissue_radius_range)  # Semi-axis length in z-direction [voxels]
    
    # Generate random 3D rotation matrix for arbitrary ellipsoid orientation
    # This eliminates axis-aligned directional bias and simulates realistic anatomical variation
    # Critical for ML training: prevents model from learning phantom coordinate system artifacts
    rotation_matrix = random_rotation_matrix(rng)
    
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
    n_tumours = rng.integers(0, max_tumours+1)
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
            cx_t = rng.integers(cx-rx+3, cx+rx-3)  # X-coordinate with boundary safety margin
            cy_t = rng.integers(cy-ry+3, cy+ry-3)  # Y-coordinate with boundary safety margin
            cz_t = rng.integers(cz-rz+3, cz+rz-3)  # Z-coordinate with boundary safety margin
            
            # Sample tumour ellipsoid dimensions with physiological and geometric constraints
            # Tumor sizes follow clinical pathology distributions while respecting container geometry
            rx_t = rng.integers(*tumour_radius_range)  # Tumor semi-axis in x-direction [voxels]
            ry_t = rng.integers(*tumour_radius_range)  # Tumor semi-axis in y-direction [voxels]
            
            # Constrain z-axis dimension to maintain realistic aspect ratios and prevent elongated artifacts
            # Clinical tumors typically maintain roughly isotropic growth patterns in 3D
            # Ensure minimum radius ≥ 3 voxels for numerical stability and maximum ≤ max(rx_t, ry_t) for proportionality
            max_rz = min(tumour_radius_range[1], max(rx_t, ry_t))  # Enforce aspect ratio constraints
            rz_t = rng.integers(3, max(4, max_rz + 1))  # Ensure valid range with minimum size guarantee

            logger.debug(f"Tumor {tumour_idx+1} attempt {attempts+1}: center=({cx_t},{cy_t},{cz_t}), radii=({rx_t},{ry_t},{rz_t})")

            # Generate independent random rotation matrix for arbitrary tumor orientation
            # This ensures tumors also exhibit random poses, not just the tissue ellipsoid
            # Critical for training data diversity: prevents correlation between tumor and tissue orientations
            tumor_rotation = random_rotation_matrix(rng)
            
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
    # Smaller values increase mesh resolution but exponentially increase solve time
    mesh_params.general_cell_size = DEFAULT_MESH_CELL_SIZE  # Empirically optimized for NIR wavelengths (λ ≈ 800nm)
    
    logger.info(f"Starting CGAL-based tetrahedral mesh generation (cell_size={mesh_params.general_cell_size})")
    logger.debug(f"Input volume shape: {volume.shape}, unique labels: {np.unique(volume)}")
    
    # Execute CGAL-based tetrahedral mesh generation
    # This calls external C++ CGAL library for robust geometric mesh generation
    logger.debug("Invoking CGAL mesh generator...")
    mesh_elements, mesh_nodes = ff.meshing.RunCGALMeshGenerator(volume, opt=mesh_params)
    
    # Perform comprehensive mesh quality validation
    # Checks for inverted elements, aspect ratios, and topological consistency
    logger.debug("Validating mesh quality and topology...")
    ff.meshing.CheckMesh3D(mesh_elements, mesh_nodes)
    
    # Calculate mesh statistics
    num_elements = mesh_elements.shape[0]
    num_nodes = mesh_nodes.shape[0]
    mesh_density = num_elements / np.prod(volume.shape)
    
    logger.info(f"✓ Mesh generation completed - {num_elements:,} tetrahedra, {num_nodes:,} nodes")
    logger.debug(f"Mesh density: {mesh_density:.3f} elements/voxel")
    logger.debug(f"Average nodes per element: {num_nodes/num_elements:.1f}")
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
    phantom_mesh.from_solid(mesh_elements, mesh_nodes)
    
    # Extract and display mesh quality statistics for validation
    mean_element_volume = phantom_mesh.element_area.mean()  # Note: 'element_area' actually stores volumes for 3D elements
    std_element_volume = phantom_mesh.element_area.std()
    logger.info(f"Mesh statistics: {mesh_elements.shape[0]} tetrahedra, {mesh_nodes.shape[0]} nodes")
    logger.info(f"Element volume statistics: {mean_element_volume:.3f} ± {std_element_volume:.3f} mm³")
    
    # Validate mesh quality metrics are within acceptable bounds
    if mean_element_volume < MIN_ELEMENT_VOLUME_MM3 or mean_element_volume > MAX_ELEMENT_VOLUME_MM3:
        logger.warning(f"Element volumes ({mean_element_volume:.3f} mm³) may be outside optimal range for FEM convergence")
    else:
        logger.debug("Mesh quality metrics are within acceptable bounds")
    
    return phantom_mesh


# ============================================================================
# STEP 3: OPTICAL PROPERTY ASSIGNMENT AND GROUND TRUTH GENERATION
# ============================================================================

def assign_optical_properties(phantom_mesh, phantom_volume, rng_seed=None):
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
        rng_seed (int): Random seed for reproducible property assignment
        
    Returns:
        tuple: (phantom_mesh, ground_truth_maps) where:
            - phantom_mesh: Updated mesh with optical properties
            - ground_truth_maps: Dense (Nx, Ny, Nz, 2) array with μₐ and μ′s maps
    """
    # Initialize controlled random number generator for reproducible property sampling
    rng = np.random.default_rng(rng_seed)
    
    # Extract unique region labels from mesh elements
    # phantom_mesh.region contains the tissue type label for each tetrahedral element
    unique_regions = np.unique(phantom_mesh.region)
    optical_properties = []  # Will store [region_id, μₐ, μ′s, n] for each tissue type

    logger.info(f"Starting optical property assignment for {len(unique_regions)} tissue regions")
    logger.debug(f"Found regions: {unique_regions} (seed={rng_seed})")

    # Sample baseline healthy tissue optical properties from physiological distributions
    # These serve as reference values for relative tumour property scaling
    healthy_mua = rng.uniform(*HEALTHY_MUA_RANGE)      # Absorption coeff. [mm⁻¹] - controls image contrast
    healthy_musp = rng.uniform(*HEALTHY_MUSP_RANGE)    # Reduced scattering [mm⁻¹] - controls penetration depth
    
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
            mua_multiplier = rng.uniform(*TUMOR_MUA_MULTIPLIER_RANGE)
            musp_multiplier = rng.uniform(*TUMOR_MUSP_MULTIPLIER_RANGE)
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
    Extract 3D coordinates of tissue surface voxels for probe placement.
    
    Identifies boundary interface between tissue and air using binary morphological 
    operations. Critical for realistic probe placement since NIR sources and 
    detectors must be positioned on accessible tissue surfaces.
    
    Technical Implementation:
    • Applies binary erosion to identify tissue interior voxels  
    • Computes surface as set difference: tissue_bulk ∖ tissue_interior
    • Uses 3×3×3 structuring element for 26-connected neighborhood
    • Returns explicit (x,y,z) coordinates for efficient spatial indexing
    
    Args:
        phantom_volume (numpy.ndarray): 3D labeled phantom volume with tissue regions
        tissue_threshold (int): Minimum label value considered as tissue (excludes air=0)
        
    Returns:
        numpy.ndarray: Surface voxel coordinates, shape (N_surface, 3)
                      Each row contains (x, y, z) indices of surface voxels
    """
    # Create binary mask identifying all tissue regions (healthy + tumours)
    # This combines all non-air labels into a single binary volume
    tissue_binary_mask = (phantom_volume >= tissue_threshold)
    
    logger.info(f"Starting surface extraction (threshold={tissue_threshold})")
    logger.debug(f"Input volume shape: {phantom_volume.shape}")
    
    initial_tissue_count = np.sum(tissue_binary_mask)
    logger.debug(f"Initial tissue voxels: {initial_tissue_count:,}")
    
    # Apply binary erosion with single iteration to identify tissue interior
    # Uses default 3×3×3 structuring element for 26-connected neighborhood
    # Interior voxels are those completely surrounded by other tissue voxels
    logger.debug("Applying morphological erosion to identify tissue interior...")
    eroded_tissue_mask = binary_erosion(tissue_binary_mask, iterations=1)
    
    # Compute morphological boundary as set difference: tissue ∖ interior
    # Surface voxels are tissue voxels that have at least one air neighbor
    surface_voxel_mask = tissue_binary_mask & (~eroded_tissue_mask)
    
    # Extract explicit (x,y,z) coordinates of surface voxels
    # numpy.argwhere returns N×3 array of indices where condition is True
    surface_voxel_coordinates = np.argwhere(surface_voxel_mask)
    
    surface_count = surface_voxel_coordinates.shape[0]
    logger.info(f"✓ Surface extraction completed - {surface_count:,} surface voxels identified")
    
    # Validate surface extraction results
    if surface_count == 0:
        logger.error("No surface voxels found - check tissue geometry and threshold")
    elif surface_count < 100:
        logger.warning(f"Very few surface voxels ({surface_count}) - may limit probe placement options")
    else:
        surface_ratio = surface_count / initial_tissue_count
        logger.debug(f"Surface-to-volume ratio: {surface_ratio:.1%}")
    
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
                           min_patch_voxels=DEFAULT_MIN_PATCH_VOXELS, rng_seed=None):
    """
    Identifies surface voxels that can support full-radius patches for robust probe placement.
    
    This function implements safe center placement by pre-filtering surface positions to ensure
    each potential patch center can accommodate the full 40mm radius patch without extending
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


def build_patch_based_probe_layout(surface_coordinates, n_measurements=DEFAULT_N_MEASUREMENTS,
                                 patch_radius=DEFAULT_PATCH_RADIUS,
                                 min_source_detector_distance=DEFAULT_MIN_PROBE_DISTANCE,
                                 max_source_detector_distance=DEFAULT_MAX_PROBE_DISTANCE,
                                 min_patch_voxels=DEFAULT_MIN_PATCH_VOXELS,
                                 rng_seed=None):
    """
    Generates realistic probe configurations using independent source-detector pairs for optimal ML training.
    
    This function implements pure random source-detector pairing strategy:
    1. Identifies safe patch centers that can support full 40mm radius patches
    2. Randomly selects one patch center per phantom for spatial diversity
    3. Creates localized surface patch within specified radius
    4. Places 256 independent source-detector pairs randomly within the patch region
    5. Each measurement is a unique source-detector pair within clinical SDS range (10-40mm)
    
    Key Advantages:
    - Eliminates measurement bias from rigid probe grouping (no 1-source + 3-detector constraints)
    - Maximizes spatial sampling diversity for better ML model generalization
    - Reduces measurement count to 256 for faster training while maintaining coverage
    - Ensures all probe positions lie exactly on tissue surface via binary erosion
    - Enforces clinical SDS constraints for physiological measurement realism
    
    Clinical Realism Features:
    - Patch size (80mm diameter) matches typical clinical probe array footprints for larger phantoms
    - SDS range (10-40mm) reflects real NIR measurement capabilities for 2mm voxel phantoms
    - Single patch placement simulates realistic partial tissue coverage scenarios
    - Surface-only placement prevents non-physical floating probe positions
    
    Args:
        surface_coordinates (numpy.ndarray): Surface voxel positions from binary erosion, shape (N_surface, 3)
        n_measurements (int): Number of independent source-detector pairs to place (256 for optimal training)
        patch_radius (float): Patch radius in mm defining local probe placement region (40mm clinical size)
        min_source_detector_distance (float): Minimum SDS in mm for diffusive regime validity (10mm)
        max_source_detector_distance (float): Maximum SDS in mm for clinical realism (40mm) 
        min_patch_voxels (int): Minimum surface voxels required for valid patch (500 for adequate sampling)
        rng_seed (int): Random seed for reproducible patch selection and probe placement
        
    Returns:
        tuple: (probe_sources, probe_detectors, measurement_links, patch_info) where:
            - probe_sources: Source positions for measurements, shape (n_measurements, 3)
            - probe_detectors: Detector positions for measurements, shape (n_measurements, 3) 
            - measurement_links: Source-detector connectivity, shape (n_measurements, 3)
            - patch_info: Dictionary containing patch metadata for visualization and analysis
    """
    # Initialize controlled randomization for reproducible patch-based placement
    rng = np.random.default_rng(rng_seed)
    
    logger.info(f"Starting patch-based probe layout generation")
    logger.info(f"Target: {n_measurements} independent source-detector pairs in {patch_radius}mm radius patch (SDS range: {min_source_detector_distance}-{max_source_detector_distance}mm)")
    logger.debug(f"Available surface positions: {len(surface_coordinates):,} voxels (seed={rng_seed})")
    
    # STEP 5.1: Identify safe patch centers that can support full radius patches
    logger.debug("Step 1/5: Identifying safe patch centers...")
    safe_patch_centers = find_safe_patch_centers(surface_coordinates, patch_radius, min_patch_voxels, rng_seed)
    
    if len(safe_patch_centers) == 0:
        logger.error("No safe patch centers available - cannot proceed with probe placement")
        return np.array([]), np.array([]), np.array([]), {}
    
    # STEP 5.2: Randomly select one patch center for this phantom
    logger.debug("Step 2/5: Selecting random patch center...")
    center_idx = rng.integers(0, len(safe_patch_centers))
    selected_patch_center = safe_patch_centers[center_idx]
    
    logger.info(f"Selected patch center: ({selected_patch_center[0]:.1f}, {selected_patch_center[1]:.1f}, {selected_patch_center[2]:.1f})")
    
    # STEP 5.3: Create patch by filtering surface voxels within radius
    logger.debug("Step 3/5: Creating surface patch...")
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
    
    # STEP 5.4: Place independent source-detector pairs randomly within the patch region
    logger.debug("Step 4/5: Placing independent source-detector pairs within patch...")
    all_probe_sources, all_probe_detectors, measurement_links = [], [], []
    placement_attempts = 0
    max_placement_attempts = n_measurements * 20  # Prevent infinite loops with generous limit
    
    for measurement_idx in range(n_measurements):
        if (measurement_idx + 1) % 50 == 0:  # Progress logging
            logger.debug(f"Measurement placement progress: {measurement_idx+1}/{n_measurements}")
            
        # Implement rejection sampling with failure detection
        pair_placed = False
        while not pair_placed and placement_attempts < max_placement_attempts:
            placement_attempts += 1
            
            # STEP 5.4.1: Randomly sample source position from patch surface voxels
            source_idx = rng.integers(0, len(patch_surface_coordinates))
            current_source_position = patch_surface_coordinates[source_idx]
            current_source_position_mm = patch_surface_coordinates_mm[source_idx]
            
            # STEP 5.4.2: Find valid detector candidates within SDS constraints
            # Use physical coordinates (mm) for accurate distance calculations
            source_to_patch_distances = cdist([current_source_position_mm], patch_surface_coordinates_mm)[0]
            
            # Apply clinical SDS range constraints within the patch
            distance_mask = (source_to_patch_distances >= min_source_detector_distance) & \
                           (source_to_patch_distances <= max_source_detector_distance)
            valid_detector_coordinates = patch_surface_coordinates[distance_mask]  # Voxel coordinates
            valid_distances = source_to_patch_distances[distance_mask]
            
            # STEP 5.4.3: Validate sufficient detector availability
            if len(valid_detector_coordinates) < 1:
                continue  # Retry with different source position within patch
            
            # STEP 5.4.4: Randomly select 1 detector with uniform sampling
            detector_idx = rng.integers(0, len(valid_detector_coordinates))
            selected_detector_position = valid_detector_coordinates[detector_idx]
            
            # STEP 5.4.5: Store successful source-detector pair
            all_probe_sources.append(current_source_position)
            all_probe_detectors.append(selected_detector_position)
            
            # STEP 5.4.6: Generate measurement connectivity link
            measurement_links.append([measurement_idx, measurement_idx, 1])  # One-to-one source-detector mapping
            pair_placed = True
    
    # STEP 5.5: Validate and report placement results
    n_placed = len(all_probe_sources)
    placement_efficiency = n_placed / placement_attempts * 100 if placement_attempts > 0 else 0
    
    logger.info(f"✓ Patch-based probe layout completed")
    logger.info(f"Successfully placed: {n_placed}/{n_measurements} independent source-detector pairs")
    logger.debug(f"Placement efficiency: {placement_efficiency:.1f}% ({placement_attempts} total attempts)")
    
    if n_placed < n_measurements:
        logger.warning(f"Could not place all requested measurements due to SDS constraints within patch")
        logger.info(f"Consider adjusting patch radius or SDS range for higher placement success rate")
    
    # STEP 5.6: Compile patch metadata for visualization and analysis
    patch_info = {
        'center_position': selected_patch_center,
        'radius': patch_radius,
        'surface_voxels_in_patch': patch_size,
        'safe_centers_available': len(safe_patch_centers),
        'patch_surface_coordinates': patch_surface_coordinates,  # For visualization
        'sds_range': (min_source_detector_distance, max_source_detector_distance)
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
        None. Writes HDF5 file and logs progress.
    """
    
    # STEP 1: Configure mesh with comprehensive optode integration and geometric validation
    # This critical step ensures proper coupling between probe positions and finite element mesh
    logger.info("Configuring mesh with source-detector optode layout and geometric validation")
    
    # Convert discrete voxel coordinates to continuous spatial coordinates for FEM compatibility
    # Ensures proper interpolation and boundary condition application in finite element solver
    probe_sources_mm = convert_voxel_to_physical_coordinates(probe_sources)
    probe_detectors_mm = convert_voxel_to_physical_coordinates(probe_detectors)
    
    phantom_mesh.source = ff.base.optode(probe_sources_mm.astype(float))   # NIRFASTer source container
    phantom_mesh.meas = ff.base.optode(probe_detectors_mm.astype(float))   # NIRFASTer detector container
    phantom_mesh.link = measurement_links  # Connectivity matrix defining active source-detector pairs
    
    logger.debug(f"Optode configuration: {len(probe_sources)} sources, {len(probe_detectors)} detectors")
    logger.debug(f"Active measurements: {np.sum(measurement_links[:, 2])} of {len(measurement_links)} total pairs")
    
    # Project optodes onto mesh surface with geometric consistency validation
    # Critical for accurate boundary condition application and prevents non-physical floating optodes
    # NIRFASTer automatically finds nearest mesh surface nodes for each optode position
    phantom_mesh.touch_optodes()
    logger.debug("Optodes successfully projected onto mesh surface with geometric validation")

    # STEP 2: Execute frequency-domain finite element forward simulation with comprehensive monitoring
    # This represents the core physics calculation solving the diffusion equation across the entire mesh
    logger.info(f"Executing frequency-domain diffusion equation simulation (modulation: {fd_frequency_hz/1e6:.1f} MHz)")
    
    # Log detailed mesh and measurement statistics for performance monitoring and debugging
    num_nodes = phantom_mesh.nodes.shape[0]
    num_elements = phantom_mesh.elements.shape[0]
    num_measurements = len(measurement_links)
    mesh_density = num_elements / np.prod(phantom_volume.shape) if phantom_volume is not None else 0
    
    logger.debug(f"FEM mesh statistics: {num_nodes:,} nodes, {num_elements:,} elements")
    logger.debug(f"Mesh density: {mesh_density:.3f} elements/voxel")
    logger.debug(f"Measurement matrix: {len(probe_sources)} sources → {num_measurements} total measurements")
    
    # Execute complex-valued frequency-domain finite element solution
    # Solves: -∇·(D∇Φ) + [μₐ + iω/c]Φ = S for complex photon fluence Φ(r,ω)
    # Returns measurement data structure containing amplitude and phase at each detector
    simulation_start_time = time.time()
    simulation_data, _ = phantom_mesh.femdata(fd_frequency_hz)  # Core NIRFASTer FEM solver
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
    # Clinical NIR systems typically achieve 40-60 dB SNR corresponding to 1-3% relative noise
    amplitude_noise_std = AMPLITUDE_NOISE_PERCENTAGE * np.mean(raw_amplitude)
    noisy_amplitude = raw_amplitude + noise_rng.normal(0, amplitude_noise_std, raw_amplitude.shape)
    
    # Phase noise: Additive Gaussian noise independent of signal magnitude
    # Models lock-in amplifier precision, timing jitter, and temperature drift
    # Commercial systems typically achieve ±1-3 degree phase precision
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
    if log_amp_range[0] < -50 or log_amp_range[1] > 50:
        logger.warning(f"Log-amplitude range may be outside optimal bounds for neural networks")
    if np.any(np.isnan(log_amplitude_processed)) or np.any(np.isnan(phase_processed)):
        logger.error("NaN values detected in processed measurements - check phantom geometry")

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
        source_dataset = h5_file.create_dataset("source_positions", data=probe_sources_mm)
        source_dataset.attrs["units"] = "mm"
        source_dataset.attrs["description"] = "NIR source positions in phantom coordinate system"
        source_dataset.attrs["coordinate_system"] = f"Physical coordinates in millimeters (voxel_size={VOXEL_SIZE_MM}mm)"
        source_dataset.attrs["placement_method"] = "Patch-based surface sampling with clinical constraints"
        
        detector_dataset = h5_file.create_dataset("detector_positions", data=probe_detectors_mm)
        detector_dataset.attrs["units"] = "mm"
        detector_dataset.attrs["description"] = "Detector positions for independent measurements (one per measurement)"
        detector_dataset.attrs["coordinate_system"] = f"Physical coordinates in millimeters (voxel_size={VOXEL_SIZE_MM}mm)"
        detector_dataset.attrs["sds_range"] = f"[{DEFAULT_MIN_PROBE_DISTANCE}, {DEFAULT_MAX_PROBE_DISTANCE}]mm"
        detector_dataset.attrs["grouping"] = "Shape: (N_measurements, 3) for [measurement_idx][x,y,z]"
        
        # SUBSTEP 5.3: Save measurement connectivity matrix for source-detector pairing validation
        links_dataset = h5_file.create_dataset("measurement_links", data=measurement_links)
        links_dataset.attrs["description"] = "Source-detector connectivity matrix defining measurement pairs"
        links_dataset.attrs["columns"] = "[source_index, detector_index, active_flag]"
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
        
        # Set file-level attributes
        h5_file.attrs["modulation_frequency_hz"] = fd_frequency_hz
        h5_file.attrs["noise_amplitude_std"] = amplitude_noise_std
        h5_file.attrs["noise_phase_std"] = PHASE_NOISE_STD_DEGREES
        h5_file.attrs["n_measurements"] = len(measurement_links)
        h5_file.attrs["n_probes"] = len(probe_sources)
        h5_file.attrs["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
        
    file_size_mb = os.path.getsize(output_h5_filename) / (1024**2)
    logger.info(f"✓ Dataset saved successfully - {output_h5_filename} ({file_size_mb:.1f} MB)")
    logger.info(f"Final dataset: {log_amplitude_processed.shape[0]} independent source-detector measurements")
    logger.debug(f"Ground truth shape: {ground_truth_maps.shape[0]}×{ground_truth_maps.shape[1]}×{ground_truth_maps.shape[2]}×{ground_truth_maps.shape[3]} voxels")

# --------------------------------------------------------------
# VISUALIZATION: 3D PROBE-MESH RENDERING FOR GEOMETRIC VALIDATION
# --------------------------------------------------------------

def visualize_probe_on_mesh(phantom_volume, phantom_mesh, source_position, detector_positions, probe_index, save_directory, patch_info=None, show_interactive=False):
    """
    Create clean 3D visualization showing key tissue regions and single source-detector pair.

    Simple, focused visualization highlighting:
    • Healthy tissue boundary
    • Tumor regions 
    • Patch region (if provided)
    • Single source-detector measurement pair
    • Phantom boundary box
    """
    from scipy.ndimage import binary_erosion  # Import for morphological operations
    logger.debug(f"Creating clean visualization for probe {probe_index+1}")

    # Initialize clean figure
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.grid(False)
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))

    # STEP 1: Show phantom boundary box
    phantom_size_mm = phantom_volume.shape[0] * VOXEL_SIZE_MM  # 128mm total size
    
    # Draw boundary box edges
    box_coords = [
        [0, 0, 0], [phantom_size_mm, 0, 0], [phantom_size_mm, phantom_size_mm, 0], [0, phantom_size_mm, 0], [0, 0, 0],  # bottom face
        [0, 0, phantom_size_mm], [phantom_size_mm, 0, phantom_size_mm], [phantom_size_mm, phantom_size_mm, phantom_size_mm], [0, phantom_size_mm, phantom_size_mm], [0, 0, phantom_size_mm]  # top face
    ]
    box_coords = np.array(box_coords)
    ax.plot(box_coords[:5, 0], box_coords[:5, 1], box_coords[:5, 2], 'white', linewidth=1, alpha=0.3)  # bottom edges
    ax.plot(box_coords[5:, 0], box_coords[5:, 1], box_coords[5:, 2], 'white', linewidth=1, alpha=0.3)  # top edges
    # Vertical edges
    for i in range(4):
        ax.plot([box_coords[i, 0], box_coords[i+5, 0]], 
                [box_coords[i, 1], box_coords[i+5, 1]], 
                [box_coords[i, 2], box_coords[i+5, 2]], 'white', linewidth=1, alpha=0.3)

    # STEP 2: Extract and show healthy tissue surface
    healthy_tissue_mask = (phantom_volume == HEALTHY_TISSUE_LABEL)
    if np.any(healthy_tissue_mask):
        # Use morphological edge detection to find tissue boundary
        healthy_surface = healthy_tissue_mask & (~binary_erosion(healthy_tissue_mask, iterations=1))
        healthy_surface_coords = np.argwhere(healthy_surface)
        
        if len(healthy_surface_coords) > 0:
            # Convert to physical coordinates
            healthy_surface_coords_mm = convert_voxel_to_physical_coordinates(healthy_surface_coords)
            # Downsample for performance - target ~1000 points
            downsample_factor = max(1, len(healthy_surface_coords_mm) // 1000)
            ax.scatter(healthy_surface_coords_mm[::downsample_factor, 0], 
                      healthy_surface_coords_mm[::downsample_factor, 1], 
                      healthy_surface_coords_mm[::downsample_factor, 2],
                      color='lime', s=4, alpha=0.6, label='Healthy tissue', marker='o')

    # STEP 3: Extract and show tumor regions
    tumor_count = 0
    for region_label in np.unique(phantom_volume):
        if region_label >= TUMOR_START_LABEL:
            tumor_mask = (phantom_volume == region_label)
            if np.any(tumor_mask):
                # Extract tumor surface
                tumor_surface = tumor_mask & (~binary_erosion(tumor_mask, iterations=1))
                tumor_surface_coords = np.argwhere(tumor_surface)
                
                if len(tumor_surface_coords) > 0:
                    tumor_count += 1
                    # Convert to physical coordinates
                    tumor_surface_coords_mm = convert_voxel_to_physical_coordinates(tumor_surface_coords)
                    # Downsample for performance
                    downsample_factor = max(1, len(tumor_surface_coords_mm) // 300)
                    ax.scatter(tumor_surface_coords_mm[::downsample_factor, 0],
                              tumor_surface_coords_mm[::downsample_factor, 1], 
                              tumor_surface_coords_mm[::downsample_factor, 2],
                              color='red', s=6, alpha=0.8, 
                              label=f'Tumor {tumor_count}' if tumor_count == 1 else '', marker='o')

    # STEP 4: Show patch region if provided
    if patch_info is not None and 'patch_surface_coordinates' in patch_info:
        patch_surface_coords = patch_info['patch_surface_coordinates']
        if len(patch_surface_coords) > 0:
            # Convert to physical coordinates
            patch_surface_coords_mm = convert_voxel_to_physical_coordinates(patch_surface_coords)
            # Downsample for performance
            patch_downsample_factor = max(1, len(patch_surface_coords_mm) // 600)
            ax.scatter(patch_surface_coords_mm[::patch_downsample_factor, 0],
                      patch_surface_coords_mm[::patch_downsample_factor, 1], 
                      patch_surface_coords_mm[::patch_downsample_factor, 2],
                      color='purple', s=5, alpha=0.7, 
                      label=f'Patch region (r={patch_info["radius"]}mm)', marker='o')

    # STEP 5: Show source and ONE detector (first detector)
    # Convert probe positions to physical coordinates
    source_position_mm = convert_voxel_to_physical_coordinates([source_position])[0]
    detector_positions_mm = convert_voxel_to_physical_coordinates(detector_positions)
    
    # Show source
    ax.scatter(source_position_mm[0], source_position_mm[1], source_position_mm[2], 
               c='yellow', s=120, edgecolor='black', linewidth=2, 
               label='NIR Source', marker='o')
    
    # Show only the first detector
    if len(detector_positions_mm) > 0:
        first_detector = detector_positions_mm[0]
        ax.scatter(first_detector[0], first_detector[1], first_detector[2], 
                   c='cyan', s=90, edgecolor='black', linewidth=1.5, 
                   label='NIR Detector', marker='o')
        
        # Draw measurement line between source and detector
        ax.plot([source_position_mm[0], first_detector[0]],
                [source_position_mm[1], first_detector[1]], 
                [source_position_mm[2], first_detector[2]], 
                'white', linewidth=2, alpha=0.8, linestyle='--', label='Measurement path')

    # STEP 6: Configure plot appearance
    ax.set_title(f"Probe {probe_index+1:03d} - Tissue Structure Overview", 
                 color='white', fontsize=16, fontweight='bold', pad=20)
                 
    ax.set_xlabel('X-axis (mm)', color='white', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y-axis (mm)', color='white', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z-axis (mm)', color='white', fontsize=12, fontweight='bold')

    ax.tick_params(colors='white', labelsize=10)
    ax.legend(facecolor='black', edgecolor='white', fontsize=10, labelcolor='white',
              loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.8)

    # Set axis limits and proper aspect ratio
    ax.set_xlim(0, phantom_size_mm)
    ax.set_ylim(0, phantom_size_mm)
    ax.set_zlim(0, phantom_size_mm)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Set optimal viewing angle
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()

    # Save the visualization
    output_image_path = os.path.join(save_directory, f"probe_{probe_index+1:03d}.png")
    plt.savefig(output_image_path, dpi=300, 
                facecolor=fig.get_facecolor(), 
                bbox_inches='tight', 
                edgecolor='none')
    logger.debug(f"Saved clean probe visualization: {output_image_path}")

    if show_interactive:
        plt.show()
    else:
        plt.close()


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

    # STEP 2: Configure dataset generation parameters for machine learning training requirements
    # Generate multiple phantoms to ensure statistical diversity and prevent overfitting in ML models
    n_phantoms = 2  # Production dataset size for robust ML training 
    expected_measurements = n_phantoms * DEFAULT_N_MEASUREMENTS  # Total measurement count for memory planning
    
    logger.info(f"Generating {n_phantoms} phantom datasets for machine learning training")
    logger.info(f"Expected measurement count: {n_phantoms} phantoms × {DEFAULT_N_MEASUREMENTS} measurements = {expected_measurements:,} measurements")
    logger.info(f"Estimated dataset size: ~{expected_measurements * 8 / 1024**2:.1f} MB (float64 measurements)")
    
    # STEP 3: Execute iterative phantom generation with comprehensive quality control
    # Each phantom is generated independently with unique random seeds to ensure statistical diversity
    pipeline_start_time = time.time()  # Track total pipeline execution time
    
    for phantom_idx in range(n_phantoms):
        phantom_start_time = time.time()  # Track per-phantom generation time for performance monitoring
        
        logger.info("="*60)
        logger.info(f"GENERATING PHANTOM {phantom_idx+1:02d}/{n_phantoms}")
        logger.info("="*60)
        
        # SUBSTEP 3.1: Create phantom-specific output directory with systematic naming convention
        # Each phantom gets its own subdirectory for organized dataset storage and easy access
        phantom_dir = data_dir / f"phantom_{phantom_idx+1:02d}"  # Use pathlib for cleaner path handling
        phantom_dir.mkdir(exist_ok=True)  # Create directory with error handling
        logger.debug(f"Created phantom directory: {phantom_dir.absolute()}")

        # SUBSTEP 3.2: Construct phantom geometry with controlled randomization and biological realism
        # Uses unique random seed per phantom to ensure statistical independence between phantoms
        # Prevents correlation artifacts that could bias machine learning model training
        logger.info("Step 1/6: Constructing randomized phantom geometry with tissue and tumor embedding")
        phantom_volume = build_phantom_with_tissue_and_tumours(rng_seed=44+phantom_idx)  # Offset seed for uniqueness
        
        # SUBSTEP 3.3: Generate high-quality finite element mesh for accurate numerical simulation
        # CGAL-based tetrahedral mesh generation ensures robust light transport modeling
        # Mesh quality directly impacts forward modeling accuracy and reconstruction performance
        logger.info("Step 2/6: Generating CGAL-based tetrahedral finite element mesh")
        
        # Temporarily suppress NIRFASTer warnings during mesh creation (they're harmless but noisy)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress all warnings during mesh creation
            mesh_elements, mesh_nodes = mesh_volume(phantom_volume)  # Raw mesh generation
            phantom_mesh = create_stndmesh(mesh_elements, mesh_nodes)  # NIRFASTer standardization
        
        # SUBSTEP 3.4: Assign physiologically realistic optical properties with controlled randomization
        # Uses literature-based optical coefficient ranges to ensure clinical relevance
        # Ground truth maps provide pixel-perfect reference for supervised learning validation
        logger.info("Step 3/6: Assigning physiological optical properties and generating ground truth maps")
        phantom_mesh, ground_truth_maps = assign_optical_properties(phantom_mesh, phantom_volume, rng_seed=42+phantom_idx)
        
        # SUBSTEP 3.5: Extract tissue surface and generate clinically realistic probe configurations
        # Patch-based placement simulates real clinical constraints and eliminates spatial bias
        # Surface extraction ensures probes are positioned only on accessible tissue boundaries
        logger.info("Step 4/6: Extracting tissue surface and generating patch-based probe layout")
        surface_coordinates = extract_surface_voxels(phantom_volume)  # Morphological surface extraction
        probe_sources, probe_detectors, measurement_links, patch_info = build_patch_based_probe_layout(
            surface_coordinates, n_measurements=DEFAULT_N_MEASUREMENTS, rng_seed=123+phantom_idx)  # Unique seed per phantom

        # SUBSTEP 3.6: Generate probe visualizations for quality assurance and validation
        # Visualization enables geometric validation and provides educational materials
        logger.info("Step 5/6: Generating probe visualization for quality assurance")
        vis_start_time = time.time()  # Track visualization generation time
        
        # Generate detailed visualization for the first probe of each phantom
        if len(probe_sources) > 0:  # Ensure probes were successfully placed
            first_source = probe_sources[0]  # Select first source for visualization
            first_detectors = probe_detectors[0:1]  # First detector associated with first measurement
            
            # Generate 3D visualization with surface boundaries and probe positioning
            # Note: show_interactive=False to avoid popup windows
            visualize_probe_on_mesh(phantom_volume, phantom_mesh, first_source, first_detectors, 0, str(phantom_dir), 
                                   patch_info=patch_info, show_interactive=False)
            
            logger.info(f"Generated static PNG visualization for phantom {phantom_idx+1}")
        
        vis_time = time.time() - vis_start_time  # Calculate visualization generation time
        logger.debug(f"Visualization generation completed in {vis_time:.1f}s")

        # SUBSTEP 3.7: Execute frequency-domain finite element simulation and save complete dataset
        # This is the core physics simulation that generates realistic NIR measurement data
        # Solves complex-valued diffusion equation and processes results for machine learning
        logger.info("Step 6/6: Executing frequency-domain diffusion equation simulation and dataset generation")
        h5_output_path = phantom_dir / f"phantom_{phantom_idx+1:03d}_scan.h5"  # Use pathlib for systematic HDF5 naming
        
        # Execute complete forward modeling pipeline with noise simulation and data processing
        run_fd_simulation_and_save(phantom_mesh, ground_truth_maps, probe_sources, probe_detectors, measurement_links, 
                                   phantom_volume=phantom_volume, output_h5_filename=str(h5_output_path))
        
        # Calculate and log per-phantom generation performance metrics
        phantom_time = time.time() - phantom_start_time
        logger.info(f"✓ PHANTOM {phantom_idx+1:02d} COMPLETED in {phantom_time:.1f}s")
        logger.debug(f"Complete dataset saved: {h5_output_path}")
        logger.debug(f"PNG visualization saved: {phantom_dir}/probe_001.png")

    # STEP 4: Generate comprehensive pipeline completion summary and validation report
    # Provides complete performance metrics and dataset validation for quality assurance
    total_pipeline_time = time.time() - pipeline_start_time  # Calculate total time for all phantoms
    average_time_per_phantom = total_pipeline_time / n_phantoms if n_phantoms > 0 else 0
    
    logger.info("="*80)
    logger.info("NIR PHANTOM DATASET GENERATION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Generated {n_phantoms} complete phantom datasets with full ground truth")
    logger.info(f"Total processing time: {total_pipeline_time:.1f}s ({average_time_per_phantom:.1f}s per phantom)")
    logger.info(f"Generation rate: {n_phantoms / (total_pipeline_time/60):.1f} phantoms/minute")
    logger.info(f"Dataset storage location: {data_dir.absolute()}")
    
    # Detailed dataset contents summary for user reference
    logger.info("\nDataset Architecture and Contents:")
    logger.info("="*50)
    logger.info("Each phantom directory contains:")
    logger.info("  • HDF5 dataset file with complete measurement data")
    logger.info("    - Log-amplitude and phase measurements with realistic noise")
    logger.info("    - Source/detector positions and measurement connectivity")
    logger.info("    - Ground truth optical property maps (μₐ, μ′s)")
    logger.info("    - Complete geometry and metadata for reproducibility")
    
    logger.info("  • High-quality 3D visualization (probe_001.png) - Generated for all phantoms")
    logger.info("    - Publication-quality rendering for geometric validation")
    logger.info("    - Tumor surfaces: unified red coloring for all tumor regions")
    logger.info("    - Patch regions: purple circular markers matching tissue surface")
    logger.info("    - Static PNG files only (no interactive displays)")
    
    # Technical specifications summary for dataset users
    logger.info("\nTechnical Dataset Specifications:")
    logger.info("="*40)
    logger.info(f"  • Phantom dimensions: {DEFAULT_PHANTOM_SHAPE[0]}×{DEFAULT_PHANTOM_SHAPE[1]}×{DEFAULT_PHANTOM_SHAPE[2]} voxels")
    logger.info(f"  • Tissue radius range: {DEFAULT_TISSUE_RADIUS_RANGE[0]}-{DEFAULT_TISSUE_RADIUS_RANGE[1]} mm")
    logger.info(f"  • Tumor radius range: {DEFAULT_TUMOR_RADIUS_RANGE[0]}-{DEFAULT_TUMOR_RADIUS_RANGE[1]} mm")
    logger.info(f"  • Measurements per phantom: {DEFAULT_N_MEASUREMENTS} independent source-detector pairs")
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
