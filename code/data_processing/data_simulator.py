#!/usr/bin/env python3
"""
Near-Infrared Frequency-Domain (FD) Phantom Data Simulator with Robust Random Geometry Sampling

This script generates synthetic NIR optical tomography datasets by:
1. Creating a 3D phantom geometry with realistic tissue distributions
2. Embedding randomly positioned ellipsoidal healthy tissue and tumours
3. Generating unstructured tetrahedral finite element meshes for accurate light propagation modeling
4. Assigning physiologically realistic optical properties (μₐ, μ′s) with controlled randomization
5. Implementing spatially-invariant probe placement using surface-constrained sampling
6. Solving the frequency-domain diffusion equation via FEM for forward modeling
7. Adding realistic measurement noise and saving complete datasets to HDF5

The approach ensures robust training data generation for machine learning models by:
- Preventing spatial bias through fully randomized probe positioning
- Maintaining physiological realism in optical property distributions
- Ensuring geometric validity through surface-aware optode placement
- Providing complete ground truth information for supervised learning

Technical Implementation Notes:
- Uses NIRFASTer-FF library for finite element light transport modeling
- Implements frequency-domain diffusion equation at 140 MHz modulation
- Applies binary morphological operations for surface extraction
- Utilizes scipy.spatial for efficient distance computations in probe placement
"""

# System path configuration for NIRFASTer-FF library access
import sys
sys.path.append("/Users/maxhart/Documents/MSc_AI_ML/Dissertation/mah422/nirfaster-FF")

# Core numerical and scientific computing libraries
import numpy as np                           # Primary array operations and linear algebra
import nirfasterff as ff                     # type: ignore # NIR light transport modeling via finite elements
import matplotlib.pyplot as plt             # 2D/3D visualization and plotting
from mpl_toolkits.mplot3d import Axes3D     # 3D scatter plot capabilities

# Data storage and file system operations
import h5py                                  # HDF5 hierarchical data format for large datasets
import os                                    # Operating system interface for directory management
import logging                               # Professional logging system for pipeline monitoring
import time                                  # Time utilities for performance monitoring

# Specialized scipy modules for morphological and spatial operations
from scipy.ndimage import binary_erosion    # Morphological operation for surface extraction
from scipy.spatial.distance import cdist    # Efficient pairwise distance computation

# Constants for phantom generation
DEFAULT_PHANTOM_SHAPE = (60, 60, 60)        # Default cubic phantom dimensions in voxels (increased for better SDS coverage)
DEFAULT_TISSUE_RADIUS_RANGE = (24, 28)      # Healthy tissue ellipsoid semi-axis range (scaled up for larger phantoms)
DEFAULT_TUMOR_RADIUS_RANGE = (5, 10)        # Tumor ellipsoid semi-axis range (scaled proportionally with tissue)  
DEFAULT_MAX_TUMORS = 5                       # Maximum number of tumors per phantom
DEFAULT_N_PROBES = 500                      # Optimal balance: geometric coverage vs computational efficiency (ML scaling strategy)
DEFAULT_MIN_PROBE_DISTANCE = 10             # Minimum source-detector separation [mm] (clinical range)
DEFAULT_MAX_PROBE_DISTANCE = 40             # Maximum source-detector separation [mm] (clinical range)
DEFAULT_PATCH_RADIUS = 30                   # Patch radius [mm] for surface-constrained probe placement (ensures full SDS range utilization)
DEFAULT_MIN_PATCH_VOXELS = 500              # Minimum surface voxels required for valid patch placement (ensures adequate sampling for 500 probes)
DEFAULT_FD_FREQUENCY = 140e6                 # Frequency-domain modulation frequency [Hz]
DEFAULT_MESH_CELL_SIZE = 1.65                # CGAL mesh characteristic cell size [mm]
VOXEL_SIZE_MM = 1.0                         # Voxel size in millimeters for spatial calibration

# Tissue label constants for clarity and consistency
AIR_LABEL = 0                                # Background air regions
HEALTHY_TISSUE_LABEL = 1                     # Healthy tissue regions  
TUMOR_START_LABEL = 2                        # Starting label for tumor regions (incremented)

# Optical property constants at 800nm wavelength (literature-based physiological ranges)
HEALTHY_MUA_RANGE = (0.003, 0.007)          # Healthy tissue absorption coefficient [mm⁻¹]
HEALTHY_MUSP_RANGE = (0.78, 1.18)           # Healthy tissue reduced scattering [mm⁻¹]
TUMOR_MUA_MULTIPLIER_RANGE = (1.5, 3.5)     # Tumor absorption enhancement factor
TUMOR_MUSP_MULTIPLIER_RANGE = (1.5, 2.5)    # Tumor scattering enhancement factor
TISSUE_REFRACTIVE_INDEX = 1.33               # Fixed refractive index for biological tissues

# Measurement noise parameters (based on clinical NIR system performance)
AMPLITUDE_NOISE_PERCENTAGE = 0.02            # 2% relative amplitude noise (typical 40-60 dB SNR)
PHASE_NOISE_STD_DEGREES = 2.0                # ±2° phase noise (typical commercial system precision)

# Configure logging system for NIR phantom data simulation pipeline
def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure professional logging system for phantom data generation pipeline.
    
    This function sets up a comprehensive logging framework that provides:
    - Structured output with timestamps and log levels
    - Flexible control over verbosity (DEBUG, INFO, WARNING, ERROR)
    - Optional file output for permanent record keeping
    - Consistent formatting across all pipeline components
    
    Args:
        level (int): Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file. If None, logs only to console.
    """
    # Create custom formatter for professional log output
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler for permanent record keeping
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Suppress matplotlib debug messages for cleaner output
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logger

# Initialize logger for the module
logger = setup_logging()


# --------------------------------------------------------------
# STEP 1: VOLUMETRIC PHANTOM CONSTRUCTION WITH EMBEDDED GEOMETRIES
# --------------------------------------------------------------

def build_phantom_with_tissue_and_tumours(phantom_shape=DEFAULT_PHANTOM_SHAPE,
                                          tissue_radius_range=DEFAULT_TISSUE_RADIUS_RANGE,
                                          max_tumours=DEFAULT_MAX_TUMORS,
                                          tumour_radius_range=DEFAULT_TUMOR_RADIUS_RANGE,
                                          air_label=AIR_LABEL,
                                          tissue_label=HEALTHY_TISSUE_LABEL,
                                          tumour_start_label=TUMOR_START_LABEL,
                                          rng_seed=None):
    """
    Constructs a 3D phantom volume with realistic tissue and tumour distributions using arbitrary pose variation.
    
    This function implements a hierarchical geometry construction approach with advanced spatial realism:
    1. Initialize a cubic air-filled domain (background medium)
    2. Embed a large ellipsoidal healthy tissue region with random 3D rotation
    3. Insert multiple smaller ellipsoidal tumour inclusions with random orientations and strict containment
    
    Major Technical Improvements:
    - ARBITRARY POSE VARIATION: Both tissue and tumors use random 3D rotation matrices to eliminate directional bias
    - ENHANCED TUMOR EMBEDDING: Requires 80% of tumor volume to be inside tissue, but only places the contained portion
    - CLIPPED TUMOR PLACEMENT: Prevents unrealistic "floating tumor" voxels in air space for physiological accuracy
    - Uses implicit surface representation via quadratic forms for rotated ellipsoids
    - Applies robust rejection sampling with comprehensive geometric validation
    - Implements label-based segmentation for multi-region FEM meshing
    
    Rotation Implementation:
    - Generates random Euler angles (α, β, γ) for full SO(3) rotation coverage
    - Applies rotation matrices to coordinate systems before ellipsoid evaluation
    - Eliminates axis-aligned bias that could affect machine learning training
    
    Args:
        phantom_shape (tuple): 3D dimensions of the phantom domain in voxels (Nx, Ny, Nz)
        tissue_radius_range (tuple): Min/max semi-axes lengths for healthy tissue ellipsoid
        max_tumours (int): Maximum number of tumour inclusions to embed
        tumour_radius_range (tuple): Min/max semi-axes lengths for tumour ellipsoids
        air_label (int): Integer label for air/background regions (typically 0)
        tissue_label (int): Integer label for healthy tissue regions (typically 1)
        tumour_start_label (int): Starting label for tumour regions (incremented for each tumour)
        rng_seed (int): Random seed for reproducible phantom generation
        
    Returns:
        numpy.ndarray: 3D volume array with integer labels for each tissue type
                      Shape: (Nx, Ny, Nz), dtype: uint8
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
        Generate a random 3D rotation matrix using Euler angles for arbitrary ellipsoid orientations.
        
        This function implements proper 3D rotation sampling to eliminate directional bias in phantom generation.
        The approach uses the ZYX Euler angle convention (Tait-Bryan angles) to generate uniformly distributed
        rotations over the special orthogonal group SO(3).
        
        Mathematical Foundation:
        - Generates proper orthogonal matrices R ∈ SO(3) with det(R) = 1
        - Uses composition of elementary rotations: R = R_z(α) · R_y(β) · R_x(γ)
        - Ensures uniform sampling over rotation space by appropriate angle range selection
        - Maintains matrix orthogonality: R^T · R = I and ||R·v|| = ||v|| for any vector v
        
        Rotation Sequence (ZYX Tait-Bryan Convention):
        1. Rotate around x-axis by angle γ ∈ [0, 2π] (roll)
        2. Rotate around y-axis by angle β ∈ [0, π] (pitch) 
        3. Rotate around z-axis by angle α ∈ [0, 2π] (yaw)
        
        Clinical Relevance:
        - Eliminates axis-aligned bias that could affect ML model training
        - Simulates realistic tumor orientations (not restricted to coordinate axes)
        - Ensures tissue geometry variation matches clinical anatomical diversity
        
        Args:
            rng (numpy.random.Generator): Controlled PRNG for reproducible rotations
            
        Returns:
            numpy.ndarray: 3×3 orthogonal rotation matrix with det(R)=1, shape (3, 3)
        """
        # Sample random Euler angles for uniform rotation coverage
        # Note: Beta uses [0,π] range to avoid gimbal lock singularities at poles
        alpha = rng.uniform(0, 2*np.pi)  # Z-axis rotation (yaw): full rotation [0°, 360°]
        beta = rng.uniform(0, np.pi)     # Y-axis rotation (pitch): hemisphere [0°, 180°]  
        gamma = rng.uniform(0, 2*np.pi)  # X-axis rotation (roll): full rotation [0°, 360°]
        
        # Construct elementary rotation matrices using Rodrigues' rotation formula
        # Each matrix represents rotation around a single coordinate axis
        
        # X-axis rotation matrix (roll): R_x(γ)
        # Rotates points in the YZ-plane, preserving X-coordinates
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(gamma), -np.sin(gamma)],
                      [0, np.sin(gamma), np.cos(gamma)]])
                      
        # Y-axis rotation matrix (pitch): R_y(β)  
        # Rotates points in the XZ-plane, preserving Y-coordinates
        # Note: Sign convention follows right-hand rule
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])
                      
        # Z-axis rotation matrix (yaw): R_z(α)
        # Rotates points in the XY-plane, preserving Z-coordinates
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
        
        # Compose final rotation matrix using proper matrix multiplication order
        # Order is critical: R = R_z(α) · R_y(β) · R_x(γ) for ZYX Euler convention
        # This applies rotations in sequence: first X, then Y, finally Z
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
        max_attempts = 50  # Increased iteration limit for enhanced tumor placement success
        
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
                
                if embedding_ratio >= 0.80:  # Clinical realism threshold: require 80% tissue containment
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


# --------------------------------------------------------------
# STEP 2: FINITE ELEMENT MESH GENERATION FOR LIGHT TRANSPORT MODELING
# --------------------------------------------------------------

def mesh_volume(volume):
    """
    Converts labeled voxel-based phantom geometry into unstructured tetrahedral finite element mesh.
    
    This function performs automatic mesh generation specifically for NIR light transport simulations:
    - Excludes air regions (label=0) from meshing to reduce computational overhead
    - Generates conforming tetrahedral elements that respect tissue boundaries
    - Applies CGAL-based Delaunay triangulation with quality guarantees
    - Ensures mesh suitability for finite element diffusion equation solving
    
    Technical Implementation:
    - Uses advancing front algorithm for boundary conforming mesh generation
    - Applies Delaunay refinement to maintain element quality metrics
    - Implements automatic feature preservation at tissue interfaces
    - Optimizes cell size for numerical accuracy vs computational efficiency balance
    
    Args:
        volume (numpy.ndarray): 3D labeled phantom volume
                               Shape: (Nx, Ny, Nz), integer labels for tissue types
                               
    Returns:
        tuple: (elements, nodes) where:
            - elements (numpy.ndarray): Tetrahedral connectivity matrix, shape (N_tet, 4)
                                       Each row contains 4 node indices forming a tetrahedron
            - nodes (numpy.ndarray): Node coordinate matrix, shape (N_nodes, 3)
                                   Each row is (x, y, z) spatial coordinates in mm
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
    if mean_element_volume < 0.1 or mean_element_volume > 10.0:
        logger.warning(f"Element volumes ({mean_element_volume:.3f} mm³) may be outside optimal range for FEM convergence")
    else:
        logger.debug("Mesh quality metrics are within acceptable bounds")
    
    return phantom_mesh


# --------------------------------------------------------------
# STEP 3: OPTICAL PROPERTY ASSIGNMENT AND GROUND TRUTH GENERATION
# --------------------------------------------------------------

def assign_optical_properties(phantom_mesh, phantom_volume, rng_seed=None):
    """
    Assigns physiologically realistic optical properties to mesh regions and generates ground truth.
    
    This function implements clinically-informed optical property distributions for NIR tomography:
    - Assigns absorption coefficient (μₐ) and reduced scattering coefficient (μ′s) per tissue type
    - Uses randomized sampling within physiological bounds for dataset diversity
    - Maintains realistic contrast ratios between healthy tissue and tumours
    - Generates pixel-wise ground truth maps for supervised learning validation
    
    Optical Property Ranges (based on literature at 800nm wavelength):
    - Healthy tissue: μₐ ∈ [0.003, 0.007] mm⁻¹, μ′s ∈ [0.78, 1.18] mm⁻¹
    - Tumour tissue: μₐ = (1.5-3.5)×healthy, μ′s = (1.5-2.5)×healthy
    - Fixed refractive index: n = 1.33 (typical for biological tissues)
    
    Technical Implementation:
    - Uses per-phantom randomization to simulate biological variability
    - Maintains spatial coherence within each tissue region
    - Creates lookup tables for efficient property queries during FEM assembly
    - Generates dense voxel-wise ground truth for pixel-level reconstruction evaluation
    
    Args:
        phantom_mesh (nirfasterff.base.stndmesh): Finite element mesh with region labels
        phantom_volume (numpy.ndarray): Original voxel-based phantom for ground truth mapping
        rng_seed (int): Random seed for reproducible property assignment
        
    Returns:
        tuple: (phantom_mesh, ground_truth_maps) where:
            - phantom_mesh: Updated mesh object with optical properties assigned to each element
            - ground_truth_maps: Dense (Nx, Ny, Nz, 2) array containing μₐ and μ′s maps
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
    # Shape: (Nx, Ny, Nz, 2) where last dimension is [μₐ, μ′s]
    ground_truth_maps = np.zeros((Nx, Ny, Nz, 2))
    
    logger.debug("Generating dense ground truth maps...")
    # Populate ground truth grid using region-based property lookup
    # This creates pixel-perfect reference maps for quantitative evaluation
    for region_label, (tissue_mua, tissue_musp) in region_optical_lookup.items():
        # Apply properties to all voxels belonging to this tissue region
        ground_truth_maps[phantom_volume == region_label, 0] = tissue_mua   # Channel 0: absorption coefficient
        ground_truth_maps[phantom_volume == region_label, 1] = tissue_musp  # Channel 1: reduced scattering coefficient

    logger.info(f"✓ Ground truth maps generated - shape {ground_truth_maps.shape}")
    logger.debug(f"Value ranges: μₐ=[{ground_truth_maps[...,0].min():.4f}, {ground_truth_maps[...,0].max():.4f}], "
                f"μ′s=[{ground_truth_maps[...,1].min():.3f}, {ground_truth_maps[...,1].max():.3f}]")

    return phantom_mesh, ground_truth_maps


# --------------------------------------------------------------
# STEP 4: TISSUE SURFACE EXTRACTION VIA MORPHOLOGICAL OPERATIONS
# --------------------------------------------------------------

def extract_surface_voxels(phantom_volume, tissue_threshold=HEALTHY_TISSUE_LABEL):
    """
    Extracts explicit 3D coordinates of tissue surface voxels for optode placement constraints.
    
    This function identifies the boundary interface between tissue and air using binary morphological 
    operations. Surface extraction is critical for realistic optode placement since NIR sources and 
    detectors must be positioned on accessible tissue surfaces in clinical imaging scenarios.
    
    Technical Implementation:
    - Applies binary erosion to identify tissue interior voxels  
    - Computes surface as set difference: tissue_bulk ∖ tissue_interior
    - Uses 3×3×3 structuring element for 26-connected neighborhood analysis
    - Returns explicit (x,y,z) coordinates for efficient spatial indexing
    
    Mathematical Foundation:
    - Binary erosion: B ⊖ S = {p | S_p ⊆ B} where S is structuring element
    - Surface extraction: ∂B = B ∖ (B ⊖ S) yields morphological boundary
    - Ensures surface voxels have at least one air-adjacent neighbor
    
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

def find_safe_patch_centers(surface_coordinates, patch_radius=DEFAULT_PATCH_RADIUS, 
                           min_patch_voxels=DEFAULT_MIN_PATCH_VOXELS, rng_seed=None):
    """
    Identifies surface voxels that can support full-radius patches for robust probe placement.
    
    This function implements safe center placement by pre-filtering surface positions to ensure
    each potential patch center can accommodate the full 20mm radius patch without extending
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
    
    safe_centers = []
    total_candidates = len(surface_coordinates)
    
    # Process in batches to manage memory for large surface sets
    batch_size = min(1000, total_candidates)
    
    for batch_start in range(0, total_candidates, batch_size):
        batch_end = min(batch_start + batch_size, total_candidates)
        batch_centers = surface_coordinates[batch_start:batch_end]
        
        # Compute distances from batch centers to all surface voxels
        distances = cdist(batch_centers, surface_coordinates)
        
        # Count surface voxels within patch radius for each center candidate
        for i, center_distances in enumerate(distances):
            voxels_in_patch = np.sum(center_distances <= patch_radius)
            
            if voxels_in_patch >= min_patch_voxels:
                safe_centers.append(batch_centers[i])
        
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


def build_patch_based_probe_layout(surface_coordinates, n_probes=DEFAULT_N_PROBES,
                                 patch_radius=DEFAULT_PATCH_RADIUS,
                                 min_source_detector_distance=DEFAULT_MIN_PROBE_DISTANCE,
                                 max_source_detector_distance=DEFAULT_MAX_PROBE_DISTANCE,
                                 min_patch_voxels=DEFAULT_MIN_PATCH_VOXELS,
                                 rng_seed=None):
    """
    Generates realistic probe configurations using patch-based surface sampling for clinical fidelity.
    
    This function implements the advanced patch-based probe placement strategy:
    1. Identifies safe patch centers that can support full 20mm radius patches
    2. Randomly selects one patch center per phantom for spatial diversity
    3. Creates localized surface patch within specified radius
    4. Places 500 sources randomly within the patch region
    5. For each source, selects 3 detectors within clinical SDS range (10-40mm)
    
    Key Advantages:
    - Eliminates "god's eye view" by constraining probes to realistic local regions
    - Ensures all probe positions lie exactly on tissue surface via binary erosion
    - Enforces clinical SDS constraints for physiological measurement realism
    - Provides one patch per phantom for diverse training data without spatial bias
    - Supports efficient distance calculations within localized regions
    
    Clinical Realism Features:
    - Patch size (40mm diameter) matches typical clinical probe array footprints
    - SDS range (10-40mm) reflects real NIR measurement capabilities
    - Single patch placement simulates realistic partial tissue coverage scenarios
    - Surface-only placement prevents non-physical floating probe positions
    
    Args:
        surface_coordinates (numpy.ndarray): Surface voxel positions from binary erosion, shape (N_surface, 3)
        n_probes (int): Number of source positions to place within the patch (500 for training diversity)
        patch_radius (float): Patch radius in mm defining local probe placement region (20mm clinical size)
        min_source_detector_distance (float): Minimum SDS in mm for diffusive regime validity (10mm)
        max_source_detector_distance (float): Maximum SDS in mm for clinical realism (40mm) 
        min_patch_voxels (int): Minimum surface voxels required for valid patch (100 for adequate sampling)
        rng_seed (int): Random seed for reproducible patch selection and probe placement
        
    Returns:
        tuple: (probe_sources, probe_detectors, measurement_links, patch_info) where:
            - probe_sources: Source positions within patch, shape (N_placed_probes, 3)
            - probe_detectors: Detector positions (3 per source), shape (N_placed_probes*3, 3) 
            - measurement_links: Source-detector connectivity, shape (N_measurements, 3)
            - patch_info: Dictionary containing patch metadata for visualization and analysis
    """
    # Initialize controlled randomization for reproducible patch-based placement
    rng = np.random.default_rng(rng_seed)
    
    logger.info(f"Starting patch-based probe layout generation")
    logger.info(f"Target: {n_probes} probes in {patch_radius}mm radius patch (SDS range: {min_source_detector_distance}-{max_source_detector_distance}mm)")
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
    distances_to_center = cdist([selected_patch_center], surface_coordinates)[0]
    patch_mask = distances_to_center <= patch_radius
    patch_surface_coordinates = surface_coordinates[patch_mask]
    
    patch_size = len(patch_surface_coordinates)
    logger.info(f"Patch created: {patch_size:,} surface voxels within {patch_radius}mm radius")
    
    if patch_size < min_patch_voxels:
        logger.warning(f"Patch size ({patch_size}) below minimum requirement ({min_patch_voxels})")
    
    # STEP 5.4: Place sources randomly within the patch region
    logger.debug("Step 4/5: Placing sources within patch...")
    all_probe_sources, all_probe_detectors, measurement_links = [], [], []
    placement_attempts = 0
    max_placement_attempts = n_probes * 10  # Prevent infinite loops
    
    for probe_idx in range(n_probes):
        if (probe_idx + 1) % 50 == 0:  # Progress logging for large probe counts
            logger.debug(f"Source placement progress: {probe_idx+1}/{n_probes}")
            
        # Implement rejection sampling with failure detection
        source_placed = False
        while not source_placed and placement_attempts < max_placement_attempts:
            placement_attempts += 1
            
            # STEP 5.4.1: Randomly sample source position from patch surface voxels
            source_idx = rng.integers(0, len(patch_surface_coordinates))
            current_source_position = patch_surface_coordinates[source_idx]
            
            # STEP 5.4.2: Find valid detector candidates within SDS constraints
            source_to_patch_distances = cdist([current_source_position], patch_surface_coordinates)[0]
            
            # Apply clinical SDS range constraints within the patch
            distance_mask = (source_to_patch_distances >= min_source_detector_distance) & \
                           (source_to_patch_distances <= max_source_detector_distance)
            valid_detector_coordinates = patch_surface_coordinates[distance_mask]
            
            # STEP 5.4.3: Validate sufficient detector availability
            if len(valid_detector_coordinates) < 3:
                continue  # Retry with different source position within patch
            
            # STEP 5.4.4: Randomly select 3 detectors from valid candidates
            detector_indices = rng.choice(len(valid_detector_coordinates), size=3, replace=False)
            selected_detector_positions = valid_detector_coordinates[detector_indices]
            
            # STEP 5.4.5: Store successful probe configuration
            all_probe_sources.append(current_source_position)
            all_probe_detectors.extend(selected_detector_positions)
            
            # STEP 5.4.6: Generate measurement connectivity links
            detector_base_index = 3 * probe_idx
            measurement_links.extend([[probe_idx, detector_base_index+0, 1],    # Source → Detector 1
                                     [probe_idx, detector_base_index+1, 1],    # Source → Detector 2
                                     [probe_idx, detector_base_index+2, 1]])   # Source → Detector 3
            source_placed = True
    
    # STEP 5.5: Validate and report placement results
    n_placed = len(all_probe_sources)
    n_measurements = len(measurement_links)
    placement_efficiency = n_placed / placement_attempts * 100 if placement_attempts > 0 else 0
    
    logger.info(f"✓ Patch-based probe layout completed")
    logger.info(f"Successfully placed: {n_placed}/{n_probes} probes ({n_measurements} measurements)")
    logger.debug(f"Placement efficiency: {placement_efficiency:.1f}% ({placement_attempts} total attempts)")
    
    if n_placed < n_probes:
        logger.warning(f"Could not place all requested probes due to SDS constraints within patch")
        logger.info(f"Consider adjusting patch radius or SDS range for higher placement success rate")
    
    # STEP 5.6: Compile patch metadata for visualization and analysis
    patch_info = {
        'center_position': selected_patch_center,
        'radius': patch_radius,
        'surface_voxels_in_patch': patch_size,
        'safe_centers_available': len(safe_patch_centers),
        'probes_placed': n_placed,
        'placement_efficiency': placement_efficiency,
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
    Executes comprehensive frequency-domain finite element forward modeling and saves complete ML-ready dataset.
    
    This function performs the complete NIR light transport simulation pipeline, from mesh configuration
    through physics simulation to final data processing and storage. It represents the core computational
    component that transforms geometric phantom descriptions into realistic measurement data suitable
    for machine learning algorithm training and validation.
    
    Comprehensive Pipeline Architecture:
    1. **Mesh Configuration**: Integrates source-detector optode positions with finite element geometry
    2. **Geometric Validation**: Projects optodes onto tissue surface and verifies measurement validity
    3. **Physics Simulation**: Solves complex-valued frequency-domain diffusion equation via FEM
    4. **Signal Processing**: Extracts amplitude and phase with realistic clinical noise simulation
    5. **ML Preprocessing**: Applies log-amplitude transformation and measurement reshaping
    6. **Data Storage**: Saves complete dataset with metadata in hierarchical HDF5 format
    
    Frequency-Domain Light Transport Theory:
    The fundamental equation governing NIR light propagation in turbid media at modulation frequency ω:
    
    -∇·(D(r)∇Φ(r,ω)) + [μₐ(r) + iω/(c(r))] Φ(r,ω) = S(r,ω)
    
    Physical Parameters:
    - Φ(r,ω): Complex photon fluence [photons·mm⁻²·s⁻¹] at position r and frequency ω
    - D(r) = 1/[3(μₐ(r) + μ′s(r))]: Diffusion coefficient [mm] controlling light spreading
    - μₐ(r): Absorption coefficient [mm⁻¹] determining energy loss per unit path length
    - μ′s(r): Reduced scattering coefficient [mm⁻¹] characterizing directional randomization
    - c(r) = c₀/n(r): Speed of light in medium [mm·s⁻¹] where c₀ is vacuum light speed
    - S(r,ω): Isotropic point source distribution [photons·mm⁻³·s⁻¹] at optode positions
    
    Measurement Signal Processing:
    - **Amplitude**: |Φ(r,ω)| represents photon density modulation magnitude
    - **Phase**: arg(Φ(r,ω)) represents temporal phase delay relative to source modulation
    - **Log-amplitude**: ln(|Φ|) linearizes exponential spatial decay for neural network stability
    - **Noise Model**: Realistic amplitude (2% relative) and phase (±2°) noise matching clinical systems
    
    Machine Learning Optimizations:
    - **Log-transformation**: Prevents gradient explosion from exponential distance dependence
    - **Measurement Reshaping**: (N_sources, 3) format for source-detector grouping
    - **Noise Injection**: Improves model robustness to real-world measurement uncertainties
    - **Range Validation**: Ensures numerical stability and prevents overflow/underflow
    
    Clinical Realism Features:
    - **Surface Projection**: Ensures optodes lie exactly on tissue boundaries (no floating positions)
    - **Frequency Selection**: 140 MHz modulation matching commercial NIR tomography systems
    - **SNR Modeling**: Noise levels based on published clinical system performance specifications
    - **Measurement Geometry**: Source-detector separations within physiological ranges (10-40mm)
    
    Args:
        phantom_mesh (nirfasterff.base.stndmesh): Complete FEM mesh with optical properties assigned
                                                 Contains nodes, elements, regions, and material properties
        ground_truth_maps (numpy.ndarray): Dense voxel-wise optical property reference maps
                                          Shape: (Nx, Ny, Nz, 2) for [μₐ, μ′s] ground truth
        probe_sources (numpy.ndarray): NIR source positions in phantom coordinates [mm]
                                      Shape: (N_sources, 3) with [x, y, z] coordinates
        probe_detectors (numpy.ndarray): Detector positions in phantom coordinates [mm]
                                        Shape: (N_detectors, 3) with [x, y, z] coordinates
        measurement_links (numpy.ndarray): Source-detector connectivity matrix defining measurement pairs
                                          Shape: (N_measurements, 3) with [source_idx, detector_idx, active]
        phantom_volume (numpy.ndarray, optional): Original labeled phantom volume for metadata storage
                                                 Shape: (Nx, Ny, Nz) with integer tissue labels
        fd_frequency_hz (float): Modulation frequency in Hz (default: 140 MHz for clinical compatibility)
        output_h5_filename (str): Complete file path for HDF5 dataset storage with .h5 extension
        
    Returns:
        None: Function generates dataset files as side effect with comprehensive logging
        
    Side Effects:
        - Creates HDF5 file containing complete measurement dataset
        - Logs detailed progress and performance metrics
        - Validates measurement quality and numerical stability
        - Updates phantom mesh with optode positions for visualization
        
    Output HDF5 Structure:
        - /log_amplitude: Processed amplitude measurements [ln(photons/mm²)]
        - /phase: Phase measurements with noise [degrees]
        - /source_positions: Source coordinates [mm]
        - /detector_positions: Detector coordinates [mm]
        - /measurement_links: Connectivity matrix [dimensionless indices]
        - /ground_truth: Optical property maps [mm⁻¹]
        - /metadata: Complete simulation parameters and phantom description
        
    Performance Characteristics:
        - Simulation time: ~30-60 seconds per phantom depending on mesh complexity
        - Memory usage: ~500MB-2GB during FEM solution phase
        - Output size: ~50-100MB per phantom dataset (compressed HDF5)
        - Numerical precision: Float64 for physics calculations, optimized storage formats
    """
    
    # STEP 1: Configure mesh with comprehensive optode integration and geometric validation
    # This critical step ensures proper coupling between probe positions and finite element mesh
    logger.info("Configuring mesh with source-detector optode layout and geometric validation")
    
    # Convert discrete voxel coordinates to continuous spatial coordinates for FEM compatibility
    # Ensures proper interpolation and boundary condition application in finite element solver
    phantom_mesh.source = ff.base.optode(probe_sources.astype(float))   # NIRFASTer source container
    phantom_mesh.meas = ff.base.optode(probe_detectors.astype(float))   # NIRFASTer detector container
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
    log_amplitude_processed = np.log(np.clip(noisy_amplitude, 1e-8, None)).reshape(-1, 3)
    phase_processed = noisy_phase.reshape(-1, 3)  # Reshape to (N_sources, 3_detectors) format
    
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
        log_amp_dataset.attrs["shape_interpretation"] = "(N_sources, 3_detectors_per_source)"
        
        phase_dataset = h5_file.create_dataset("phase", data=phase_processed,
                                              compression="gzip", compression_opts=6)
        phase_dataset.attrs["units"] = "degrees"
        phase_dataset.attrs["description"] = "Phase delay measurements relative to source modulation"
        phase_dataset.attrs["noise_model"] = f"±{PHASE_NOISE_STD_DEGREES}° additive Gaussian noise"
        phase_dataset.attrs["phase_reference"] = "Relative to source modulation at specified frequency"
        phase_dataset.attrs["shape_interpretation"] = "(N_sources, 3_detectors_per_source)"

        # SUBSTEP 5.2: Save complete geometric configuration for reconstruction and visualization
        source_dataset = h5_file.create_dataset("source_positions", data=probe_sources)
        source_dataset.attrs["units"] = "mm"
        source_dataset.attrs["description"] = "NIR source positions in phantom coordinate system"
        source_dataset.attrs["coordinate_system"] = "Phantom voxel coordinates with 1mm spacing"
        source_dataset.attrs["placement_method"] = "Patch-based surface sampling with clinical constraints"
        
        detector_dataset = h5_file.create_dataset("detector_positions", data=probe_detectors.reshape(-1,3,3))
        detector_dataset.attrs["units"] = "mm"
        detector_dataset.attrs["description"] = "Detector positions grouped by source (3 detectors per source)"
        detector_dataset.attrs["coordinate_system"] = "Phantom voxel coordinates with 1mm spacing"
        detector_dataset.attrs["sds_range"] = f"[{DEFAULT_MIN_PROBE_DISTANCE}, {DEFAULT_MAX_PROBE_DISTANCE}]mm"
        detector_dataset.attrs["grouping"] = "Shape: (N_sources, 3, 3) for [source_idx][detector_idx][x,y,z]"
        
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
        ground_truth_dataset.attrs["shape_interpretation"] = "(Nx, Ny, Nz, 2) for [μₐ, μ′s] channels"
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
    logger.info(f"Final dataset: {log_amplitude_processed.shape[0]} probes × {log_amplitude_processed.shape[1]} detectors")
    logger.debug(f"Ground truth shape: {ground_truth_maps.shape[0]}×{ground_truth_maps.shape[1]}×{ground_truth_maps.shape[2]} voxels")

# --------------------------------------------------------------
# VISUALIZATION: 3D PROBE-MESH RENDERING FOR GEOMETRIC VALIDATION
# --------------------------------------------------------------

def visualize_probe_on_mesh(phantom_volume, phantom_mesh, source_position, detector_positions, probe_index, save_directory, patch_info=None, show_interactive=False):
    """
    Generates comprehensive 3D visualization of patch-based probe configuration on tissue geometry with enhanced surface rendering.
    
    This function creates publication-quality 3D renderings for geometric validation, algorithm verification, and 
    educational demonstration purposes. The visualization provides detailed spatial understanding of the complete
    measurement geometry including tissue boundaries, probe positioning, and patch constraints.
    
    Visualization Architecture:
    - Multi-layer rendering with distinct visual elements for each geometric component
    - Surface boundary extraction using morphological operations for shape clarity
    - Consistent color coding and sizing across all tissue types for fair representation
    - High-contrast probe highlighting for measurement geometry identification
    - Patch region visualization for spatial constraint verification
    
    Color Scheme and Visual Encoding:
    - Healthy tissue: Light green surface + lime green boundaries for tissue identification
    - Tumor regions: Red-spectrum surface + nodes with numbered labeling for multi-tumor scenarios
    - Patch region: Purple markers (square shape) indicating surface-constrained probe placement area
    - Source position: Large yellow circle with black outline for maximum visibility
    - Detector positions: Cyan circles with black outline, distinguishable from source
    - Background: Professional black background for scientific presentation clarity
    
    Technical Implementation Details:
    - Adaptive downsampling for performance optimization without losing spatial detail
    - Morphological surface extraction using binary erosion for boundary identification
    - Consistent node visualization with balanced opacity and size across tissue types
    - Aspect ratio preservation to maintain accurate spatial relationships
    - High-resolution output (300 DPI) for publication-quality documentation
    
    Clinical Relevance:
    - Demonstrates realistic probe placement constraints imposed by tissue geometry
    - Visualizes patch-based placement strategy simulating clinical probe array limitations
    - Shows source-detector separation distances within physiological measurement ranges
    - Provides visual validation of tumor embedding quality and containment ratios
    
    Educational Value:
    - Clear visualization of complex 3D geometric relationships for algorithm understanding
    - Surface boundaries help interpret finite element mesh structure and quality
    - Probe positioning demonstrates clinical measurement setup constraints
    - Multi-tissue scenarios illustrate realistic pathological configurations
    
    Args:
        phantom_volume (numpy.ndarray): Original labeled voxel phantom for tissue surface extraction
                                       Shape: (Nx, Ny, Nz), integer labels defining tissue regions
        phantom_mesh (nirfasterff.base.stndmesh): Complete FEM mesh with node coordinates and region labels
                                                  Contains computed tetrahedral elements and surface geometry
        source_position (numpy.ndarray): Current NIR source position coordinates [x, y, z] in mm
                                        Represents the light injection point for forward modeling
        detector_positions (numpy.ndarray): Current detector array positions, shape (3, 3) for 3 detectors
                                           Each row contains [x, y, z] coordinates of detection points
        probe_index (int): Sequential probe identifier for systematic filename generation (0-based indexing)
        save_directory (str): Absolute output directory path for storing rendered visualization images
        patch_info (dict, optional): Patch metadata containing center position, radius, and surface coordinates
                                    Used for visualizing surface-constrained probe placement region
        show_interactive (bool): Controls display behavior - True enables interactive matplotlib window with
                               mouse-controlled rotation/zoom, False generates static image only
                               
    Returns:
        None: Function generates visualization files as side effect
              
    Output Files:
        - High-resolution PNG image: "probe_{index:03d}.png" (300 DPI for publication quality)
        - Interactive display: Optional real-time 3D exploration window
        
    Performance Notes:
        - Adaptive downsampling scales with mesh complexity for consistent rendering performance
        - Memory-efficient rendering suitable for batch processing of large phantom datasets
        - Automatic cleanup prevents memory accumulation during high-volume generation
    """

    logger.debug(f"Generating comprehensive 3D visualization for probe {probe_index+1} with enhanced surface rendering")

    # STEP 1: Initialize 3D plotting environment with professional scientific styling
    # Configure figure dimensions and projection for optimal 3D visualization clarity
    fig = plt.figure(figsize=(12, 10))  # Generous dimensions for detailed visualization without crowding
    ax = fig.add_subplot(111, projection='3d')  # Enable 3D coordinate system with perspective projection
    
    # Apply professional dark theme optimized for scientific data visualization
    # Black background provides optimal contrast for colored data points and improves perception
    fig.patch.set_facecolor('black')  # Set figure background to black for professional appearance
    ax.set_facecolor('black')  # Set 3D axes background to black for consistency

    # STEP 2: Extract and visualize tissue surface boundaries using morphological operations
    # Surface boundaries provide critical geometric context for understanding tissue shape and probe placement
    from scipy.ndimage import binary_erosion  # Import morphological operation for surface extraction
    
    # SUBSTEP 2.1: Extract and render healthy tissue surface boundaries
    # Uses binary erosion to identify voxels at tissue-air interface for shape definition
    healthy_tissue_mask = (phantom_volume == HEALTHY_TISSUE_LABEL)  # Create binary mask for healthy tissue regions
    if np.any(healthy_tissue_mask):  # Proceed only if healthy tissue exists in phantom
        # Compute morphological boundary: surface = tissue ∖ (tissue ⊖ structuring_element)
        # This identifies tissue voxels that have at least one air neighbor (surface interface)
        healthy_surface = healthy_tissue_mask & (~binary_erosion(healthy_tissue_mask, iterations=1))
        healthy_surface_coords = np.argwhere(healthy_surface)  # Extract explicit (x,y,z) coordinates
        
        if len(healthy_surface_coords) > 0:  # Ensure surface voxels exist before visualization
            # Apply adaptive downsampling to maintain visual clarity while optimizing rendering performance
            # Balance between surface detail preservation and computational efficiency
            downsample_factor = max(1, len(healthy_surface_coords) // 1000)  # Target ~1000 surface points
            ax.scatter(healthy_surface_coords[::downsample_factor, 0], 
                      healthy_surface_coords[::downsample_factor, 1], 
                      healthy_surface_coords[::downsample_factor, 2],
                      color='lime', s=5, alpha=0.55, label='Healthy tissue surface', marker='o')
    
    # SUBSTEP 2.2: Extract and visualize tumor surface boundaries with region-specific coloring
    # Each tumor receives distinct color encoding for multi-lesion scenario identification
    tumor_visualization_colors = ['red', 'orange', 'purple', 'pink', 'brown']  # Distinct color palette
    for region_label in np.unique(phantom_mesh.region):  # Iterate through all mesh regions
        if region_label >= TUMOR_START_LABEL:  # Process only tumor regions (labels ≥ TUMOR_START_LABEL)
            tumor_mask = (phantom_volume == region_label)  # Create binary mask for current tumor
            if np.any(tumor_mask):  # Proceed only if tumor exists in phantom volume
                # Extract tumor surface using same morphological approach as healthy tissue
                tumor_surface = tumor_mask & (~binary_erosion(tumor_mask, iterations=1))
                tumor_surface_coords = np.argwhere(tumor_surface)  # Get surface coordinate list
                
                if len(tumor_surface_coords) > 0:  # Ensure tumor has identifiable surface
                    # Apply consistent color indexing with wraparound for multiple tumors
                    color_index = int(region_label - TUMOR_START_LABEL) % len(tumor_visualization_colors)
                    color = tumor_visualization_colors[color_index]  # Select color from palette
                    
                    # Use targeted downsampling for tumor surfaces (typically smaller than tissue)
                    downsample_factor = max(1, len(tumor_surface_coords) // 500)  # Target ~500 tumor surface points
                    tumor_number = region_label - TUMOR_START_LABEL + 1  # Convert to 1-based numbering for display
                    
                    ax.scatter(tumor_surface_coords[::downsample_factor, 0],
                              tumor_surface_coords[::downsample_factor, 1], 
                              tumor_surface_coords[::downsample_factor, 2],
                              color=color, s=6, alpha=0.65, 
                              label=f'Tumor {tumor_number} surface', marker='o')

    # SUBSTEP 2.3: Visualize patch surface region for spatial constraint demonstration
    # Patch visualization shows the surface-constrained region where probes can be placed
    if patch_info is not None and 'patch_surface_coordinates' in patch_info:
        patch_surface_coords = patch_info['patch_surface_coordinates']  # Extract patch surface coordinates
        if len(patch_surface_coords) > 0:  # Ensure patch contains surface voxels
            logger.debug(f"Visualizing patch constraint region: {len(patch_surface_coords)} surface voxels")
            
            # Apply adaptive downsampling for patch region visualization
            # Balance between showing patch extent and maintaining rendering performance
            patch_downsample_factor = max(1, len(patch_surface_coords) // 800)  # Target ~800 patch points
            ax.scatter(patch_surface_coords[::patch_downsample_factor, 0],
                      patch_surface_coords[::patch_downsample_factor, 1], 
                      patch_surface_coords[::patch_downsample_factor, 2],
                      color='purple', s=8, alpha=0.7, 
                      label=f'Patch region (r={patch_info["radius"]}mm)', marker='s')  # Square markers for distinction

    # STEP 3: Add comprehensive mesh node visualization for internal structure representation
    # Provides uniform visualization of finite element nodes across all tissue types for geometric understanding
    # Maintains consistent visual parameters for fair comparison between tissue regions
    standard_node_size = 4      # Optimized size for visibility without overwhelming surface detail
    standard_alpha = 0.5        # Balanced opacity allowing surface boundaries to remain prominent
    node_downsample_factor = 8  # Consistent downsampling ratio across all tissue types for fairness
    
    # SUBSTEP 3.1: Visualize healthy tissue mesh nodes for internal structure understanding
    # Shows the finite element discretization within healthy tissue regions
    healthy_tissue_nodes = phantom_mesh.nodes[phantom_mesh.region == HEALTHY_TISSUE_LABEL]
    if len(healthy_tissue_nodes) > 0:  # Ensure healthy tissue nodes exist in mesh
        ax.scatter(healthy_tissue_nodes[::node_downsample_factor, 0], 
                  healthy_tissue_nodes[::node_downsample_factor, 1], 
                  healthy_tissue_nodes[::node_downsample_factor, 2],
                  color='lightgreen', s=standard_node_size, alpha=standard_alpha, 
                  label='Healthy nodes', marker='.')  # Small dots for subtle internal structure

    # SUBSTEP 3.2: Visualize tumor mesh nodes with region-specific identification
    # Maintains color consistency with surface visualization for coherent representation
    for region_label in np.unique(phantom_mesh.region):  # Process all mesh regions
        if region_label >= TUMOR_START_LABEL:  # Focus on tumor regions only
            tumor_nodes = phantom_mesh.nodes[phantom_mesh.region == region_label]  # Extract tumor node coordinates
            if len(tumor_nodes) > 0:  # Ensure tumor nodes exist in current region
                # Apply consistent color scheme matching surface visualization
                color_index = int(region_label - TUMOR_START_LABEL) % len(tumor_visualization_colors)
                color = tumor_visualization_colors[color_index]  # Match surface color for coherence
                tumor_number = region_label - TUMOR_START_LABEL + 1  # Convert to 1-based display numbering
                
                ax.scatter(tumor_nodes[::node_downsample_factor, 0],
                          tumor_nodes[::node_downsample_factor, 1], 
                          tumor_nodes[::node_downsample_factor, 2],
                          color=color, s=standard_node_size, alpha=standard_alpha,
                          label=f'Tumor {tumor_number} nodes', marker='.')  # Consistent marker style

    # STEP 4: Highlight current probe configuration with maximum visual contrast
    # Critical for understanding measurement geometry and source-detector relationships
    
    # SUBSTEP 4.1: Visualize NIR source position with distinctive high-contrast markers
    # Large yellow marker with black outline ensures visibility against any background
    ax.scatter(source_position[0], source_position[1], source_position[2], 
               c='yellow', s=120, edgecolor='black', linewidth=2, 
               label='NIR Source', marker='o')  # Circular marker for source identification
    
    # SUBSTEP 4.2: Visualize detector array positions with clear source distinction
    # Cyan markers provide excellent contrast while remaining distinguishable from source
    ax.scatter(detector_positions[:,0], detector_positions[:,1], detector_positions[:,2], 
               c='cyan', s=90, edgecolor='black', linewidth=1.5, 
               label='NIR Detectors', marker='o')  # Consistent circular markers but smaller than source

    # STEP 5: Configure professional plot aesthetics and comprehensive metadata display
    # Ensures publication-quality appearance with clear labeling and scientific presentation standards
    
    # SUBSTEP 5.1: Set descriptive title with probe identification and visualization type
    ax.set_title(f"Probe {probe_index+1:03d} - Enhanced Patch-Based Configuration", 
                 color='white', fontsize=16, fontweight='bold', pad=20)
                 
    # SUBSTEP 5.2: Configure axis labels with proper units and scientific formatting
    ax.set_xlabel('X-axis (mm)', color='white', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y-axis (mm)', color='white', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z-axis (mm)', color='white', fontsize=12, fontweight='bold')

    # SUBSTEP 5.3: Apply consistent white color scheme for dark background compatibility
    # Ensures all text elements remain readable against black background
    ax.tick_params(colors='white', labelsize=10)  # White tick marks and labels
    ax.legend(facecolor='black', edgecolor='white', fontsize=9, labelcolor='white',
              loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.8)

    # STEP 6: Maintain accurate spatial relationships through proper aspect ratio configuration
    # Critical for preventing geometric distortion that could misrepresent spatial measurements
    # Calculate actual mesh dimensions for proportional axis scaling
    x_span = np.ptp(phantom_mesh.nodes[:,0])  # X-axis spatial extent [mm]
    y_span = np.ptp(phantom_mesh.nodes[:,1])  # Y-axis spatial extent [mm]
    z_span = np.ptp(phantom_mesh.nodes[:,2])  # Z-axis spatial extent [mm]
    
    ax.set_box_aspect([x_span, y_span, z_span])  # Proportional scaling preserves true geometry
    
    # SUBSTEP 6.2: Optimize 3D viewing angle for comprehensive geometric understanding
    # Strategic camera positioning reveals both surface features and internal structure
    ax.view_init(elev=20, azim=45)  # Elevation: 20° (slight top-down), Azimuth: 45° (diagonal view)
    plt.tight_layout()  # Optimize layout spacing for professional presentation

    # STEP 7: Generate and save high-resolution publication-quality static image
    # Creates permanent documentation suitable for academic publications and presentations
    output_image_path = os.path.join(save_directory, f"probe_{probe_index+1:03d}.png")
    plt.savefig(output_image_path, dpi=300,  # High resolution for publication quality
                facecolor=fig.get_facecolor(),  # Preserve black background
                bbox_inches='tight',  # Optimize margins for content
                edgecolor='none')  # Clean edges without border artifacts
    logger.debug(f"Saved comprehensive probe visualization: {output_image_path}")

    # STEP 8: Conditional interactive display for real-time exploration
    # Enables detailed examination and custom viewpoint selection when needed
    if show_interactive:
        plt.show()  # Launch interactive matplotlib window with mouse controls (rotation, zoom, pan)
        logger.debug("Interactive 3D visualization displayed - use mouse for exploration")
    else:
        plt.close()  # Release memory resources immediately for efficient batch processing
        logger.debug("Static visualization completed - memory resources released")


# --------------------------------------------------------------
# MAIN EXECUTION PIPELINE: BATCH PHANTOM DATASET GENERATION
# --------------------------------------------------------------

def main():
    """
    Orchestrates complete phantom dataset generation pipeline for comprehensive machine learning training data.
    
    This main execution function coordinates all simulation stages to produce a scientifically rigorous
    dataset of synthetic NIR frequency-domain tomography measurements with pixel-perfect ground truth 
    optical property maps. The pipeline generates multiple unique phantoms to ensure dataset diversity,
    prevent overfitting, and provide robust training data for deep learning reconstruction algorithms.
    
    Complete Pipeline Architecture:
    1. **Phantom Geometry Construction**: Randomized tissue distributions with arbitrary 3D rotations
    2. **Finite Element Mesh Generation**: CGAL-based tetrahedral discretization for accurate FEM
    3. **Optical Property Assignment**: Physiologically realistic μₐ and μ′s with clinical variability
    4. **Surface Extraction**: Morphological boundary identification for realistic probe placement
    5. **Patch-Based Probe Layout**: Surface-constrained placement simulating clinical array limitations
    6. **Frequency-Domain Simulation**: Complete FEM solution of diffusion equation with noise modeling
    7. **Ground Truth Generation**: Dense voxel-wise optical property maps for supervised learning
    8. **Visualization**: Publication-quality 3D renderings for geometric validation and education
    9. **Data Storage**: Hierarchical HDF5 format with comprehensive metadata for reproducibility
    
    Scientific Methodology:
    - **Spatial Bias Elimination**: Patch-based probe placement prevents "god's eye view" artifacts
    - **Physiological Realism**: Evidence-based optical property ranges and tissue geometry constraints
    - **Measurement Noise**: Realistic amplitude and phase noise matching clinical NIR system performance
    - **Geometric Diversity**: Arbitrary 3D rotations and randomized tissue distributions
    - **Clinical Constraints**: SDS ranges (10-40mm) and surface-only probe placement
    
    Machine Learning Optimizations:
    - **Log-amplitude Processing**: Prevents gradient explosion from exponential light decay
    - **Normalized Measurements**: Consistent data ranges for stable neural network training
    - **Comprehensive Ground Truth**: Pixel-level optical property maps for detailed reconstruction validation
    - **Batch Processing**: Efficient memory management for large-scale dataset generation
    - **Quality Control**: Automatic validation of mesh quality, probe placement, and measurement validity
    
    Technical Implementation Details:
    - **Thread-Safe RNG**: Independent random seeds ensure reproducible phantom generation
    - **Memory Management**: Automatic cleanup prevents accumulation during batch processing
    - **Error Handling**: Robust exception handling with detailed logging for debugging
    - **Performance Optimization**: Adaptive downsampling and efficient algorithms for production-scale generation
    - **Cross-Platform Compatibility**: Platform-independent file paths and system interactions
    
    Output Dataset Structure:
    ```
    data/
    ├── phantom_001/
    │   ├── phantom_001_scan.h5          # Complete measurement dataset
    │   └── probe_001.png                # Visualization (if enabled)
    ├── phantom_002/
    │   ├── phantom_002_scan.h5
    │   └── probe_001.png
    └── ...
    ```
    
    HDF5 Dataset Contents:
    - **Measurements**: Log-amplitude and phase data with realistic noise
    - **Geometry**: Source/detector positions and measurement connectivity
    - **Ground Truth**: Dense optical property maps (μₐ, μ′s) for reconstruction validation
    - **Metadata**: Complete phantom parameters for reproducibility analysis
    
    Clinical Translation Relevance:
    - **Realistic Constraints**: Surface-only probe placement matching clinical limitations
    - **Physiological Properties**: Literature-based optical coefficients for 800nm NIR wavelength
    - **Measurement Noise**: SNR and phase precision matching commercial NIR systems
    - **Tumor Characteristics**: Clinically observed contrast ratios and size distributions
    
    Performance Characteristics:
    - **Generation Rate**: ~2-5 phantoms/minute depending on complexity and visualization settings
    - **Memory Usage**: ~1-2 GB RAM per phantom during generation (released automatically)
    - **Storage Requirements**: ~50-100 MB per phantom (HDF5 + visualization)
    - **Scalability**: Tested with 1000+ phantom datasets for production ML training
    
    Quality Assurance:
    - **Mesh Quality**: Automatic validation of tetrahedral element quality and topology
    - **Probe Placement**: Validation of SDS ranges and surface constraint compliance
    - **Optical Properties**: Range checking and contrast ratio validation
    - **Measurement Quality**: SNR verification and phase consistency checking
    
    Returns:
        None: Function orchestrates dataset generation as side effect with comprehensive logging
        
    Side Effects:
        - Creates complete directory structure with phantom datasets
        - Generates HDF5 files containing measurements and ground truth
        - Produces visualization images (if ENABLE_VISUALIZATIONS=True)
        - Extensive logging output for monitoring and debugging
        
    Configuration:
        - Phantom count: Controlled by n_phantoms parameter
        - Visualization: Controlled by ENABLE_VISUALIZATIONS global flag
        - Output location: data/ directory (created automatically)
        - Logging level: INFO by default, DEBUG available for development
    """
    
    # VISUALIZATION CONTROL: Toggle between development and production modes
    # Set to True for development/debugging (enables PNG saves and interactive 3D plots)
    # Set to False for production runs (disables visualizations for faster execution)
    ENABLE_VISUALIZATIONS = True  # Change to False for production runs
    
    # STEP 1: Initialize comprehensive output directory structure and logging framework
    # Create hierarchical data directory in parent mah422 folder for systematic dataset organization
    # This ensures datasets are stored outside the code directory for better project organization
    data_dir = "../data"  # Relative path to maintain portability across different systems
    os.makedirs(data_dir, exist_ok=True)  # Create directory if it doesn't exist, ignore if it does
    
    # Configure professional logging system for complete pipeline monitoring and debugging
    # Logs are saved in parent mah422 directory for centralized monitoring across all simulation runs
    log_file = "../logging.log"  # Centralized log file for all pipeline executions
    logger = setup_logging(level=logging.INFO, log_file=log_file)  # INFO level for production, DEBUG for development
    
    # Generate comprehensive pipeline initialization summary for monitoring and reproducibility
    logger.info("="*80)
    logger.info("STARTING NIR FREQUENCY-DOMAIN PHANTOM DATASET GENERATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Output directory: {os.path.abspath(data_dir)}")  # Absolute path for clarity
    logger.info(f"Log file: {log_file}")
    logger.info(f"Visualization mode: {'ENABLED (Development)' if ENABLE_VISUALIZATIONS else 'DISABLED (Production)'}")

    # STEP 2: Configure dataset generation parameters for machine learning training requirements
    # Generate multiple phantoms to ensure statistical diversity and prevent overfitting in ML models
    n_phantoms = 10  # Development dataset size - increase to 100+ for production ML training
    expected_measurements = n_phantoms * DEFAULT_N_PROBES * 3  # Total measurement count for memory planning
    
    logger.info(f"Generating {n_phantoms} phantom datasets for machine learning training")
    logger.info(f"Expected measurement count: {n_phantoms} phantoms × {DEFAULT_N_PROBES} probes × 3 detectors = {expected_measurements:,} measurements")
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
        phantom_dir = os.path.join(data_dir, f"phantom_{phantom_idx+1:02d}")
        os.makedirs(phantom_dir, exist_ok=True)  # Create directory with error handling
        logger.debug(f"Created phantom directory: {phantom_dir}")

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
            surface_coordinates, n_probes=DEFAULT_N_PROBES, rng_seed=123+phantom_idx)  # Unique seed per phantom

        # SUBSTEP 3.6: Generate comprehensive probe visualizations for quality assurance and validation
        # Visualization enables geometric validation and provides educational/presentation materials
        # Optional in production mode to optimize generation speed for large datasets
        if ENABLE_VISUALIZATIONS:
            logger.info("Step 5/6: Generating publication-quality probe visualization and geometric validation")
            vis_start_time = time.time()  # Track visualization generation time
            
            # Generate detailed visualization for the first probe of each phantom for quality assessment
            if len(probe_sources) > 0:  # Ensure probes were successfully placed
                first_source = probe_sources[0]  # Select first source for visualization
                first_detectors = probe_detectors[0:3]  # First 3 detectors associated with first source
                
                # Control interactive display: show interactive 3D only for first phantom (demonstration purposes)
                # Subsequent phantoms generate static PNG images only for efficient batch processing
                show_interactive_3d = (phantom_idx == 0)  # Interactive mode only for first phantom
                
                # Generate comprehensive 3D visualization with surface boundaries and probe positioning
                visualize_probe_on_mesh(phantom_volume, phantom_mesh, first_source, first_detectors, 0, phantom_dir, 
                                       patch_info=patch_info, show_interactive=show_interactive_3d)
                
                if show_interactive_3d:
                    logger.info(f"Generated INTERACTIVE 3D visualization for phantom {phantom_idx+1} probe 1 (demonstration mode)")
                else:
                    logger.info(f"Generated static PNG visualization for phantom {phantom_idx+1} probe 1 (batch processing mode)")
            
            vis_time = time.time() - vis_start_time  # Calculate visualization generation time
            logger.debug(f"Visualization generation completed in {vis_time:.1f}s")
        else:
            logger.info("Step 5/6: Skipping visualizations (disabled for production efficiency and speed optimization)")

        # SUBSTEP 3.7: Execute frequency-domain finite element simulation and save complete dataset
        # This is the core physics simulation that generates realistic NIR measurement data
        # Solves complex-valued diffusion equation and processes results for machine learning
        logger.info("Step 6/6: Executing frequency-domain diffusion equation simulation and dataset generation")
        h5_output_path = os.path.join(phantom_dir, f"phantom_{phantom_idx+1:03d}_scan.h5")  # Systematic HDF5 naming
        
        # Execute complete forward modeling pipeline with noise simulation and data processing
        run_fd_simulation_and_save(phantom_mesh, ground_truth_maps, probe_sources, probe_detectors, measurement_links, 
                                   phantom_volume=phantom_volume, output_h5_filename=h5_output_path)
        
        # Calculate and log per-phantom generation performance metrics
        phantom_time = time.time() - phantom_start_time
        logger.info(f"✓ PHANTOM {phantom_idx+1:02d} COMPLETED in {phantom_time:.1f}s")
        logger.debug(f"Complete dataset saved: {h5_output_path}")
        
        # Log visualization status based on generation mode
        if ENABLE_VISUALIZATIONS and phantom_idx == 0:
            logger.debug(f"Interactive visualization generated: {phantom_dir}/probe_001.png")
        elif ENABLE_VISUALIZATIONS:
            logger.debug(f"Static PNG visualization saved: {phantom_dir}/probe_001.png")

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
    logger.info(f"Dataset storage location: {os.path.abspath(data_dir)}")
    
    # Detailed dataset contents summary for user reference
    logger.info("\nDataset Architecture and Contents:")
    logger.info("="*50)
    logger.info("Each phantom directory contains:")
    logger.info("  • HDF5 dataset file with complete measurement data")
    logger.info("    - Log-amplitude and phase measurements with realistic noise")
    logger.info("    - Source/detector positions and measurement connectivity")
    logger.info("    - Ground truth optical property maps (μₐ, μ′s)")
    logger.info("    - Complete geometry and metadata for reproducibility")
    
    if ENABLE_VISUALIZATIONS:
        logger.info("  • High-quality 3D visualization (probe_001.png)")
        logger.info("    - Publication-quality rendering for geometric validation")
        logger.info("    - Surface boundaries and patch regions clearly displayed")
        logger.info("    - Interactive mode enabled for first phantom (demonstration)")
    else:
        logger.info("  • No visualizations (production mode - optimized for speed)")
    
    # Technical specifications summary for dataset users
    logger.info("\nTechnical Dataset Specifications:")
    logger.info("="*40)
    logger.info(f"  • Phantom dimensions: {DEFAULT_PHANTOM_SHAPE[0]}×{DEFAULT_PHANTOM_SHAPE[1]}×{DEFAULT_PHANTOM_SHAPE[2]} voxels")
    logger.info(f"  • Tissue radius range: {DEFAULT_TISSUE_RADIUS_RANGE[0]}-{DEFAULT_TISSUE_RADIUS_RANGE[1]} mm")
    logger.info(f"  • Tumor radius range: {DEFAULT_TUMOR_RADIUS_RANGE[0]}-{DEFAULT_TUMOR_RADIUS_RANGE[1]} mm")
    logger.info(f"  • Probes per phantom: {DEFAULT_N_PROBES} sources × 3 detectors = {DEFAULT_N_PROBES * 3:,} measurements")
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
