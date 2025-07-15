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
sys.path.append("/Users/maxhart/Documents/MSc_AI_ML/Dissertation/gitlab/mah422/nirfaster-FF")

# Core numerical and scientific computing libraries
import numpy as np                    # Primary array operations and linear algebra
import nirfasterff as ff             # NIR light transport modeling via finite elements
import matplotlib.pyplot as plt      # 2D/3D visualization and plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D scatter plot capabilities

# Data storage and file system operations
import h5py                          # HDF5 hierarchical data format for large datasets
import os                            # Operating system interface for directory management
import logging                       # Professional logging system for pipeline monitoring
import time                          # Time utilities for performance monitoring

# Specialized scipy modules for morphological and spatial operations
from scipy.ndimage import binary_erosion      # Morphological operation for surface extraction
from scipy.spatial.distance import cdist     # Efficient pairwise distance computation

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

def build_phantom_with_tissue_and_tumours(cube_shape=(50,50,50),
                                          tissue_radius_range=(15,23),
                                          max_tumours=5,
                                          tumour_radius_range=(4,8),
                                          air_label=0,
                                          tissue_label=1,
                                          tumour_start_label=2,
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
        cube_shape (tuple): 3D dimensions of the phantom domain in voxels (Nx, Ny, Nz)
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
    Nx, Ny, Nz = cube_shape
    
    # Create background air-filled volume using efficient memory allocation
    vol = np.full(cube_shape, air_label, dtype=np.uint8)

    # Pre-compute 3D coordinate meshgrids for vectorized ellipsoid calculations
    # This avoids nested loops and leverages NumPy's broadcasting capabilities
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')

    def random_rotation_matrix(rng):
        """Generate a random 3D rotation matrix using Euler angles for arbitrary ellipsoid orientations"""
        # Sample random Euler angles for full 3D rotation coverage
        alpha = rng.uniform(0, 2*np.pi)  # Rotation around z-axis (0 to 360°)
        beta = rng.uniform(0, np.pi)     # Rotation around y-axis (0 to 180°)
        gamma = rng.uniform(0, 2*np.pi)  # Rotation around x-axis (0 to 360°)
        
        # Construct individual rotation matrices using standard rotation formulas
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(gamma), -np.sin(gamma)],
                      [0, np.sin(gamma), np.cos(gamma)]])
                      
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])
                      
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
        
        # Combine rotations: R = Rz * Ry * Rx (order matters for Euler angles)
        return Rz @ Ry @ Rx

    # ----------------------
    # ROTATED HEALTHY TISSUE ELLIPSOID EMBEDDING
    # ----------------------
    # Position the primary tissue ellipsoid at the geometric center of the domain
    cx, cy, cz = Nx//2, Ny//2, Nz//2  # Integer division ensures exact center positioning
    
    # Sample random semi-axes lengths for tissue ellipsoid anisotropy
    rx = rng.integers(*tissue_radius_range)  # Semi-axis length in x-direction
    ry = rng.integers(*tissue_radius_range)  # Semi-axis length in y-direction  
    rz = rng.integers(*tissue_radius_range)  # Semi-axis length in z-direction
    
    # Generate random 3D rotation matrix for arbitrary ellipsoid orientation
    # This eliminates directional bias by creating ellipsoids at random poses
    rotation_matrix = random_rotation_matrix(rng)
    
    # Apply rotation to coordinate system for arbitrary ellipsoid orientation
    # Transform coordinates: (X-cx, Y-cy, Z-cz) -> rotated coordinates
    coords = np.stack([X-cx, Y-cy, Z-cz], axis=-1)  # Shape: (Nx, Ny, Nz, 3)
    rotated_coords = coords @ rotation_matrix.T  # Apply rotation transformation
    
    # Extract rotated coordinate components
    X_rot = rotated_coords[..., 0]
    Y_rot = rotated_coords[..., 1]
    Z_rot = rotated_coords[..., 2]
    
    # Compute tissue mask using rotated ellipsoid equation: (X'/rx)² + (Y'/ry)² + (Z'/rz)² ≤ 1
    # where X', Y', Z' are the rotated coordinates
    tissue_mask = ((X_rot/rx)**2 + (Y_rot/ry)**2 + (Z_rot/rz)**2) <= 1
    vol[tissue_mask] = tissue_label

    # ----------------------
    # ROTATED TUMOUR INCLUSION EMBEDDING WITH ENHANCED SPATIAL CONSTRAINTS
    # ----------------------
    # Sample the number of tumour inclusions from uniform distribution [0, max_tumours]
    n_tumours = rng.integers(0, max_tumours+1)
    logger.info(f"Generating phantom with {n_tumours} tumour(s) using seed {rng_seed}")
    logger.debug(f"Tissue ellipsoid parameters: center=({cx},{cy},{cz}), radii=({rx},{ry},{rz})")
    current_label = tumour_start_label

    # Iteratively place each tumour with geometric validity checking
    for tumour_idx in range(n_tumours):
        attempts = 0
        max_attempts = 50  # Increased attempts for better tumor placement success
        logger.debug(f"Starting placement attempts for tumor {tumour_idx+1}")
        
        # Implement rejection sampling with maximum attempt limit to prevent infinite loops
        while attempts < max_attempts:
            # Sample tumour center coordinates within the tissue ellipsoid bounds
            # Apply safety margins (±3 voxels) to ensure tumours don't extend beyond tissue
            cx_t = rng.integers(cx-rx+3, cx+rx-3)
            cy_t = rng.integers(cy-ry+3, cy+ry-3) 
            cz_t = rng.integers(cz-rz+3, cz+rz-3)
            
            # Sample tumour ellipsoid dimensions with physiological constraints
            rx_t = rng.integers(*tumour_radius_range)
            ry_t = rng.integers(*tumour_radius_range)
            # Constrain z-axis to maintain realistic aspect ratios and prevent elongated artifacts
            # Ensure minimum radius is at least 3, and maximum doesn't exceed the smaller of rx_t, ry_t
            max_rz = min(tumour_radius_range[1], max(rx_t, ry_t))
            rz_t = rng.integers(3, max(4, max_rz + 1))  # Ensure valid range

            logger.debug(f"Tumor {tumour_idx+1} attempt {attempts+1}: center=({cx_t},{cy_t},{cz_t}), radii=({rx_t},{ry_t},{rz_t})")

            # Generate random rotation matrix for arbitrary tumor orientation
            # This ensures tumors also have random poses, not just axis-aligned
            tumor_rotation = random_rotation_matrix(rng)
            
            # Apply rotation to tumor coordinate system
            tumor_coords = np.stack([X-cx_t, Y-cy_t, Z-cz_t], axis=-1)  # Shape: (Nx, Ny, Nz, 3)
            rotated_tumor_coords = tumor_coords @ tumor_rotation.T  # Apply tumor rotation
            
            # Extract rotated tumor coordinate components
            X_tumor_rot = rotated_tumor_coords[..., 0]
            Y_tumor_rot = rotated_tumor_coords[..., 1]
            Z_tumor_rot = rotated_tumor_coords[..., 2]

            # Compute tumour mask using rotated ellipsoid formulation
            tumour_mask = ((X_tumor_rot/rx_t)**2 + (Y_tumor_rot/ry_t)**2 + (Z_tumor_rot/rz_t)**2) <= 1
            
            # Enhanced containment validation: require 80% of tumor volume inside tissue
            tumor_voxels = np.sum(tumour_mask)  # Total tumor voxels
            contained_voxels = np.sum(tumour_mask & tissue_mask)  # Tumor voxels within tissue
            
            if tumor_voxels > 0:
                embedding_ratio = contained_voxels / tumor_voxels
                
                if embedding_ratio >= 0.80:  # Require 80% containment for well-embedded tumors
                    # Apply tumour label ONLY to tumor voxels that are inside tissue (clipping approach)
                    # This ensures no "floating tumor" voxels exist in air space
                    vol[tumour_mask & tissue_mask] = current_label
                    current_label += 1  # Increment label for next tumour
                    logger.info(f"Tumor {tumour_idx+1} successfully placed: {contained_voxels}/{tumor_voxels} voxels ({embedding_ratio:.1%} embedded)")
                    break  # Exit retry loop on successful placement
                else:
                    logger.debug(f"Tumor {tumour_idx+1} attempt {attempts+1}: insufficient embedding {embedding_ratio:.1%} < 80%")
            
            attempts += 1
        
        if attempts >= max_attempts:
            logger.warning(f"Failed to place tumor {tumour_idx+1} after {max_attempts} attempts - skipping")

    # Calculate final tissue coverage statistics
    total_voxels = Nx * Ny * Nz
    tissue_voxels = np.sum(vol == tissue_label)  # Healthy tissue only
    tumor_voxels = np.sum(vol >= tumour_start_label)  # All tumor regions
    total_tissue_voxels = tissue_voxels + tumor_voxels  # Total tissue (healthy + tumors)
    tissue_coverage = total_tissue_voxels / total_voxels
    
    logger.info(f"Phantom construction completed: {tissue_coverage:.1%} tissue coverage")
    logger.debug(f"Final voxel counts - Air: {np.sum(vol==0):,}, Healthy tissue: {tissue_voxels:,}, Tumors: {tumor_voxels:,}, Total tissue: {total_tissue_voxels:,}")
    logger.debug(f"Coverage breakdown - Healthy: {tissue_voxels/total_voxels:.1%}, Tumors: {tumor_voxels/total_voxels:.1%}, Total: {tissue_coverage:.1%}")

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
    params = ff.utils.MeshingParams()
    # Set characteristic cell size balancing accuracy and computational cost
    # Smaller values increase mesh resolution but exponentially increase solve time
    params.general_cell_size = 1.65  # Empirically optimized for NIR wavelengths (λ ≈ 800nm)
    
    logger.info(f"Starting CGAL-based tetrahedral mesh generation with cell_size={params.general_cell_size}")
    logger.debug(f"Input volume shape: {volume.shape}, unique labels: {np.unique(volume)}")
    
    # Execute CGAL-based tetrahedral mesh generation
    # This calls external C++ CGAL library for robust geometric mesh generation
    ele, nodes = ff.meshing.RunCGALMeshGenerator(volume, opt=params)
    
    # Perform comprehensive mesh quality validation
    # Checks for inverted elements, aspect ratios, and topological consistency
    ff.meshing.CheckMesh3D(ele, nodes)
    
    logger.info(f"Mesh generation completed: {ele.shape[0]} tetrahedra, {nodes.shape[0]} nodes")
    logger.debug(f"Mesh quality validation passed successfully")
    
    return ele, nodes

def create_stndmesh(ele, nodes):
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
        ele (numpy.ndarray): Element connectivity matrix from mesh generation
        nodes (numpy.ndarray): Node coordinate matrix from mesh generation
        
    Returns:
        nirfasterff.base.stndmesh: Complete mesh object ready for optical property assignment
                                  and finite element simulation
    """
    # Initialize NIRFASTer standard mesh data structure
    mesh = ff.base.stndmesh()
    
    logger.debug("Creating standardized mesh object from raw connectivity data")
    
    # Populate mesh with tetrahedral solid geometry and compute derived quantities
    # This automatically calculates:
    # - Element volumes via determinant of Jacobian matrix
    # - Surface face identification through adjacency analysis  
    # - Boundary node marking for Dirichlet/Robin boundary conditions
    mesh.from_solid(ele, nodes)
    
    # Extract and display mesh quality statistics for validation
    mean_vol = mesh.element_area.mean()  # Note: 'element_area' actually stores volumes for 3D elements
    std_vol = mesh.element_area.std()
    logger.info(f"Mesh statistics: {ele.shape[0]} tetrahedra, {nodes.shape[0]} nodes")
    logger.info(f"Element volume statistics: {mean_vol:.3f} ± {std_vol:.3f} mm³")
    
    # Validate mesh quality metrics are within acceptable bounds
    if mean_vol < 0.1 or mean_vol > 10.0:
        logger.warning(f"Element volumes ({mean_vol:.3f} mm³) may be outside optimal range for FEM convergence")
    else:
        logger.debug("Mesh quality metrics are within acceptable bounds")
    
    return mesh


# --------------------------------------------------------------
# STEP 3: OPTICAL PROPERTY ASSIGNMENT AND GROUND TRUTH GENERATION
# --------------------------------------------------------------

def assign_optical_properties(mesh, volume, rng_seed=None):
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
        mesh (nirfasterff.base.stndmesh): Finite element mesh with region labels
        volume (numpy.ndarray): Original voxel-based phantom for ground truth mapping
        rng_seed (int): Random seed for reproducible property assignment
        
    Returns:
        tuple: (mesh, ground_truth_grid) where:
            - mesh: Updated mesh object with optical properties assigned to each element
            - ground_truth_grid: Dense (Nx, Ny, Nz, 2) array containing μₐ and μ′s maps
    """
    # Initialize controlled random number generator for reproducible property sampling
    rng = np.random.default_rng(rng_seed)
    
    # Extract unique region labels from mesh elements
    # mesh.region contains the tissue type label for each tetrahedral element
    regions = np.unique(mesh.region)
    prop = []  # Will store [region_id, μₐ, μ′s, n] for each tissue type

    logger.info(f"Assigning optical properties to {len(regions)} tissue regions using seed {rng_seed}")
    logger.debug(f"Found regions: {regions}")

    # Sample baseline healthy tissue optical properties from physiological distributions
    # These serve as reference values for relative tumour property scaling
    mua_sub = rng.uniform(0.003, 0.007)    # Absorption coeff. [mm⁻¹] - controls image contrast
    musp_sub = rng.uniform(0.78, 1.18)     # Reduced scattering [mm⁻¹] - controls penetration depth
    
    logger.debug(f"Baseline tissue properties: μₐ={mua_sub:.4f} mm⁻¹, μ′s={musp_sub:.3f} mm⁻¹")
    
    # Dictionary for efficient ground truth lookup during voxel assignment
    region_lookup = {}

    # Assign optical properties to each tissue region with appropriate physiological scaling
    for region in regions:
        if region == 1:  # Healthy tissue baseline
            mua, musp = mua_sub, musp_sub
            tissue_type = "healthy"
        else:  # Tumour regions (label ≥ 2)
            # Apply controlled randomization within clinically observed ranges
            # Tumours typically show increased absorption (higher blood volume)
            # and altered scattering (modified cellular architecture)
            mua = mua_sub * rng.uniform(1.5, 3.5)   # 50-250% increase in absorption
            musp = musp_sub * rng.uniform(1.5, 2.5)  # 50-150% increase in scattering
            tissue_type = f"tumor_{region-1}"
            
        logger.debug(f"Region {region} ({tissue_type}): μₐ={mua:.4f} mm⁻¹, μ′s={musp:.3f} mm⁻¹")
            
        # Store optical properties in NIRFASTer format: [region, μₐ, μ′s, n]
        # Refractive index n=1.33 is fixed for biological tissues at NIR wavelengths
        prop.append([region, mua, musp, 1.33])
        region_lookup[region] = (mua, musp)

    # Apply optical properties to mesh for FEM simulation
    # This populates mesh.mua and mesh.musp arrays used in diffusion equation assembly
    mesh.set_prop(np.array(prop))
    logger.info("Optical properties successfully assigned to mesh elements")

    # Generate dense voxel-wise ground truth maps for reconstruction evaluation
    Nx, Ny, Nz = volume.shape
    # Shape: (Nx, Ny, Nz, 2) where last dimension is [μₐ, μ′s]
    gt_grid = np.zeros((Nx, Ny, Nz, 2))
    
    # Populate ground truth grid using region-based property lookup
    # This creates pixel-perfect reference maps for quantitative evaluation
    for region, (mua, musp) in region_lookup.items():
        # Apply properties to all voxels belonging to this tissue region
        gt_grid[volume==region, 0] = mua   # Channel 0: absorption coefficient
        gt_grid[volume==region, 1] = musp  # Channel 1: reduced scattering coefficient

    logger.info(f"Ground truth maps generated: shape {gt_grid.shape}")
    logger.debug(f"Ground truth value ranges: μₐ=[{gt_grid[...,0].min():.4f}, {gt_grid[...,0].max():.4f}], μ′s=[{gt_grid[...,1].min():.3f}, {gt_grid[...,1].max():.3f}]")

    return mesh, gt_grid


# --------------------------------------------------------------
# STEP 4: TISSUE SURFACE EXTRACTION VIA MORPHOLOGICAL OPERATIONS
# --------------------------------------------------------------

def extract_surface_voxels(volume, tissue_threshold=1):
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
        volume (numpy.ndarray): 3D labeled phantom volume with tissue regions
        tissue_threshold (int): Minimum label value considered as tissue (excludes air=0)
        
    Returns:
        numpy.ndarray: Surface voxel coordinates, shape (N_surface, 3)
                      Each row contains (x, y, z) indices of surface voxels
    """
    # Create binary mask identifying all tissue regions (healthy + tumours)
    # This combines all non-air labels into a single binary volume
    binary_mask = (volume >= tissue_threshold)
    
    logger.debug(f"Surface extraction from volume shape {volume.shape} with threshold {tissue_threshold}")
    logger.debug(f"Initial tissue voxels: {np.sum(binary_mask):,}")
    
    # Apply binary erosion with single iteration to identify tissue interior
    # Uses default 3×3×3 structuring element for 26-connected neighborhood
    # Interior voxels are those completely surrounded by other tissue voxels
    eroded = binary_erosion(binary_mask, iterations=1)
    
    # Compute morphological boundary as set difference: tissue ∖ interior
    # Surface voxels are tissue voxels that have at least one air neighbor
    surface_mask = binary_mask & (~eroded)
    
    # Extract explicit (x,y,z) coordinates of surface voxels
    # numpy.argwhere returns N×3 array of indices where condition is True
    surface_coords = np.argwhere(surface_mask)
    
    logger.info(f"Surface extraction completed: {surface_coords.shape[0]:,} surface voxels identified")
    
    # Validate surface extraction results
    if surface_coords.shape[0] == 0:
        logger.error("No surface voxels found - check tissue geometry and threshold")
    elif surface_coords.shape[0] < 100:
        logger.warning(f"Very few surface voxels ({surface_coords.shape[0]}) - may limit probe placement options")
    else:
        surface_ratio = surface_coords.shape[0] / np.sum(binary_mask)
        logger.debug(f"Surface-to-volume ratio: {surface_ratio:.1%}")
    
    return surface_coords


# --------------------------------------------------------------
# STEP 5: SPATIALLY-INVARIANT PROBE LAYOUT GENERATION
# --------------------------------------------------------------

def build_random_probe_layout(surface_coords, n_probes=256,
                              min_distance=5,
                              rng_seed=None):
    """
    Generates fully randomized source-detector probe configurations for spatially-invariant learning.
    
    This function implements robust probe placement strategy that eliminates spatial bias in training data:
    - Uniformly samples source positions across entire tissue surface
    - Applies distance-constrained detector placement for realistic measurement sensitivity
    - Uses rejection sampling to ensure geometric validity of all probe configurations
    - Generates multiple detectors per source to increase measurement information density
    
    Clinical Motivation:
    In real NIR imaging, probe placement is constrained by patient anatomy and clinical protocols.
    This simulation mimics realistic source-detector separations while ensuring comprehensive
    sampling of the measurement space for robust machine learning model training.
    
    Technical Implementation:
    - Source sampling: Uniform random selection from extracted surface voxels
    - Detector constraints: Euclidean distance filtering within physiological ranges
    - Multi-detector probes: 3 detectors per source for improved spatial encoding
    - Rejection sampling: Ensures all probes meet geometric validity requirements
    
    Distance Constraints Rationale:
    - min_distance: Prevents near-field artifacts and ensures diffusive regime validity
    - No max_distance: Allows maximum spatial diversity for robust ML training
    - Typical clinical min_distance: 5-10mm to ensure diffusive light transport
    
    Args:
        surface_coords (numpy.ndarray): Available surface voxel positions, shape (N_surface, 3)
        n_probes (int): Number of source-detector probe configurations to generate
        min_distance (float): Minimum source-detector separation [mm] for diffusive regime
        rng_seed (int): Random seed for reproducible probe layout generation
        
    Returns:
        tuple: (sources, detectors, link) where:
            - sources: Source positions, shape (N_probes, 3) with (x,y,z) coordinates
            - detectors: Detector positions, shape (N_probes*3, 3) stacked for all probes  
            - link: Measurement connectivity matrix, shape (N_measurements, 3)
                   Format: [source_idx, detector_idx, active_flag]
    """
    # Initialize controlled randomization for reproducible probe generation
    rng = np.random.default_rng(rng_seed)
    
    logger.info(f"Starting probe layout generation: {n_probes} probes, min_distance={min_distance}mm (no max limit)")
    logger.debug(f"Available surface coordinates: {len(surface_coords)}")
    
    # Initialize storage arrays for probe configuration data
    all_sources, all_detectors, link = [], [], []
    n_attempts = 0  # Track total placement attempts for convergence monitoring

    # Generate each probe configuration with geometric validation
    for i in range(n_probes):
        if (i + 1) % 50 == 0:  # Progress logging for large probe counts
            logger.debug(f"Probe placement progress: {i+1}/{n_probes}")
            
        # Implement rejection sampling loop with failure detection
        while True:
            n_attempts += 1
            # Prevent infinite loops due to overly restrictive geometric constraints
            if n_attempts > n_probes * 20:
                logger.error(f"Excessive placement failures - stopping early after {n_attempts} attempts")
                logger.warning(f"Successfully placed {len(all_sources)}/{n_probes} requested probes")
                break

            # STEP 5.1: Uniformly sample source position from tissue surface
            source_idx = rng.integers(0, len(surface_coords))
            source_pos = surface_coords[source_idx]
            
            # STEP 5.2: Compute distances from source to all potential detector locations
            # Uses efficient scipy implementation of Euclidean distance computation
            dists = cdist([source_pos], surface_coords)[0]
            
            # STEP 5.3: Filter detector candidates based on distance constraints
            # Ensures measurements are in diffusive regime with unlimited maximum distance
            valid_detectors = surface_coords[dists >= min_distance]

            # STEP 5.4: Validate sufficient detector availability for multi-detector probe
            if valid_detectors.shape[0] < 3:
                continue  # Retry with different source position

            # STEP 5.5: Randomly sample 3 detectors from valid candidates
            # Multiple detectors per source increase measurement information content
            detector_indices = rng.choice(len(valid_detectors), size=3, replace=False)
            detector_positions = valid_detectors[detector_indices]
            
            # STEP 5.6: Calculate and log actual source-detector distances for this probe
            detector_distances = cdist([source_pos], detector_positions)[0]
            logger.info(f"Probe {i+1} placed - Source at {source_pos}, Detector distances: "
                        f"{detector_distances[0]:.1f}mm, {detector_distances[1]:.1f}mm, {detector_distances[2]:.1f}mm")

            # STEP 5.7: Store validated probe configuration
            all_sources.append(source_pos)
            all_detectors.extend(detector_positions)  # Flatten detector array across all probes
            
            # STEP 5.7: Generate measurement connectivity links
            # Links define which source-detector pairs generate measurements
            base = 3 * i  # Detector indexing offset for current probe
            link.extend([[i, base+0, 1],    # Source i → Detector base+0, active
                        [i, base+1, 1],    # Source i → Detector base+1, active  
                        [i, base+2, 1]])   # Source i → Detector base+2, active
            break  # Exit retry loop on successful probe placement

    placement_efficiency = len(all_sources) / n_attempts * 100 if n_attempts > 0 else 0
    logger.info(f"Probe layout completed: {len(all_sources)} probes placed")
    logger.debug(f"Placement statistics: {n_attempts} total attempts, {placement_efficiency:.1f}% efficiency")
    
    # Convert lists to NumPy arrays for efficient numerical processing
    return np.array(all_sources), np.array(all_detectors), np.array(link)


# --------------------------------------------------------------
# STEP 6: FREQUENCY-DOMAIN FORWARD MODELING AND DATA STORAGE
# --------------------------------------------------------------

def run_fd_and_save(mesh, ground_truth, all_sources, all_detectors, link,
                    fd_freq_hz=140e6, h5_filename="phantom_fd_scan.h5"):
    """
    Executes frequency-domain finite element forward modeling and saves complete dataset to HDF5.
    
    This function performs the core NIR light transport simulation and data processing:
    1. Configures mesh with source-detector optode positions and connectivity
    2. Solves frequency-domain diffusion equation via finite element method
    3. Extracts amplitude and phase measurements with realistic noise modeling
    4. Processes measurements into machine learning-ready format
    5. Saves complete dataset with geometry and ground truth for supervised learning
    
    Frequency-Domain Theory:
    The FD diffusion equation at modulation frequency ω is:
    -∇·(D∇Φ) + (μₐ + iωμₐ/(μ′s*c))Φ = S(r)
    where:
    - Φ(r,ω): Complex photon fluence at position r and frequency ω
    - D = 1/[3(μₐ + μ′s)]: Diffusion coefficient  
    - c = c₀/n: Speed of light in medium
    - S(r): Isotropic point source distribution
    
    Measurement Processing:
    - Amplitude: |Φ| represents photon density modulation magnitude
    - Phase: arg(Φ) represents modulation phase delay relative to source
    - Log-amplitude: ln(|Φ|) linearizes exponential decay for neural networks
    
    Args:
        mesh (nirfasterff.base.stndmesh): FEM mesh with optical properties assigned
        ground_truth (numpy.ndarray): Voxel-wise optical property maps for validation
        all_sources (numpy.ndarray): Source positions, shape (N_sources, 3)
        all_detectors (numpy.ndarray): Detector positions, shape (N_detectors, 3)  
        link (numpy.ndarray): Source-detector connectivity, shape (N_measurements, 3)
        fd_freq_hz (float): Modulation frequency in Hz (typical: 100-200 MHz)
        h5_filename (str): Output HDF5 file path for dataset storage
    """
    # STEP 6.1: Configure mesh with optode positions and measurement connectivity
    # Convert integer voxel coordinates to floating-point spatial coordinates [mm]
    mesh.source = ff.base.optode(all_sources.astype(float))
    mesh.meas = ff.base.optode(all_detectors.astype(float))
    mesh.link = link
    
    logger.debug(f"Configured optodes: {len(all_sources)} sources, {len(all_detectors)} detectors")
    
    # Project optodes onto mesh surface and validate geometric consistency
    # This ensures sources/detectors lie exactly on tissue boundary for accurate modeling
    mesh.touch_optodes()
    logger.debug("Optodes projected onto mesh surface")

    # STEP 6.2: Execute frequency-domain finite element forward simulation
    logger.info(f"Starting FD simulation at {fd_freq_hz/1e6:.1f} MHz")
    logger.info(f"Mesh configuration: {mesh.nodes.shape[0]:,} nodes, {mesh.elements.shape[0]:,} elements")
    logger.info(f"Measurement configuration: {len(all_sources)} sources, {len(link)} total measurements")
    
    # Solve complex-valued frequency-domain diffusion equation
    # Returns complex photon fluence at each detector for each source activation
    data, _ = mesh.femdata(fd_freq_hz)
    
    # Extract amplitude and phase from complex solution
    amplitude = data.amplitude  # |Φ|: Photon fluence magnitude
    phase = np.degrees(data.phase)  # arg(Φ): Phase delay in degrees
    
    logger.info("FD simulation completed successfully")
    logger.debug(f"Raw measurement ranges: amplitude=[{amplitude.min():.2e}, {amplitude.max():.2e}], phase=[{phase.min():.1f}°, {phase.max():.1f}°]")

    # STEP 6.3: Add realistic measurement noise for robust model training
    # Noise parameters based on typical experimental NIR system performance
    rng = np.random.default_rng()
    
    # Amplitude noise: Multiplicative noise proportional to signal magnitude
    # Typical SNR for clinical NIR systems: ~40-60 dB
    amplitude_noise_std = 0.02 * np.mean(amplitude)  # 2% relative noise
    amplitude += rng.normal(0, amplitude_noise_std, amplitude.shape)
    
    # Phase noise: Additive Gaussian noise independent of signal magnitude  
    # Typical phase precision: ±1-3 degrees for commercial systems
    phase_noise_std = 2.0  # degrees
    phase += rng.normal(0, phase_noise_std, phase.shape)
    
    logger.debug(f"Applied measurement noise: amplitude_std={amplitude_noise_std:.2e}, phase_std={phase_noise_std}°")

    # STEP 6.4: Process measurements for machine learning compatibility
    # Reshape measurements to probe-based format: (N_probes, 3) for 3 detectors per source
    
    # Log-amplitude transformation for neural network training stability
    # Prevents gradient explosion due to exponential amplitude decay with distance
    log_amp = np.log(np.clip(amplitude, 1e-8, None)).reshape(-1, 3)
    phase = phase.reshape(-1, 3)
    
    logger.debug(f"Processed measurements: log_amplitude=[{log_amp.min():.2f}, {log_amp.max():.2f}], phase=[{phase.min():.1f}°, {phase.max():.1f}°]")

    # STEP 6.5: Save complete dataset to HDF5 with hierarchical structure
    # HDF5 provides efficient storage for large multi-dimensional arrays with metadata
    logger.info(f"Saving dataset to {h5_filename}")
    
    with h5py.File(h5_filename, "w") as f:
        # Create datasets first, then set attributes separately
        log_amp_dset = f.create_dataset("log_amplitude", data=log_amp)
        log_amp_dset.attrs["units"] = "ln(photons/mm²)"
        log_amp_dset.attrs["description"] = "Natural log of photon fluence amplitude"
        
        phase_dset = f.create_dataset("phase", data=phase)
        phase_dset.attrs["units"] = "degrees"
        phase_dset.attrs["description"] = "Phase delay relative to source modulation"
        
        source_dset = f.create_dataset("source_pos", data=all_sources)
        source_dset.attrs["units"] = "mm"
        source_dset.attrs["description"] = "Source positions in mesh coordinates"
        
        det_dset = f.create_dataset("det_pos", data=all_detectors.reshape(-1,3,3))
        det_dset.attrs["units"] = "mm"
        det_dset.attrs["description"] = "Detector positions grouped by probe"
        
        gt_dset = f.create_dataset("ground_truth", data=ground_truth)
        gt_dset.attrs["channels"] = "absorption, reduced_scattering"
        gt_dset.attrs["units"] = "mm^-1"
        gt_dset.attrs["description"] = "Voxel-wise optical property maps"
        
        # Set file-level attributes
        f.attrs["modulation_frequency_hz"] = fd_freq_hz
        f.attrs["noise_amplitude_std"] = amplitude_noise_std
        f.attrs["noise_phase_std"] = phase_noise_std
        f.attrs["n_measurements"] = len(link)
        f.attrs["n_probes"] = len(all_sources)
        f.attrs["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
        
    logger.info("Dataset saved successfully")
    logger.info(f"Final dataset: {log_amp.shape[0]} probes × {log_amp.shape[1]} detectors")
    logger.info(f"Ground truth shape: {ground_truth.shape[0]}×{ground_truth.shape[1]}×{ground_truth.shape[2]} voxels")
    logger.debug(f"HDF5 file size: {os.path.getsize(h5_filename)/1024**2:.1f} MB")

# --------------------------------------------------------------
# VISUALIZATION: 3D PROBE-MESH RENDERING FOR GEOMETRIC VALIDATION
# --------------------------------------------------------------

def visualize_probe_on_mesh(volume, mesh, src, dets, probe_idx, save_dir, show_interactive=False):
    """
    Generates comprehensive 3D visualization of probe configuration on tissue geometry with surface outlines.
    
    This function creates publication-quality 3D renderings for geometric validation and 
    educational purposes, showing the spatial relationship between:
    - Tissue phantom geometry with surface boundaries clearly defined
    - Uniform node visualization across all tissue types
    - Current probe configuration (source position and detector array)
    - Surface outlines for enhanced shape perception
    
    Visualization Strategy:
    - Healthy tissue: Uniform green nodes with surface outline
    - Tumour regions: Uniform red nodes with distinct surface outlines  
    - Current probe: High-contrast yellow source and cyan detectors
    - Surface boundaries: Wireframe outlines to show tissue shapes clearly
    - Background: Black background for professional scientific presentation
    
    Technical Implementation:
    - Uses consistent node sizes and downsampling across all tissue types
    - Adds surface extraction and wireframe rendering for shape clarity
    - Maintains aspect ratio fidelity for accurate spatial relationships
    - Supports both interactive exploration and automated batch image generation
    
    Args:
        volume (numpy.ndarray): Original voxel-based phantom for tissue visualization
        mesh (nirfasterff.base.stndmesh): FEM mesh containing node coordinates and region labels
        src (numpy.ndarray): Current source position [x, y, z] in mm coordinates
        dets (numpy.ndarray): Current detector positions, shape (3, 3) for 3 detectors
        probe_idx (int): Probe index for systematic filename generation
        save_dir (str): Output directory path for saving rendered images
        show_interactive (bool): Whether to display interactive matplotlib window
    """

    logger.debug(f"Generating enhanced visualization for probe {probe_idx+1} with surface outlines")

    # STEP 1: Initialize 3D plotting environment with professional styling
    fig = plt.figure(figsize=(12, 10))  # Larger figure for better detail visibility
    ax = fig.add_subplot(111, projection='3d')
    
    # Apply dark theme for scientific visualization aesthetic
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # STEP 2: Extract and visualize surface boundaries for shape clarity
    from scipy.ndimage import binary_erosion
    
    # Extract healthy tissue surface
    healthy_mask = (volume == 1)
    if np.any(healthy_mask):
        healthy_surface = healthy_mask & (~binary_erosion(healthy_mask, iterations=1))
        healthy_surface_coords = np.argwhere(healthy_surface)
        if len(healthy_surface_coords) > 0:
            # Plot surface boundary with reduced contrast - more balanced with interior
            downsample_factor = max(1, len(healthy_surface_coords) // 1000)  # Adaptive downsampling
            ax.scatter(healthy_surface_coords[::downsample_factor, 0], 
                      healthy_surface_coords[::downsample_factor, 1], 
                      healthy_surface_coords[::downsample_factor, 2],
                      color='lime', s=5, alpha=0.55, label='Healthy tissue surface', marker='o')
    
    # Extract and visualize tumor surfaces with distinct colors  
    tumor_colors = ['red', 'orange', 'purple', 'pink', 'brown']
    for reg in np.unique(mesh.region):
        if reg >= 2:  # Tumour regions have labels ≥ 2
            tumor_mask = (volume == reg)
            if np.any(tumor_mask):
                tumor_surface = tumor_mask & (~binary_erosion(tumor_mask, iterations=1))
                tumor_surface_coords = np.argwhere(tumor_surface)
                if len(tumor_surface_coords) > 0:
                    # Use consistent color indexing: same as interior nodes
                    color = tumor_colors[int(reg-2) % len(tumor_colors)]
                    downsample_factor = max(1, len(tumor_surface_coords) // 500)
                    ax.scatter(tumor_surface_coords[::downsample_factor, 0],
                              tumor_surface_coords[::downsample_factor, 1], 
                              tumor_surface_coords[::downsample_factor, 2],
                              color=color, s=6, alpha=0.65, 
                              label=f'Tumor {int(reg-1)} surface', marker='o')

    # STEP 3: Add uniform mesh node visualization for all tissue types
    # Use consistent size and downsampling for fair representation with balanced contrast
    standard_node_size = 4  # Slightly larger for better visibility
    standard_alpha = 0.5    # Slightly more opaque to balance with surface
    node_downsample = 8     # Consistent downsampling across all tissues
    
    # Healthy tissue nodes
    healthy_nodes = mesh.nodes[mesh.region == 1]
    if len(healthy_nodes) > 0:
        ax.scatter(healthy_nodes[::node_downsample, 0], 
                  healthy_nodes[::node_downsample, 1], 
                  healthy_nodes[::node_downsample, 2],
                  color='lightgreen', s=standard_node_size, alpha=standard_alpha, 
                  label='Healthy nodes', marker='.')

    # Tumor nodes with consistent visualization
    for reg in np.unique(mesh.region):
        if reg >= 2:
            tumor_nodes = mesh.nodes[mesh.region == reg]
            if len(tumor_nodes) > 0:
                color = tumor_colors[int(reg-2) % len(tumor_colors)]
                ax.scatter(tumor_nodes[::node_downsample, 0],
                          tumor_nodes[::node_downsample, 1], 
                          tumor_nodes[::node_downsample, 2],
                          color=color, s=standard_node_size, alpha=standard_alpha,
                          label=f'Tumor {int(reg-1)} nodes', marker='.')

    # STEP 4: Highlight current probe configuration with high-contrast colors
    # Source visualization: Large yellow marker with black outline for maximum visibility
    ax.scatter(src[0], src[1], src[2], c='yellow', s=120, edgecolor='black', 
               linewidth=2, label='Source', marker='o')
    
    # Detector visualization: Cyan markers distinguishable from source
    ax.scatter(dets[:,0], dets[:,1], dets[:,2], c='cyan', s=90, edgecolor='black', 
               linewidth=1.5, label='Detectors', marker='o')

    # STEP 5: Configure plot aesthetics and metadata
    ax.set_title(f"Probe {probe_idx+1:03d} - Enhanced Surface View", 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X (mm)', color='white', fontsize=12)
    ax.set_ylabel('Y (mm)', color='white', fontsize=12)
    ax.set_zlabel('Z (mm)', color='white', fontsize=12)

    # Apply white color scheme to axis elements for dark background compatibility
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', edgecolor='white', fontsize=8, labelcolor='white',
              loc='upper left', bbox_to_anchor=(0.02, 0.98))

    # STEP 6: Maintain realistic aspect ratio based on actual mesh dimensions
    # Prevents distortion that could misrepresent spatial relationships
    ax.set_box_aspect([np.ptp(mesh.nodes[:,0]),   # X-axis span
                       np.ptp(mesh.nodes[:,1]),   # Y-axis span  
                       np.ptp(mesh.nodes[:,2])])  # Z-axis span
    
    # Improve 3D viewing angle for better shape perception
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()

    # STEP 7: Generate and save high-resolution static image
    out_path = os.path.join(save_dir, f"probe_{probe_idx+1:03d}.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), 
                bbox_inches='tight', edgecolor='none')
    logger.debug(f"Saved enhanced probe visualization: {out_path}")

    # STEP 8: Optionally display interactive 3D plot for manual exploration
    if show_interactive:
        plt.show()  # Enables mouse-controlled rotation and zooming
    else:
        plt.close()  # Free memory for batch processing


# --------------------------------------------------------------
# MAIN EXECUTION PIPELINE: BATCH PHANTOM DATASET GENERATION
# --------------------------------------------------------------

def main():
    """
    Orchestrates complete phantom dataset generation pipeline for machine learning training.
    
    This main execution function coordinates all simulation stages to produce a comprehensive
    dataset of synthetic NIR tomography measurements with ground truth optical property maps.
    The pipeline generates multiple unique phantoms to ensure dataset diversity and prevent
    overfitting in machine learning models.
    
    Pipeline Stages:
    1. Phantom geometry construction with randomized tissue distributions
    2. Finite element mesh generation for accurate light transport modeling  
    3. Optical property assignment with physiological parameter ranges
    4. Surface-constrained probe placement for realistic measurement configurations
    5. Frequency-domain forward modeling with noise simulation
    6. Comprehensive visualization for quality assurance and validation
    7. HDF5 dataset storage with complete metadata preservation
    
    Dataset Structure:
    Each phantom generates:
    - HDF5 file containing measurements, geometry, and ground truth
    - Series of probe visualization images for geometric validation
    - Comprehensive metadata for reproducibility and analysis
    
    Technical Notes:
    - Uses different random seeds per phantom to ensure statistical independence
    - Implements progressive visualization (interactive for first 10 probes per phantom)
    - Applies consistent naming conventions for systematic dataset organization
    - Includes comprehensive error handling and progress reporting
    """
    
    # STEP 1: Initialize output directory structure
    # Create data directory in parent mah422 folder, not in code subfolder
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Configure logging for this pipeline run - save in parent mah422 directory
    log_file = "../logging.log"
    logger = setup_logging(level=logging.INFO, log_file=log_file)  # Change to DEBUG for verbose output
    
    logger.info("="*80)
    logger.info("STARTING NIR PHANTOM DATASET GENERATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Output directory: {os.path.abspath(data_dir)}")
    logger.info(f"Log file: {log_file}")

    # STEP 2: Generate multiple phantoms for dataset diversity
    n_phantoms = 10  # Full production dataset for ML training
    logger.info(f"Generating {n_phantoms} phantom datasets for machine learning training")
    
    for i in range(n_phantoms):
        phantom_start_time = time.time()
        logger.info("="*60)
        logger.info(f"GENERATING PHANTOM {i+1:02d}/{n_phantoms}")
        logger.info("="*60)
        
        # Create phantom-specific output directory
        phantom_dir = os.path.join(data_dir, f"phantom_{i+1:02d}")
        os.makedirs(phantom_dir, exist_ok=True)
        logger.debug(f"Created phantom directory: {phantom_dir}")

        # STEP 2.1: Construct phantom geometry with controlled randomization
        # Use different seeds to ensure statistical independence between phantoms
        logger.info("Step 1/6: Constructing phantom geometry")
        vol = build_phantom_with_tissue_and_tumours(rng_seed=44+i)
        logger.info(f"Phantom geometry completed: {vol.shape} voxels")

        # STEP 2.2: Generate finite element mesh for numerical simulation
        logger.info("Step 2/6: Generating finite element mesh")
        ele, nodes = mesh_volume(vol)
        mesh = create_stndmesh(ele, nodes)
        logger.info("FEM mesh generation completed")

        # STEP 2.3: Assign optical properties and generate ground truth maps
        logger.info("Step 3/6: Assigning optical properties")
        mesh, gt = assign_optical_properties(mesh, vol, rng_seed=42+i)
        logger.info("Optical properties assignment completed")

        # STEP 2.4: Extract tissue surface and generate probe configurations
        logger.info("Step 4/6: Generating probe layout")
        surface_coords = extract_surface_voxels(vol)
        srcs, dets, link = build_random_probe_layout(surface_coords, n_probes=256, rng_seed=123+i)
        logger.info(f"Probe layout completed: {len(srcs)} probes, {len(link)} measurements")

        # STEP 2.5: Generate probe visualizations for quality assurance
        logger.info("Step 5/6: Generating probe visualizations")
        vis_start_time = time.time()
        
        # Only generate visualization for the first probe to save time
        # All 256 probes will still be simulated and saved to HDF5
        if len(srcs) > 0:
            this_src = srcs[0]
            this_dets = dets[0:3]  # First 3 detectors for first probe
            
            # Show both interactive 3D and save PNG for first probe only
            visualize_probe_on_mesh(vol, mesh, this_src, this_dets, 0, phantom_dir, 
                                   show_interactive=True)
            logger.info(f"Generated visualization for probe 1/{len(srcs)} (remaining {len(srcs)-1} probes will be simulated without visualization)")
        
        vis_time = time.time() - vis_start_time
        logger.info(f"Visualization completed: 1 image in {vis_time:.1f}s (skipped {len(srcs)-1} images to save time)")

        # STEP 2.6: Execute frequency-domain simulation and save complete dataset
        logger.info("Step 6/6: Running frequency-domain simulation")
        h5_path = os.path.join(phantom_dir, "scan.h5")
        run_fd_and_save(mesh, gt, srcs, dets, link, h5_filename=h5_path)
        
        phantom_time = time.time() - phantom_start_time
        logger.info(f"PHANTOM {i+1:02d} COMPLETED in {phantom_time:.1f}s")
        logger.info(f"Dataset saved: {h5_path}")
        logger.info(f"Sample visualization: {phantom_dir}/probe_001.png")

    # STEP 3: Final validation and summary
    total_time = time.time() - phantom_start_time if 'phantom_start_time' in locals() else 0
    logger.info("="*80)
    logger.info("BATCH GENERATION COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Generated {n_phantoms} complete phantom datasets")
    logger.info(f"Total processing time: {total_time:.1f}s")
    logger.info(f"Output location: {os.path.abspath(data_dir)}")
    logger.info("Dataset contents per phantom:")
    logger.info("  • HDF5 file with measurements, geometry, and ground truth")
    logger.info("  • Sample probe visualization (probe_001.png) for geometric validation")
    logger.info("  • Complete metadata for reproducibility")
    logger.info("All phantoms generated and saved successfully!")

if __name__ == "__main__":
    main()
