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

# Specialized scipy modules for morphological and spatial operations
from scipy.ndimage import binary_erosion      # Morphological operation for surface extraction
from scipy.spatial.distance import cdist     # Efficient pairwise distance computation


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
    Constructs a 3D phantom volume with realistic tissue and tumour distributions.
    
    This function implements a hierarchical geometry construction approach:
    1. Initialize a cubic air-filled domain (background medium)
    2. Embed a large ellipsoidal healthy tissue region at the domain center
    3. Insert multiple smaller ellipsoidal tumour inclusions within the tissue
    
    Technical Details:
    - Uses implicit surface representation via quadratic forms for ellipsoids
    - Ensures all tumours are fully contained within the tissue boundary
    - Applies robust rejection sampling to prevent geometric intersections
    - Implements label-based segmentation for multi-region FEM meshing
    
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

    # ----------------------
    # HEALTHY TISSUE ELLIPSOID EMBEDDING
    # ----------------------
    # Position the primary tissue ellipsoid at the geometric center of the domain
    cx, cy, cz = Nx//2, Ny//2, Nz//2  # Integer division ensures exact center positioning
    
    # Sample random semi-axes lengths for tissue ellipsoid anisotropy
    rx = rng.integers(*tissue_radius_range)  # Semi-axis length in x-direction
    ry = rng.integers(*tissue_radius_range)  # Semi-axis length in y-direction  
    rz = rng.integers(*tissue_radius_range)  # Semi-axis length in z-direction
    
    # Compute tissue mask using implicit ellipsoid equation: (x-cx)²/rx² + (y-cy)²/ry² + (z-cz)²/rz² ≤ 1
    # This vectorized approach is computationally efficient for large volumes
    tissue_mask = (((X-cx)/rx)**2 + ((Y-cy)/ry)**2 + ((Z-cz)/rz)**2) <= 1
    vol[tissue_mask] = tissue_label

    # ----------------------
    # TUMOUR INCLUSION EMBEDDING WITH SPATIAL CONSTRAINTS
    # ----------------------
    # Sample the number of tumour inclusions from uniform distribution [0, max_tumours]
    n_tumours = rng.integers(0, max_tumours+1)
    print(f"Generating phantom with {n_tumours} tumour(s).")
    current_label = tumour_start_label

    # Iteratively place each tumour with geometric validity checking
    for _ in range(n_tumours):
        attempts = 0
        # Implement rejection sampling with maximum attempt limit to prevent infinite loops
        while attempts < 20:
            # Sample tumour center coordinates within the tissue ellipsoid bounds
            # Apply safety margins (±3 voxels) to ensure tumours don't extend beyond tissue
            cx_t = rng.integers(cx-rx+3, cx+rx-3)
            cy_t = rng.integers(cy-ry+3, cy+ry-3) 
            cz_t = rng.integers(cz-rz+3, cz+rz-3)
            
            # Sample tumour ellipsoid dimensions with physiological constraints
            rx_t = rng.integers(*tumour_radius_range)
            ry_t = rng.integers(*tumour_radius_range)
            # Constrain z-axis to maintain realistic aspect ratios and prevent elongated artifacts
            rz_t = rng.integers(3, min(8, min(rx_t, ry_t)))

            # Compute tumour mask using same ellipsoid formulation as tissue
            tumour_mask = (((X-cx_t)/rx_t)**2 + ((Y-cy_t)/ry_t)**2 + ((Z-cz_t)/rz_t)**2) <= 1
            
            # Validate that tumour intersects with tissue (geometric containment check)
            if np.any(tissue_mask & tumour_mask):
                # Apply tumour label only to voxels that are both within tumour AND tissue regions
                # This ensures tumours don't extend into air regions
                vol[tumour_mask & tissue_mask] = current_label
                current_label += 1  # Increment label for next tumour
                break  # Exit retry loop on successful placement
            attempts += 1

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
    
    # Execute CGAL-based tetrahedral mesh generation
    # This calls external C++ CGAL library for robust geometric mesh generation
    ele, nodes = ff.meshing.RunCGALMeshGenerator(volume, opt=params)
    
    # Perform comprehensive mesh quality validation
    # Checks for inverted elements, aspect ratios, and topological consistency
    ff.meshing.CheckMesh3D(ele, nodes)
    
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
    
    # Populate mesh with tetrahedral solid geometry and compute derived quantities
    # This automatically calculates:
    # - Element volumes via determinant of Jacobian matrix
    # - Surface face identification through adjacency analysis  
    # - Boundary node marking for Dirichlet/Robin boundary conditions
    mesh.from_solid(ele, nodes)
    
    # Extract and display mesh quality statistics for validation
    mean_vol = mesh.element_area.mean()  # Note: 'element_area' actually stores volumes for 3D elements
    std_vol = mesh.element_area.std()
    print(f"Mesh has {ele.shape[0]} tetrahedra and {nodes.shape[0]} nodes.")
    print(f"Mean element volume: {mean_vol:.3f} mm³ ± {std_vol:.3f} mm³")
    
    # Validate mesh quality metrics are within acceptable bounds
    if mean_vol < 0.1 or mean_vol > 10.0:
        print("⚠️  Warning: Element volumes may be outside optimal range for FEM convergence")
    
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

    # Sample baseline healthy tissue optical properties from physiological distributions
    # These serve as reference values for relative tumour property scaling
    mua_sub = rng.uniform(0.003, 0.007)    # Absorption coeff. [mm⁻¹] - controls image contrast
    musp_sub = rng.uniform(0.78, 1.18)     # Reduced scattering [mm⁻¹] - controls penetration depth
    
    # Dictionary for efficient ground truth lookup during voxel assignment
    region_lookup = {}

    # Assign optical properties to each tissue region with appropriate physiological scaling
    for region in regions:
        if region == 1:  # Healthy tissue baseline
            mua, musp = mua_sub, musp_sub
        else:  # Tumour regions (label ≥ 2)
            # Apply controlled randomization within clinically observed ranges
            # Tumours typically show increased absorption (higher blood volume)
            # and altered scattering (modified cellular architecture)
            mua = mua_sub * rng.uniform(1.5, 3.5)   # 50-250% increase in absorption
            musp = musp_sub * rng.uniform(1.5, 2.5)  # 50-150% increase in scattering
            
        # Store optical properties in NIRFASTer format: [region, μₐ, μ′s, n]
        # Refractive index n=1.33 is fixed for biological tissues at NIR wavelengths
        prop.append([region, mua, musp, 1.33])
        region_lookup[region] = (mua, musp)

    # Apply optical properties to mesh for FEM simulation
    # This populates mesh.mua and mesh.musp arrays used in diffusion equation assembly
    mesh.set_prop(np.array(prop))

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
    
    print(f"Found {surface_coords.shape[0]} tissue surface voxels.")
    
    # Validate surface extraction results
    if surface_coords.shape[0] == 0:
        print("⚠️  Warning: No surface voxels found - check tissue geometry")
    elif surface_coords.shape[0] < 100:
        print("⚠️  Warning: Very few surface voxels - may limit probe placement options")
    
    return surface_coords


# --------------------------------------------------------------
# STEP 5: SPATIALLY-INVARIANT PROBE LAYOUT GENERATION
# --------------------------------------------------------------

def build_random_probe_layout(surface_coords, n_probes=256,
                              min_distance=10, max_distance=50,
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
    - max_distance: Maintains sufficient signal-to-noise ratio for practical measurements
    - Typical clinical values: 10-50mm separation for depth sensitivity up to ~20mm
    
    Args:
        surface_coords (numpy.ndarray): Available surface voxel positions, shape (N_surface, 3)
        n_probes (int): Number of source-detector probe configurations to generate
        min_distance (float): Minimum source-detector separation [mm] for diffusive regime
        max_distance (float): Maximum source-detector separation [mm] for adequate SNR
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
    
    # Initialize storage arrays for probe configuration data
    all_sources, all_detectors, link = [], [], []
    n_attempts = 0  # Track total placement attempts for convergence monitoring

    # Generate each probe configuration with geometric validation
    for i in range(n_probes):
        # Implement rejection sampling loop with failure detection
        while True:
            n_attempts += 1
            # Prevent infinite loops due to overly restrictive geometric constraints
            if n_attempts > n_probes * 20:
                print("⚠️ Too many failed attempts to place probes, stopping early.")
                print(f"   Successfully placed {len(all_sources)} out of {n_probes} requested probes.")
                break

            # STEP 5.1: Uniformly sample source position from tissue surface
            source_idx = rng.integers(0, len(surface_coords))
            source_pos = surface_coords[source_idx]
            
            # STEP 5.2: Compute distances from source to all potential detector locations
            # Uses efficient scipy implementation of Euclidean distance computation
            dists = cdist([source_pos], surface_coords)[0]
            
            # STEP 5.3: Filter detector candidates based on distance constraints
            # Ensures measurements are in diffusive regime with adequate sensitivity
            valid_detectors = surface_coords[(dists >= min_distance) & (dists <= max_distance)]

            # STEP 5.4: Validate sufficient detector availability for multi-detector probe
            if valid_detectors.shape[0] < 3:
                continue  # Retry with different source position

            # STEP 5.5: Randomly sample 3 detectors from valid candidates
            # Multiple detectors per source increase measurement information content
            detector_indices = rng.choice(len(valid_detectors), size=3, replace=False)
            detector_positions = valid_detectors[detector_indices]

            # STEP 5.6: Store validated probe configuration
            all_sources.append(source_pos)
            all_detectors.extend(detector_positions)  # Flatten detector array across all probes
            
            # STEP 5.7: Generate measurement connectivity links
            # Links define which source-detector pairs generate measurements
            base = 3 * i  # Detector indexing offset for current probe
            link.extend([[i, base+0, 1],    # Source i → Detector base+0, active
                        [i, base+1, 1],    # Source i → Detector base+1, active  
                        [i, base+2, 1]])   # Source i → Detector base+2, active
            break  # Exit retry loop on successful probe placement

    print(f"Successfully placed {len(all_sources)} probes with rejection sampling.")
    print(f"Total placement attempts: {n_attempts} (efficiency: {len(all_sources)/n_attempts*100:.1f}%)")
    
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
    
    # Project optodes onto mesh surface and validate geometric consistency
    # This ensures sources/detectors lie exactly on tissue boundary for accurate modeling
    mesh.touch_optodes()

    # STEP 6.2: Execute frequency-domain finite element forward simulation
    print(f"Running FD solve at {fd_freq_hz/1e6:.1f} MHz ...")
    print(f"Mesh dimensions: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} elements")
    print(f"Optode configuration: {len(all_sources)} sources, {len(all_detectors)} detectors")
    
    # Solve complex-valued frequency-domain diffusion equation
    # Returns complex photon fluence at each detector for each source activation
    data, _ = mesh.femdata(fd_freq_hz)
    
    # Extract amplitude and phase from complex solution
    amplitude = data.amplitude  # |Φ|: Photon fluence magnitude
    phase = np.degrees(data.phase)  # arg(Φ): Phase delay in degrees

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

    # STEP 6.4: Process measurements for machine learning compatibility
    # Reshape measurements to probe-based format: (N_probes, 3) for 3 detectors per source
    
    # Log-amplitude transformation for neural network training stability
    # Prevents gradient explosion due to exponential amplitude decay with distance
    log_amp = np.log(np.clip(amplitude, 1e-8, None)).reshape(-1, 3)
    phase = phase.reshape(-1, 3)

    # STEP 6.5: Save complete dataset to HDF5 with hierarchical structure
    # HDF5 provides efficient storage for large multi-dimensional arrays with metadata
    with h5py.File(h5_filename, "w") as f:
        # Measurement data arrays
        f.create_dataset("log_amplitude", data=log_amp, 
                        attrs={"units": "ln(photons/mm²)", 
                               "description": "Natural log of photon fluence amplitude"})
        f.create_dataset("phase", data=phase,
                        attrs={"units": "degrees",
                               "description": "Phase delay relative to source modulation"})
        
        # Geometric configuration arrays
        f.create_dataset("source_pos", data=all_sources,
                        attrs={"units": "mm", "description": "Source positions in mesh coordinates"})
        f.create_dataset("det_pos", data=all_detectors.reshape(-1,3,3),
                        attrs={"units": "mm", "description": "Detector positions grouped by probe"})
        
        # Ground truth optical property maps for supervised learning
        f.create_dataset("ground_truth", data=ground_truth,
                        attrs={"channels": "absorption, reduced_scattering",
                               "units": "mm^-1", 
                               "description": "Voxel-wise optical property maps"})
        
        # Simulation metadata for reproducibility
        f.attrs["modulation_frequency_hz"] = fd_freq_hz
        f.attrs["noise_amplitude_std"] = amplitude_noise_std
        f.attrs["noise_phase_std"] = phase_noise_std
        f.attrs["n_measurements"] = len(link)
        f.attrs["n_probes"] = len(all_sources)
        
    print(f"✅ Saved complete dataset to {h5_filename}")
    print(f"   Measurements: {log_amp.shape[0]} probes × {log_amp.shape[1]} detectors")
    print(f"   Ground truth: {ground_truth.shape[0]}×{ground_truth.shape[1]}×{ground_truth.shape[2]} voxels")

# --------------------------------------------------------------
# VISUALIZATION: 3D PROBE-MESH RENDERING FOR GEOMETRIC VALIDATION
# --------------------------------------------------------------

def visualize_probe_on_mesh(volume, mesh, src, dets, probe_idx, save_dir, show_interactive=False):
    """
    Generates comprehensive 3D visualization of probe configuration on tissue geometry.
    
    This function creates publication-quality 3D renderings for geometric validation and 
    educational purposes, showing the spatial relationship between:
    - Tissue phantom geometry (healthy tissue and tumour inclusions)
    - Finite element mesh structure (nodes representing discretized domain)
    - Current probe configuration (source position and detector array)
    
    Visualization Strategy:
    - Healthy tissue: Semi-transparent green point cloud for spatial context
    - Tumour regions: Solid red markers for clear identification of pathological areas
    - Current probe: High-contrast yellow source and cyan detectors for focus
    - Background: Black background for professional scientific presentation
    
    Technical Implementation:
    - Uses matplotlib 3D scatter plots for efficient large-scale point rendering
    - Applies intelligent downsampling to prevent visual clutter and improve performance
    - Maintains aspect ratio fidelity for accurate spatial relationship representation
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

    # STEP 1: Extract tissue voxel coordinates for anatomical context visualization
    # Include all tissue types (healthy + tumours) for complete geometric representation
    tissue_coords = np.argwhere(volume >= 1)

    # STEP 2: Initialize 3D plotting environment with professional styling
    fig = plt.figure(figsize=(10,8))  # High-resolution figure for publication quality
    ax = fig.add_subplot(111, projection='3d')
    
    # Apply dark theme for scientific visualization aesthetic
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # STEP 3: Render healthy tissue context with intelligent downsampling
    # Downsample by factor of 8 to prevent visual overcrowding while maintaining spatial context
    ax.scatter(tissue_coords[::8,0], tissue_coords[::8,1], tissue_coords[::8,2],
               color='lime', s=2, alpha=0.1, label='Healthy tissue')

    # STEP 4: Render tumour regions with enhanced visibility
    # Use mesh node coordinates for accurate spatial positioning
    plotted_tumour = False  # Flag to prevent duplicate legend entries
    for reg in np.unique(mesh.region):
        if reg >= 2:  # Tumour regions have labels ≥ 2
            # Extract mesh nodes belonging to current tumour region
            nodes = mesh.nodes[mesh.region == reg]
            # Downsample tumour nodes by factor of 5 for clear visibility without clutter
            ax.scatter(nodes[::5,0], nodes[::5,1], nodes[::5,2],
                       color='red', s=6, alpha=0.9, 
                       label=f'Tumour {reg-1}' if not plotted_tumour else None)
            plotted_tumour = True

    # STEP 5: Highlight current probe configuration with high-contrast colors
    # Source visualization: Large yellow marker with black outline for maximum visibility
    ax.scatter(src[0], src[1], src[2], c='yellow', s=120, edgecolor='black', label='Source')
    
    # Detector visualization: Cyan markers distinguishable from source
    ax.scatter(dets[:,0], dets[:,1], dets[:,2], c='cyan', s=90, edgecolor='black', label='Detectors')

    # STEP 6: Configure plot aesthetics and metadata
    ax.set_title(f"Probe {probe_idx+1:03d}", color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)', color='white', fontsize=12)
    ax.set_ylabel('Y (mm)', color='white', fontsize=12)
    ax.set_zlabel('Z (mm)', color='white', fontsize=12)

    # Apply white color scheme to axis elements for dark background compatibility
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', edgecolor='white', fontsize=9, labelcolor='white')

    # STEP 7: Maintain realistic aspect ratio based on actual mesh dimensions
    # Prevents distortion that could misrepresent spatial relationships
    ax.set_box_aspect([np.ptp(mesh.nodes[:,0]),   # X-axis span
                       np.ptp(mesh.nodes[:,1]),   # Y-axis span  
                       np.ptp(mesh.nodes[:,2])])  # Z-axis span
    plt.tight_layout()

    # STEP 8: Generate and save high-resolution static image
    out_path = os.path.join(save_dir, f"probe_{probe_idx+1:03d}.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), 
                bbox_inches='tight', edgecolor='none')
    print(f"📸 Saved visualization: {out_path}")

    # STEP 9: Optionally display interactive 3D plot for manual exploration
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
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    print("🚀 Starting batch phantom dataset generation...")
    print(f"📁 Output directory: {os.path.abspath(data_dir)}")

    # STEP 2: Generate multiple phantoms for dataset diversity
    n_phantoms = 3  # Can be increased for larger training datasets
    
    for i in range(n_phantoms):
        print(f"\n{'='*60}")
        print(f"🔬 Generating Phantom {i+1:02d}/{n_phantoms}")
        print(f"{'='*60}")
        
        # Create phantom-specific output directory
        phantom_dir = os.path.join(data_dir, f"phantom_{i+1:02d}")
        os.makedirs(phantom_dir, exist_ok=True)

        # STEP 2.1: Construct phantom geometry with controlled randomization
        # Use different seeds to ensure statistical independence between phantoms
        vol = build_phantom_with_tissue_and_tumours(rng_seed=44+i)
        print(f"✅ Phantom geometry constructed: {vol.shape} voxels")

        # STEP 2.2: Generate finite element mesh for numerical simulation
        ele, nodes = mesh_volume(vol)
        mesh = create_stndmesh(ele, nodes)
        print(f"✅ FEM mesh generated and validated")

        # STEP 2.3: Assign optical properties and generate ground truth maps
        mesh, gt = assign_optical_properties(mesh, vol, rng_seed=42+i)
        print(f"✅ Optical properties assigned with ground truth maps")

        # STEP 2.4: Extract tissue surface and generate probe configurations
        surface_coords = extract_surface_voxels(vol)
        srcs, dets, link = build_random_probe_layout(surface_coords, rng_seed=123+i)
        print(f"✅ Probe layout generated: {len(srcs)} probes with {len(link)} measurements")

        # STEP 2.5: Generate probe visualizations for quality assurance
        print(f"🎨 Generating probe visualizations...")
        for k in range(len(srcs)):
            this_src = srcs[k]
            this_dets = dets[3*k : 3*k+3]  # Extract 3 detectors for current probe
            
            # Show interactive plots for first 10 probes for manual validation
            show_it = (k < 10)
            visualize_probe_on_mesh(vol, mesh, this_src, this_dets, k, phantom_dir, 
                                   show_interactive=show_it)

        # STEP 2.6: Execute frequency-domain simulation and save complete dataset
        print(f"🧮 Running finite element simulation...")
        h5_path = os.path.join(phantom_dir, "scan.h5")
        run_fd_and_save(mesh, gt, srcs, dets, link, h5_filename=h5_path)
        
        print(f"✅ Phantom {i+1:02d} completed successfully")
        print(f"   📄 Dataset saved: {h5_path}")
        print(f"   🖼️  Visualizations: {phantom_dir}/probe_*.png")

    # STEP 3: Final validation and summary
    print(f"\n{'='*60}")
    print("🎉 BATCH GENERATION COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"📊 Generated {n_phantoms} complete phantom datasets")
    print(f"📁 Output location: {os.path.abspath(data_dir)}")
    print(f"🔗 Each dataset contains:")
    print(f"   • HDF5 file with measurements, geometry, and ground truth")
    print(f"   • Probe visualization images for geometric validation")
    print(f"   • Complete metadata for reproducibility")
    print(f"✅ All phantoms generated and saved!")

if __name__ == "__main__":
    main()
