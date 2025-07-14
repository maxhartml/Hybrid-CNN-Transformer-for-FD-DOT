#!/usr/bin/env python3
"""
Generates a breast phantom with 0–5 randomly placed ellipsoidal tumours,
meshes it using NIRFASTer-FF, assigns distinct optical properties,
places a rigid probe (1 source + 3 detectors) and raster scans it
across the phantom, running FD simulations at each position.

Outputs:
- mesh visualisation showing tumours + first probe
- FD scan plots of amplitude & phase vs scan position
"""

import numpy as np
import nirfasterff as ff
import matplotlib.pyplot as plt
import h5py
import os


# --------------------------------------------------------------
# STEP 1 - Create breast volume with random ellipsoidal tumours
# --------------------------------------------------------------
def build_breast_volume_with_random_tumours(shape=(50, 50, 24),
                                            max_tumours=5,
                                            tumour_radius_range=(4, 10),
                                            background_label=1,
                                            tumour_start_label=2,
                                            rng_seed=None):
    """
    Creates a 3D phantom volume:
    - all voxels start as healthy tissue (label 1)
    - inserts 0–5 random ellipsoidal tumours with labels 2,3,4,...

    Now explicitly matches Robin Dale's experimental phantom volume dimensions.
    """
    Nx, Ny, Nz = shape
    vol = np.full(shape, background_label, dtype=np.uint8)

    rng = np.random.default_rng(rng_seed)
    n_tumours = rng.integers(0, max_tumours + 1)  # sample number of tumours (0 to max)

    print(f"Generating phantom with {n_tumours} tumour(s).")

    current_label = tumour_start_label
    # Precompute meshgrid for ellipsoid formula
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')

    for _ in range(n_tumours):
        # Random tumour centre, keep away from edges
        cx = rng.integers(10, Nx - 10)
        cy = rng.integers(10, Ny - 10)
        cz = rng.integers(5, Nz - 5)

        # Random ellipsoid radii
        rx = rng.integers(*tumour_radius_range)
        ry = rng.integers(*tumour_radius_range)
        rz = rng.integers(3, 8)

        # Create ellipsoid mask and update volume labels
        ellipsoid = (((X - cx)/rx)**2 + ((Y - cy)/ry)**2 + ((Z - cz)/rz)**2) <= 1
        vol[ellipsoid] = current_label
        current_label += 1

    return vol


# --------------------------------------------------------------
# STEP 2 - Mesh the volume into tetrahedra
# --------------------------------------------------------------
def mesh_volume(volume):
    """
    Converts the labelled voxel volume into a tetrahedral FEM mesh.

    Uses CGAL mesher via NIRFASTer-FF with uniform mesh density.

    Current density (general_cell_size=1.65) gives ~1.04 mm³ element
    volumes matching Robin's pipeline.
    """
    params = ff.utils.MeshingParams()
    params.general_cell_size = 1.65 # same density everywhere as per Robin's setup
   
    ele, nodes = ff.meshing.RunCGALMeshGenerator(volume, opt=params)
    ff.meshing.CheckMesh3D(ele, nodes)
    return ele, nodes


# --------------------------------------------------------------
# STEP 2.5 - Create stndmesh + print mean element volume
# --------------------------------------------------------------
def create_stndmesh(ele, nodes):
    """
    Wraps nodes & elements into a NIRFASTer standard mesh object
    and prints the mean/std volume to check against Robin's stats.
    """
    mesh = ff.base.stndmesh()
    mesh.from_solid(ele, nodes)

    mean_vol = mesh.element_area.mean()
    std_vol = mesh.element_area.std()

    print(f"Mesh has {ele.shape[0]} tetrahedra and {nodes.shape[0]} nodes.")
    print(f"Mean tetrahedron volume: {mean_vol:.3f} mm³ ± {std_vol:.3f} mm³")

    return mesh


# --------------------------------------------------------------
# STEP 3 - Set optical properties for each region AND build ground truth
# --------------------------------------------------------------
def assign_optical_properties(mesh, volume, rng_seed=None):
    """
    Assigns μₐ and μ′s to each region in the mesh using Robin's exact Table 5.1 style:
    - Sample substrate μₐ ~ U[0.003,0.007]
    - Sample substrate μ′s ~ U[0.78,1.18]
    - Then for each tumour region, multiply by contrast:
        μₐ_tumour = μₐ_substrate × U[1.5,3.5]
        μ′s_tumour = μ′s_substrate × U[1.5,2.5]
    Also builds a (Nx,Ny,Nz,2) ground truth grid with [μₐ, μ′s].
    """
    rng = np.random.default_rng(rng_seed)
    regions = np.unique(mesh.region)
    prop = []

    # Sample substrate optical properties
    mua_sub = rng.uniform(0.003, 0.007)
    musp_sub = rng.uniform(0.78, 1.18)

    region_lookup = {}

    for region in regions:
        if region == 1:
            mua = mua_sub
            musp = musp_sub
        else:
            mua = mua_sub * rng.uniform(1.5, 3.5)
            musp = musp_sub * rng.uniform(1.5, 2.5)
        prop.append([region, mua, musp, 1.33])
        region_lookup[region] = (mua, musp)

    prop = np.array(prop)
    mesh.set_prop(prop)
    print("Optical properties set:\n", prop)

    # Build ground truth grid
    Nx, Ny, Nz = volume.shape
    ground_truth_grid = np.zeros((Nx, Ny, Nz, 2))
    for region, (mua, musp) in region_lookup.items():
        mask = (volume == region)
        ground_truth_grid[mask, 0] = mua
        ground_truth_grid[mask, 1] = musp

    return mesh, ground_truth_grid


# --------------------------------------------------------------
# STEP 4 - Build rigid probe geometry
# --------------------------------------------------------------
def make_probe_line(source_xy, offsets=(20, 30, 40)):
    """
    Returns coordinates for a probe:
    - 1 source at source_xy
    - 3 detectors offset along +x by 20, 30, 40 mm
    """
    sx, sy = source_xy
    src = np.array([[sx, sy, 0]])
    det = np.array([[sx + d, sy, 0] for d in offsets])
    return src, det

# --------------------------------------------------------------
# STEP 5 - Visualisation: mesh + one probe
# --------------------------------------------------------------
def visualize_mesh_with_probe(mesh, src, det, save_as="mesh_with_probe.png"):
    """
    Uses matplotlib 3D scatter to show:
    - sparse green dots for healthy tissue nodes
    - dense red dots for tumour nodes
    - optode positions overlaid
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    regions = np.unique(mesh.region)

    for reg in regions:
        nodes = mesh.nodes[mesh.region == reg]
        if reg == 1:
            ax.scatter(nodes[::30,0], nodes[::30,1], nodes[::30,2],
                       color='green', s=1, alpha=0.5, label='Healthy tissue')
        else:
            ax.scatter(nodes[::5,0], nodes[::5,1], nodes[::5,2],
                       color='red', s=6, alpha=0.8, label=f'Tumour {reg-1}')

    ax.scatter(src[:,0], src[:,1], src[:,2], c='yellow', s=80, edgecolor='black', label='Source')
    ax.scatter(det[:,0], det[:,1], det[:,2], c='blue', s=80, edgecolor='black', label='Detectors')
    ax.set_title("Mesh with example probe position")
    ax.legend()
    ax.set_box_aspect([np.ptp(mesh.nodes[:,0]),
                       np.ptp(mesh.nodes[:,1]),
                       np.ptp(mesh.nodes[:,2])])
    plt.savefig(save_as, dpi=200)
    plt.show()
    print(f"Saved mesh figure to {save_as}")

# --------------------------------------------------------------
# STEP 6 - Randomised probe positions (256) + parallel solve
# --------------------------------------------------------------
def random_probe_scan_fd_and_save(mesh, 
                                  ground_truth,
                                  x_bounds, y_bounds,
                                  n_probes=256,
                                  fd_freq_hz=140e6,
                                  h5_filename="phantom_fd_scan.h5",
                                  rng_seed=123):
    """
    Places 256 random rigid probes across the phantom surface, then runs a *single*
    frequency-domain FEM solve to get data from all probes simultaneously.

    Stores amplitude, phase, all probe geometries, and ground truth into HDF5.
    """
    rng = np.random.default_rng(rng_seed)
    probe_positions = rng.integers([x_bounds[0], y_bounds[0]],
                                   [x_bounds[1]+1, y_bounds[1]+1],
                                   size=(n_probes,2))
    print(f"Generated {n_probes} random probe positions.")

    all_sources = []
    all_detectors = []
    link = []

    # Build all sources, detectors, and link rows
    for i, (sx, sy) in enumerate(probe_positions):
        src, det = make_probe_line((sx, sy))
        all_sources.append(src[0])
        all_detectors.extend(det)
        # Each detector row links source_idx, detector_idx, active=1
        base_det_idx = 3*i
        link.extend([[i, base_det_idx+0, 1],
                     [i, base_det_idx+1, 1],
                     [i, base_det_idx+2, 1]])

    # Place all optodes
    mesh.source = ff.base.optode(np.array(all_sources))
    mesh.meas   = ff.base.optode(np.array(all_detectors))
    mesh.link   = np.array(link)
    mesh.touch_optodes()

    # Solve once for all probes
    print(f"Running FD simulation at {fd_freq_hz/1e6:.1f} MHz ...")
    data, info = mesh.femdata(fd_freq_hz)
    amplitude = data.amplitude  # shape: (n_channels,)
    phase = np.degrees(data.phase)

    # Add noise to amplitude and phase
    amp_noise_std = 0.02 * np.mean(amplitude)  # 2% relative noise
    pha_noise_std = 2.0                        # 2 degree phase noise
    amplitude += rng.normal(0, amp_noise_std, amplitude.shape)
    phase += rng.normal(0, pha_noise_std, phase.shape)

    # Take the natural logarithm of amplitude
    # Use np.clip to avoid log(0) issues
    log_amplitude = np.log(np.clip(amplitude, 1e-8, None))

    # Reshape into (n_probes,3) because each probe has 3 detectors
    log_amplitude = log_amplitude.reshape(n_probes,3)
    phase = phase.reshape(n_probes,3)

    # Save everything
    with h5py.File(h5_filename, "w") as f:
        f.create_dataset("log_amplitude", data=log_amplitude)
        f.create_dataset("phase",     data=phase)
        f.create_dataset("source_pos", data=np.array(all_sources))
        f.create_dataset("det_pos",    data=np.array(all_detectors).reshape(n_probes,3,3))
        f.create_dataset("ground_truth", data=ground_truth)
    print(f"Finished scan. Data saved to {h5_filename}")


# --------------------------------------------------------------
# Main program orchestrating the entire pipeline
# --------------------------------------------------------------
def main():
    N_PHANTOMS = 3  # Number of phantoms to generate and scan

    # Make a top-level data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for i in range(N_PHANTOMS):
        phantom_dir = os.path.join(data_dir, f"phantom_{i+1:02d}")
        os.makedirs(phantom_dir, exist_ok=True)
        print(f"\n=== Generating phantom {i+1}/{N_PHANTOMS} ===")

        # 1. Build volume & mesh
        vol = build_breast_volume_with_random_tumours(rng_seed=44 + i)
        ele, nodes = mesh_volume(vol)
        mesh = create_stndmesh(ele, nodes)
        mesh, ground_truth = assign_optical_properties(mesh, vol, rng_seed=42 + i)

        # 2. Visualise & save inside this phantom's folder
        src, det = make_probe_line((10,30))
        img_path = os.path.join(phantom_dir, f"phantom_{i+1}_visualisation.png")
        visualize_mesh_with_probe(mesh, src, det, save_as=img_path)

        # 3. Perform scan and save HDF5 in same folder
        h5_path = os.path.join(phantom_dir, f"phantom_{i+1}_fd_scan.h5")
        random_probe_scan_fd_and_save(mesh, ground_truth,
                                      x_bounds=(10,30), y_bounds=(10,40),
                                      h5_filename=h5_path)

    print("✅ All phantoms done. Data saved neatly in 'data/' directory.")


if __name__ == "__main__":
    main()