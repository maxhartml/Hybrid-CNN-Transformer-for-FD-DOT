#!/usr/bin/env python3
"""
Generate Missing Phantom 4716

This script generates a single phantom (phantom_4716) to complete the 5000-phantom dataset.
The phantom will be generated with a unique seed to ensure it's different from all other phantoms.

Usage:
    python generate_phantom_4716.py
"""

import sys
import os
from pathlib import Path
import time
import logging

# Add the project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the data simulator functions
from code.data_processing.data_simulator import (
    build_phantom_with_tissue_and_tumours,
    mesh_volume,
    create_stndmesh,
    assign_optical_properties,
    extract_surface_voxels,
    build_patch_based_probe_layout,
    run_fd_simulation_and_save,
    visualize_probe_on_mesh,
    setup_random_state,
    DEFAULT_N_VISUAL_PROBE_PAIRS,
    ENABLE_3D_INTERACTIVE_VISUAL,
    MAX_PHANTOM_RETRY_ATTEMPTS,
    RETRY_SEED_OFFSET
)
from code.utils.logging_config import get_data_logger, NIRDOTLogger

def generate_phantom_4716():
    """Generate the missing phantom_4716 with a unique seed."""
    
    # Setup logging
    logs_dir = project_root / "logs"
    NIRDOTLogger.setup_logging(log_dir=str(logs_dir))
    logger = get_data_logger(__name__)
    
    # Create data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("GENERATING MISSING PHANTOM 4716")
    logger.info("="*80)
    
    # Phantom-specific configuration
    phantom_idx = 4715  # 0-based index for phantom_4716 (since we use phantom_idx+1 in filenames)
    
    # Generate unique seed for phantom 4716 that's different from the original seed sequence
    # Use a distinct seed that wasn't in the original phantom generation
    # Original seeds were based on MASTER_RANDOM_SEED=42, we'll use a completely different base
    unique_base_seed = 999999  # Different from original MASTER_RANDOM_SEED=42
    phantom_seed = unique_base_seed + phantom_idx
    
    logger.info(f"🎲 Generating Phantom 4716 with unique seed: {phantom_seed}")
    logger.info(f"📊 This seed ensures uniqueness compared to other phantoms in dataset")
    
    # Create phantom directory
    phantom_dir = data_dir / f"phantom_{phantom_idx+1}"  # This will create phantom_4716
    phantom_dir.mkdir(exist_ok=True)
    logger.info(f"📂 Output directory: {phantom_dir}")
    
    # Check if phantom already exists
    h5_file = phantom_dir / f"phantom_{phantom_idx+1:03d}_scan.h5"
    if h5_file.exists():
        logger.warning(f"Phantom 4716 already exists at {h5_file}")
        response = input("Phantom 4716 already exists. Regenerate? (y/n): ")
        if response.lower() != 'y':
            logger.info("Skipping generation.")
            return
    
    # Retry loop for robust generation
    phantom_success = False
    retry_attempt = 0
    
    while not phantom_success and retry_attempt < MAX_PHANTOM_RETRY_ATTEMPTS:
        if retry_attempt > 0:
            logger.warning(f"Retrying phantom 4716 generation (attempt {retry_attempt+1}/{MAX_PHANTOM_RETRY_ATTEMPTS})")
            retry_seed = phantom_seed + (retry_attempt * RETRY_SEED_OFFSET)
            logger.info(f"🎲 Retry with new seed: {retry_seed}")
        else:
            retry_seed = phantom_seed
            
        phantom_rng, _ = setup_random_state(retry_seed)
        phantom_start_time = time.time()
        
        try:
            # Step 1: Geometry Construction
            logger.info("▶️  STEP 1/6: Constructing phantom geometry...")
            step1_start = time.time()
            phantom_volume = build_phantom_with_tissue_and_tumours(phantom_rng=phantom_rng)
            step1_time = time.time() - step1_start
            logger.info(f"✅ STEP 1/6 COMPLETED: Geometry construction ({step1_time:.2f}s)")
            
            # Step 2: Mesh Generation
            logger.info("▶️  STEP 2/6: Generating finite element mesh...")
            step2_start = time.time()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mesh_elements, mesh_nodes = mesh_volume(phantom_volume)
                phantom_mesh = create_stndmesh(mesh_elements, mesh_nodes)
            step2_time = time.time() - step2_start
            logger.info(f"✅ STEP 2/6 COMPLETED: FEM mesh generation ({step2_time:.2f}s)")
            
            # Step 3: Optical Properties
            logger.info("▶️  STEP 3/6: Assigning optical properties...")
            step3_start = time.time()
            phantom_mesh, ground_truth_maps = assign_optical_properties(phantom_mesh, phantom_volume, phantom_rng=phantom_rng)
            step3_time = time.time() - step3_start
            logger.info(f"✅ STEP 3/6 COMPLETED: Optical properties ({step3_time:.2f}s)")
            
            # Step 4: Surface & Probe Layout
            logger.info("▶️  STEP 4/6: Extracting surface and generating probe layout...")
            step4_start = time.time()
            surface_coordinates = extract_surface_voxels(phantom_volume)
            probe_sources, probe_detectors, measurement_links, patch_info = build_patch_based_probe_layout(
                surface_coordinates, n_sources=50, detectors_per_source=20, phantom_rng=phantom_rng)
            step4_time = time.time() - step4_start
            logger.info(f"✅ STEP 4/6 COMPLETED: Surface & probe layout ({step4_time:.2f}s)")
            
            # Step 5: Visualization (always generate for this special phantom)
            logger.info("▶️  STEP 5/6: Generating visualization...")
            step5_start = time.time()
            if len(probe_sources) > 0:
                visualize_probe_on_mesh(phantom_volume, phantom_mesh, probe_sources, probe_detectors, measurement_links, 
                                       0, str(phantom_dir), patch_info=patch_info, 
                                       show_interactive=ENABLE_3D_INTERACTIVE_VISUAL,
                                       n_visual_pairs=DEFAULT_N_VISUAL_PROBE_PAIRS)
                logger.info("✅ Generated probe layout visualization")
            step5_time = time.time() - step5_start
            logger.info(f"✅ STEP 5/6 COMPLETED: Visualization ({step5_time:.2f}s)")
            
            # Step 6: Simulation & Storage
            logger.info("▶️  STEP 6/6: Running FD simulation and saving data...")
            step6_start = time.time()
            h5_output_path = phantom_dir / f"phantom_{phantom_idx+1:03d}_scan.h5"
            
            simulation_success = run_fd_simulation_and_save(
                phantom_mesh, ground_truth_maps, probe_sources, probe_detectors, measurement_links, 
                phantom_volume=phantom_volume, output_h5_filename=str(h5_output_path))
            
            step6_time = time.time() - step6_start
            logger.info(f"✅ STEP 6/6 COMPLETED: FD simulation & storage ({step6_time:.2f}s)")
            
            if simulation_success:
                phantom_success = True
                total_time = time.time() - phantom_start_time
                
                logger.info("="*70)
                logger.info("🎉 PHANTOM 4716 GENERATED SUCCESSFULLY!")
                logger.info("="*70)
                logger.info("⏱️  TIMING BREAKDOWN:")
                logger.info(f"   Step 1 - Geometry:      {step1_time:6.2f}s")
                logger.info(f"   Step 2 - FEM Mesh:      {step2_time:6.2f}s")
                logger.info(f"   Step 3 - Optical Props: {step3_time:6.2f}s")
                logger.info(f"   Step 4 - Surface/Probe: {step4_time:6.2f}s")
                logger.info(f"   Step 5 - Visualization: {step5_time:6.2f}s")
                logger.info(f"   Step 6 - FD Simulation: {step6_time:6.2f}s")
                logger.info(f"   🏁 TOTAL TIME:          {total_time:6.2f}s")
                logger.info("="*70)
                
                # Verify files were created
                h5_file = phantom_dir / f"phantom_{phantom_idx+1:03d}_scan.h5"
                png_file = phantom_dir / f"phantom_{phantom_idx+1:03d}_probe_layout.png"
                
                if h5_file.exists():
                    file_size = h5_file.stat().st_size / (1024*1024)  # MB
                    logger.info(f"✅ HDF5 file created: {h5_file.name} ({file_size:.1f} MB)")
                else:
                    logger.error(f"❌ HDF5 file not found: {h5_file}")
                    
                if png_file.exists():
                    logger.info(f"✅ Visualization created: {png_file.name}")
                else:
                    logger.info("ℹ️  No visualization file (may be skipped)")
                
                logger.info("🎯 Phantom 4716 successfully added to dataset!")
                logger.info("🔄 Dataset now complete: 5000/5000 phantoms")
                
            else:
                logger.warning("Simulation failed due to NaN values, retrying...")
                retry_attempt += 1
                
        except Exception as e:
            logger.error(f"Error during phantom generation: {e}")
            retry_attempt += 1
            
    if not phantom_success:
        logger.error(f"❌ Failed to generate phantom 4716 after {MAX_PHANTOM_RETRY_ATTEMPTS} attempts")
        return False
        
    return True

if __name__ == "__main__":
    success = generate_phantom_4716()
    if success:
        print("✅ Phantom 4716 generated successfully!")
        print("🎯 Your dataset is now complete with 5000 phantoms")
    else:
        print("❌ Failed to generate phantom 4716")
        sys.exit(1)
