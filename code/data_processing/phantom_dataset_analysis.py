#!/usr/bin/env python3
"""
🧬 PHANTOM DATASET COMPREHENSIVE ANALYSIS 🧬
===========================================

Generates a comprehensive 2×2 analysis figure of the NIR phantom dataset:
1. Tumour sizes/counts distribution
2. Tissue optical properties (μa, μs′) 
3. Source-detector separation distances
4. Probe layout coverage visualization

Author: Max Hart
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import logging
from collections import defaultdict
from scipy.spatial.distance import cdist
import gc  # For garbage collection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configuration
DATA_DIR = Path("data")
VOXEL_SIZE_MM = 1.0  # 1mm voxel resolution
CLINICAL_SDS_TARGET = 30.0  # Clinical sweet spot for SDS [mm]

# Tissue labels from data simulator
AIR_LABEL = 0
HEALTHY_TISSUE_LABEL = 1
TUMOR_START_LABEL = 2

# Phantom dimensions and optical property ranges (from data simulator)
PHANTOM_SHAPE = (64, 64, 64)
HEALTHY_MUA_RANGE = (0.003, 0.007)
HEALTHY_MUSP_RANGE = (0.78, 1.18)
TUMOR_RADIUS_RANGE = (5, 15)  # mm

def load_phantom_data(phantom_file):
    """Load and parse a single phantom HDF5 file."""
    try:
        with h5py.File(phantom_file, 'r') as f:
            # Load measurement data
            log_amplitude = f['log_amplitude'][:]
            phase = f['phase'][:]
            source_positions = f['source_positions'][:]
            detector_positions = f['detector_positions'][:]
            
            # Load ground truth volumes (μa, μs′)
            ground_truth = f['ground_truth'][:]  # Shape: (2, 64, 64, 64)
            tissue_labels = f['tissue_labels'][:]  # Shape: (64, 64, 64)
            
            return {
                'log_amplitude': log_amplitude,
                'phase': phase,
                'source_positions': source_positions,
                'detector_positions': detector_positions,
                'ground_truth': ground_truth,
                'tissue_labels': tissue_labels,
                'phantom_file': phantom_file
            }
    except Exception as e:
        logger.warning(f"Failed to load {phantom_file}: {e}")
        return None

def analyze_tumor_geometry(tissue_labels):
    """
    Extract tumor count information from tissue label volume.
    
    Returns:
        dict: Contains tumor_count and tumor_volumes (in voxels)
    """
    # Find unique tumor labels (labels >= 2)
    unique_labels = np.unique(tissue_labels)
    tumor_labels = unique_labels[unique_labels >= TUMOR_START_LABEL]
    
    tumor_count = len(tumor_labels)
    tumor_volumes = []
    
    for tumor_label in tumor_labels:
        # Count voxels for this tumor
        tumor_mask = tissue_labels == tumor_label
        tumor_volume_voxels = np.sum(tumor_mask)
        tumor_volumes.append(tumor_volume_voxels)
    
    return {
        'tumor_count': tumor_count,
        'tumor_volumes': tumor_volumes  # Keep as voxel counts
    }

def extract_optical_properties(ground_truth, tissue_labels):
    """
    Extract optical properties for healthy tissue and tumors separately.
    
    Returns:
        dict: Contains healthy_mua, healthy_musp, tumor_mua, tumor_musp
    """
    mua_volume = ground_truth[0]  # Absorption coefficient
    musp_volume = ground_truth[1]  # Reduced scattering coefficient
    
    # Extract healthy tissue properties
    healthy_mask = tissue_labels == HEALTHY_TISSUE_LABEL
    healthy_mua = mua_volume[healthy_mask]
    healthy_musp = musp_volume[healthy_mask]
    
    # Extract tumor properties (all labels >= 2)
    tumor_mask = tissue_labels >= TUMOR_START_LABEL
    tumor_mua = mua_volume[tumor_mask]
    tumor_musp = musp_volume[tumor_mask]
    
    return {
        'healthy_mua': healthy_mua,
        'healthy_musp': healthy_musp,
        'tumor_mua': tumor_mua,
        'tumor_musp': tumor_musp
    }

def calculate_source_detector_distances(source_positions, detector_positions):
    """Calculate all source-detector separation distances."""
    # Calculate pairwise distances between all sources and detectors
    distances = []
    for i in range(len(source_positions)):
        src_pos = source_positions[i:i+1]  # Keep as 2D for cdist
        det_pos = detector_positions[i:i+1]
        distance = cdist(src_pos, det_pos, metric='euclidean')[0, 0]
        distances.append(distance)
    
    return np.array(distances)

def analyze_dataset(max_phantoms=500):
    """Load and analyze a random subsample of the phantom dataset."""
    logger.info("Starting comprehensive dataset analysis...")
    
    # Find all phantom files
    phantom_files = sorted(list(DATA_DIR.glob("phantom_*/phantom_*_scan.h5")))
    logger.info(f"Found {len(phantom_files)} phantom files")
    
    # Randomly subsample to prevent memory issues
    if len(phantom_files) > max_phantoms:
        # Take random samples from the full dataset
        np.random.seed(42)  # For reproducible results
        indices = np.random.choice(len(phantom_files), max_phantoms, replace=False)
        phantom_files = [phantom_files[i] for i in indices]
        logger.info(f"Randomly subsampled to {len(phantom_files)} phantoms to prevent memory issues")
    
    if len(phantom_files) == 0:
        raise FileNotFoundError(f"No phantom files found in {DATA_DIR}")
    
    # Storage for analysis results
    all_tumor_counts = []
    all_tumor_volumes = []
    all_healthy_mua = []
    all_healthy_musp = []
    all_tumor_mua = []
    all_tumor_musp = []
    all_sds_distances = []
    all_source_positions = []
    all_detector_positions = []
    
    # Analyze each phantom
    successful_loads = 0
    for i, phantom_file in enumerate(phantom_files):
        if i % 100 == 0:
            logger.info(f"Processing phantom {i+1}/{len(phantom_files)} (Memory efficient mode)")
            # Periodic garbage collection to prevent memory buildup
            gc.collect()
        
        phantom_data = load_phantom_data(phantom_file)
        if phantom_data is None:
            continue
        
        successful_loads += 1
        
        # Analyze tumor geometry
        tumor_analysis = analyze_tumor_geometry(phantom_data['tissue_labels'])
        all_tumor_counts.append(tumor_analysis['tumor_count'])
        all_tumor_volumes.extend(tumor_analysis['tumor_volumes'])
        
        # Extract optical properties (SAMPLE to prevent memory overload)
        optical_props = extract_optical_properties(
            phantom_data['ground_truth'], 
            phantom_data['tissue_labels']
        )
        
        # Sample optical properties to reduce memory usage
        max_samples_per_type = 1000  # Limit samples per phantom
        if len(optical_props['healthy_mua']) > max_samples_per_type:
            indices = np.random.choice(len(optical_props['healthy_mua']), max_samples_per_type, replace=False)
            all_healthy_mua.extend(optical_props['healthy_mua'][indices])
            all_healthy_musp.extend(optical_props['healthy_musp'][indices])
        else:
            all_healthy_mua.extend(optical_props['healthy_mua'])
            all_healthy_musp.extend(optical_props['healthy_musp'])
        
        if len(optical_props['tumor_mua']) > max_samples_per_type:
            indices = np.random.choice(len(optical_props['tumor_mua']), max_samples_per_type, replace=False)
            all_tumor_mua.extend(optical_props['tumor_mua'][indices])
            all_tumor_musp.extend(optical_props['tumor_musp'][indices])
        else:
            all_tumor_mua.extend(optical_props['tumor_mua'])
            all_tumor_musp.extend(optical_props['tumor_musp'])
        
        # Calculate source-detector distances
        sds_distances = calculate_source_detector_distances(
            phantom_data['source_positions'],
            phantom_data['detector_positions']
        )
        all_sds_distances.extend(sds_distances)
        
        # Collect probe positions for coverage analysis (sample to reduce memory)
        # Only collect positions from a subset of phantoms to prevent memory overload
        if len(all_source_positions) < 10000:  # Limit total positions
            all_source_positions.extend(phantom_data['source_positions'])
            all_detector_positions.extend(phantom_data['detector_positions'])
    
    logger.info(f"Successfully analyzed {successful_loads} phantoms")
    
    # Convert to numpy arrays for analysis
    results = {
        'tumor_counts': np.array(all_tumor_counts),
        'tumor_volumes': np.array(all_tumor_volumes),
        'healthy_mua': np.array(all_healthy_mua),
        'healthy_musp': np.array(all_healthy_musp),
        'tumor_mua': np.array(all_tumor_mua),
        'tumor_musp': np.array(all_tumor_musp),
        'sds_distances': np.array(all_sds_distances),
        'source_positions': np.array(all_source_positions),
        'detector_positions': np.array(all_detector_positions),
        'n_phantoms': successful_loads
    }
    
    return results

def create_comprehensive_figure(results):
    """Generate the comprehensive 2×2 analysis figure."""
    # Set up the figure with more spacing between subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Create title with phantom count information
    phantom_count = results['n_phantoms']
    title = f'NIR Phantom Dataset Comprehensive Analysis (Subsample: {phantom_count} phantoms)'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Color scheme
    healthy_color = '#2E8B57'  # Sea green
    tumor_color = '#DC143C'    # Crimson
    source_color = '#FF6B6B'   # Light red
    detector_color = '#4ECDC4' # Teal
    
    # =========================================================================
    # Panel 1: Tumor Count Distribution
    # =========================================================================
    ax1 = axes[0, 0]
    
    if len(results['tumor_counts']) > 0:
        # Create histogram of tumor counts per phantom
        unique_counts, count_frequencies = np.unique(results['tumor_counts'], return_counts=True)
        
        # Create bar plot
        bars = ax1.bar(unique_counts, count_frequencies, width=0.8, 
                      alpha=0.7, color=tumor_color, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Number of Tumors per Phantom', fontsize=12)
        ax1.set_ylabel('Number of Phantoms', fontsize=12)
        ax1.set_title('Tumor Count Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(unique_counts)
        
        # Add statistics text
        mean_count = np.mean(results['tumor_counts'])
        median_count = np.median(results['tumor_counts'])
        max_count = np.max(results['tumor_counts'])
        min_count = np.min(results['tumor_counts'])
        ax1.text(0.05, 0.95, f'Mean: {mean_count:.1f} tumors\nMedian: {median_count:.1f} tumors\nRange: {min_count}-{max_count} tumors', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No tumor data available', transform=ax1.transAxes, 
                horizontalalignment='center', verticalalignment='center')
    
    # =========================================================================
    # Panel 2: Optical Properties (Split into two subplots for different scales)
    # =========================================================================
    ax2 = axes[0, 1]
    
    # Create two separate plots for μₐ and μₛ' due to different scales
    if (len(results['healthy_mua']) > 0 and len(results['tumor_mua']) > 0 and 
        len(results['healthy_musp']) > 0 and len(results['tumor_musp']) > 0):
        
        # Split the subplot into two parts
        ax2.clear()  # Clear the axis
        
        # Create two separate y-axes
        ax2_left = ax2
        ax2_right = ax2.twinx()
        
        # Plot μₐ on left axis (smaller values)
        mua_data = [results['healthy_mua'] * 1000, results['tumor_mua'] * 1000]
        mua_labels = ['Healthy', 'Tumor']
        mua_colors = [healthy_color, tumor_color]
        
        box_plot_mua = ax2_left.boxplot(mua_data, positions=[1, 2], labels=['', ''], 
                                       patch_artist=True, widths=0.3,
                                       boxprops=dict(alpha=0.7),
                                       medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(box_plot_mua['boxes'], mua_colors):
            patch.set_facecolor(color)
        
        ax2_left.set_ylabel('μₐ (×10⁻³ mm⁻¹)', color='black', fontsize=12)
        ax2_left.tick_params(axis='y', labelcolor='black')
        ax2_left.set_ylim(bottom=0)
        
        # Plot μₛ' on right axis (larger values)
        musp_data = [results['healthy_musp'], results['tumor_musp']]
        musp_labels = ['Healthy', 'Tumor']
        
        box_plot_musp = ax2_right.boxplot(musp_data, positions=[3.5, 4.5], labels=['', ''],
                                         patch_artist=True, widths=0.3,
                                         boxprops=dict(alpha=0.7),
                                         medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(box_plot_musp['boxes'], mua_colors):
            patch.set_facecolor(color)
        
        ax2_right.set_ylabel('μₛ\' (mm⁻¹)', color='black', fontsize=12)
        ax2_right.tick_params(axis='y', labelcolor='black')
        ax2_right.set_ylim(bottom=0)
        
        # Set x-axis labels and title
        ax2_left.set_xlim(0.5, 5)
        ax2_left.set_xticks([1.5, 4])
        ax2_left.set_xticklabels(['μₐ', 'μₛ\''], fontsize=12)
        ax2_left.set_title('Tissue Optical Properties (Dual Scale)', fontsize=14, fontweight='bold')
        ax2_left.grid(True, alpha=0.3)
        
        # Add improved labels positioned below the boxplots but visible
        y_label_pos = -0.08  # Adjusted position to be visible with increased spacing
        ax2_left.text(1, y_label_pos, 'Healthy\nμₐ', 
                     horizontalalignment='center', verticalalignment='top',
                     transform=ax2_left.get_xaxis_transform(),
                     fontsize=10, weight='bold')
        ax2_left.text(2, y_label_pos, 'Tumor\nμₐ', 
                     horizontalalignment='center', verticalalignment='top',
                     transform=ax2_left.get_xaxis_transform(),
                     fontsize=10, weight='bold')
        ax2_left.text(3.5, y_label_pos, 'Healthy\nμₛ\'', 
                     horizontalalignment='center', verticalalignment='top',
                     transform=ax2_left.get_xaxis_transform(),
                     fontsize=10, weight='bold')
        ax2_left.text(4.5, y_label_pos, 'Tumor\nμₛ\'', 
                     horizontalalignment='center', verticalalignment='top',
                     transform=ax2_left.get_xaxis_transform(),
                     fontsize=10, weight='bold')
        
        # Remove right axis labels to avoid confusion
        ax2_right.set_xticks([])
    else:
        ax2.text(0.5, 0.5, 'Insufficient optical property data', 
                transform=ax2.transAxes, horizontalalignment='center', 
                verticalalignment='center')
    
    # =========================================================================
    # Panel 3: Source-Detector Separations
    # =========================================================================
    ax3 = axes[1, 0]
    
    if len(results['sds_distances']) > 0:
        # Create histogram
        n_bins = 30
        hist_data, bin_edges = np.histogram(results['sds_distances'], bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = bin_centers[1] - bin_centers[0]
        
        ax3.bar(bin_centers, hist_data, width=bar_width*0.8, 
               alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add median line (without legend to avoid duplication)
        median_sds = np.median(results['sds_distances'])
        ax3.axvline(x=median_sds, color='orange', linestyle='-', linewidth=2)
        
        ax3.set_xlabel('Source-Detector Separation (mm)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Source-Detector Separation Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics (include median value since we removed legend)
        mean_sds = np.mean(results['sds_distances'])
        median_sds = np.median(results['sds_distances'])
        ax3.text(0.05, 0.95, f'Mean: {mean_sds:.1f} mm\nMedian: {median_sds:.1f} mm\nStd: {np.std(results["sds_distances"]):.1f} mm', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Panel 4: Probe Layout Coverage
    # =========================================================================
    ax4 = axes[1, 1]
    
    if len(results['source_positions']) > 0 and len(results['detector_positions']) > 0:
        # Sample positions for visualization (to avoid overcrowding)
        n_sample = min(5000, len(results['source_positions']))
        indices = np.random.choice(len(results['source_positions']), n_sample, replace=False)
        
        source_sample = results['source_positions'][indices]
        detector_sample = results['detector_positions'][indices]
        
        # Plot top-down view (x-y plane)
        ax4.scatter(source_sample[:, 0], source_sample[:, 1], 
                   c=source_color, s=1, alpha=0.6, label='Sources')
        ax4.scatter(detector_sample[:, 0], detector_sample[:, 1], 
                   c=detector_color, s=1, alpha=0.6, label='Detectors')
        
        ax4.set_xlabel('X Position (mm)', fontsize=12)
        ax4.set_ylabel('Y Position (mm)', fontsize=12)
        ax4.set_title('Probe Layout Coverage (Top View)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10, loc='upper right')
        ax4.set_aspect('equal')
        
        # Set proper coordinate system (0-64 mm in both X and Y)
        ax4.set_xlim(0, 64)
        ax4.set_ylim(0, 64)
    
    # Final figure adjustments with increased spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.35)  # Increased spacing between subplots
    
    # Add dataset summary text
    fig.text(0.5, 0.02, f'Dataset: {results["n_phantoms"]} phantoms analyzed', 
             horizontalalignment='center', fontsize=12, style='italic')
    
    return fig

def main():
    """Main execution function."""
    logger.info("🧬 Starting NIR Phantom Dataset Comprehensive Analysis 🧬")
    
    try:
        # Analyze the dataset (limit to 500 phantoms to prevent memory issues)
        results = analyze_dataset(max_phantoms=500)
        
        # Print summary statistics
        logger.info("=" * 60)
        logger.info("DATASET ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total phantoms analyzed: {results['n_phantoms']}")
        logger.info(f"Total measurements: {len(results['sds_distances'])}")
        logger.info(f"Tumor count range: {results['tumor_counts'].min()}-{results['tumor_counts'].max()}")
        if len(results['tumor_volumes']) > 0:
            logger.info(f"Tumor volume range: {np.min(results['tumor_volumes'])}-{np.max(results['tumor_volumes'])} voxels")
        logger.info(f"SDS range: {results['sds_distances'].min():.1f}-{results['sds_distances'].max():.1f} mm")
        
        # Generate the comprehensive figure
        logger.info("Generating comprehensive analysis figure...")
        fig = create_comprehensive_figure(results)
        
        # Save the figure
        output_file = "phantom_analysis.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved as: {output_file}")
        
        # Show the figure
        plt.show()
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
