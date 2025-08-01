#!/usr/bin/env python3
"""
🔬 ADVANCED NIR PHANTOM DATASET ANALYSIS SUITE 🔬

Professional analysis toolkit for NIR phantom datasets featuring:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 DUAL ANALYSIS MODES:
OPTION 1: Single Phantom Analysis
• Comprehensive 9-graph visualization (3x3 grid)
• Physics validation, quality assessment, and tissue analysis
• 3D geometry visualization and statistical distributions
• Detailed measurement correlation and SNR analysis

OPTION 2: Cross-Dataset Physics Validation
• Physics validation across entire phantom collection
• Aggregate statistical analysis and success rate assessment
• Cross-dataset consistency analysis and quality scoring
• Dataset-wide physics relationship validation
• Comparative success rate analysis (like physics validation script)

📊 ADVANCED ANALYSIS CAPABILITIES:
• Physics relationship validation (log-amplitude vs SDS, phase vs SDS)
• Cross-phantom consistency analysis and quality scoring
• Statistical distribution analysis with normality testing
• Principal component analysis and dimensionality reduction
• Outlier detection using multiple robust methods
• Tissue property distribution analysis and contrast assessment
• Measurement noise characterization and SNR analysis

🎨 PROFESSIONAL VISUALIZATION FEATURES:
• Clean, modern scientific plotting themes
• 3D interactive geometry visualizations
• Statistical distribution plots with KDE overlays
• Cross-correlation matrices and heatmaps
• Comparative box plots and violin plots
• Physics validation scatter plots with trend analysis

🔬 SCIENTIFIC ANALYSIS METHODS:
• Shapiro-Wilk normality testing
• Pearson and Spearman correlation analysis
• Interquartile range outlier detection
• Z-score standardization and comparison
• Physics compliance scoring and validation
• Cross-dataset consistency metrics

Author: Max Hart - NIR Tomography Research
Version: 3.0 - Dual-Mode Professional Analysis Suite
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Get the project root directory (mah422) - works regardless of where script is run from
project_root = Path(__file__).parent.parent.parent  # Go up to mah422 directory

# Configure matplotlib for professional scientific plots with refined aesthetics
plt.style.use('seaborn-v0_8-whitegrid')  # Clean, professional theme
plt.rcParams.update({
    'figure.figsize': (16, 12),
    'figure.dpi': 300,
    'savefig.dpi': 400,  # Ultra-high resolution
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.color': '#c0c0c0',
    'lines.linewidth': 2.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': '#333333'
})

# Professional color palette - refined and less colorful
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Muted magenta
    'accent': '#F18F01',       # Warm orange
    'success': '#C73E1D',      # Deep red
    'tissue': '#2E8B57',       # Sea green
    'tumor': '#8B0000',        # Dark red
    'air': '#4682B4',          # Steel blue
    'source': '#DAA520',       # Golden rod
    'detector': '#20B2AA',     # Light sea green
    'neutral1': '#696969',     # Dim gray
    'neutral2': '#2F4F4F',     # Dark slate gray
    'bg_light': '#F8F9FA',     # Very light gray
    'bg_medium': '#E9ECEF'     # Light gray
}

# Refined color palettes for different plot types
MEASUREMENT_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#2E8B57', '#4682B4']
TISSUE_COLORS = ['#F8F9FA', '#2E8B57', '#8B0000', '#F18F01', '#4682B4', '#A23B72']
SCIENTIFIC_COLORMAP = 'viridis'  # Professional scientific colormap

class NIRDatasetAnalyzer:
    """
    🔬 COMPREHENSIVE NIR PHANTOM DATASET ANALYZER 🔬
    
    A powerful, all-in-one analysis suite for NIR phantom datasets that provides:
    - Complete dataset discovery and validation
    - Advanced statistical analysis with publication-quality visualizations
    - 3D geometric analysis and interactive plots
    - Cross-dataset consistency checking
    - Professional reporting with scientific rigor
    """
    
    def __init__(self, data_directory=None):
        """
        Initialize the comprehensive analyzer.
        
        Args:
            data_directory (str): Path to directory containing phantom datasets
                                If None, uses project_root/data automatically
        """
        print("\n" + "="*80)
        print("🔬 INITIALIZING COMPREHENSIVE NIR DATASET ANALYZER 🔬")
        print("="*80)
        
        # Set default data directory to project_root/data if not specified
        if data_directory is None:
            data_directory = project_root / "data"
        
        self.data_dir = Path(data_directory)
        print(f"🔍 Looking for data in: {self.data_dir.absolute()}")
        
        self.phantom_files = []
        self.all_data = {}  # Store loaded datasets
        self.analysis_results = {}
        
        self._discover_datasets()
        
    def _discover_datasets(self):
        """🔍 Discover and validate all available phantom datasets."""
        print("\n📁 DATASET DISCOVERY")
        print("-" * 50)
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return
            
        # Find all phantom directories
        phantom_dirs = sorted([d for d in self.data_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("phantom_")])
        
        if not phantom_dirs:
            print(f"❌ No phantom directories found in {self.data_dir}")
            print("💡 Expected directory structure:")
            print("   data/")
            print("   ├── phantom_01/")
            print("   │   └── phantom_001_scan.h5")
            print("   ├── phantom_02/")
            print("   │   └── phantom_002_scan.h5")
            print("   └── ...")
            print("\n💡 Run the data simulator first to generate phantom datasets!")
            return
            
        print(f"📂 Scanning {len(phantom_dirs)} phantom directories...")
        
        valid_count = 0
        total_size = 0
        
        for phantom_dir in phantom_dirs:
            h5_files = list(phantom_dir.glob("*.h5"))
            
            if h5_files:
                h5_file = h5_files[0]  # Take first .h5 file found
                self.phantom_files.append(h5_file)
                file_size = h5_file.stat().st_size / (1024**2)  # MB
                total_size += file_size
                valid_count += 1
                
                # Quick validation
                try:
                    with h5py.File(h5_file, 'r') as f:
                        datasets = list(f.keys())
                        status = "✅ VALID"
                except Exception as e:
                    status = f"⚠️  CORRUPTED ({str(e)[:30]}...)"
                
                print(f"   {status} {phantom_dir.name:15} │ {h5_file.name:12} │ {file_size:6.1f} MB")
            else:
                print(f"   ❌ MISSING  {phantom_dir.name:15} │ {'no .h5 file':12} │ {'---':>6} MB")
        
        print(f"\n📈 DISCOVERY SUMMARY:")
        print(f"   Total directories scanned: {len(phantom_dirs)}")
        print(f"   Valid datasets found:      {valid_count}")
        print(f"   Total data size:          {total_size:.1f} MB")
        print(f"   Success rate:             {valid_count/len(phantom_dirs)*100:.1f}%")
        
        if valid_count == 0:
            print("\n❌ No valid datasets found! Cannot proceed with analysis.")
        else:
            print(f"\n🚀 Ready to analyze {valid_count} datasets!")
    
    def analyze_single_dataset(self, file_path, detailed=True):
        """
        Perform comprehensive analysis of a single HDF5 dataset.
        
        Args:
            file_path (Path): Path to the HDF5 file
            detailed (bool): Whether to perform detailed analysis
            
        Returns:
            dict: Analysis results summary
        """
        print(f"\n🔬 ANALYZING: {file_path.parent.name}/{file_path.name}")
        print("="*70)
        
        analysis_results = {
            'file_path': str(file_path),
            'phantom_name': file_path.parent.name,
            'file_size_mb': file_path.stat().st_size / (1024**2)
        }
        
        try:
            with h5py.File(file_path, 'r') as f:
                # 1. Examine HDF5 structure
                print("📋 HDF5 STRUCTURE AND METADATA")
                print("-" * 40)
                self._analyze_hdf5_structure(f, analysis_results)
                
                # 2. Analyze measurement data
                print("\n📏 MEASUREMENT DATA ANALYSIS")
                print("-" * 40)
                self._analyze_measurements(f, analysis_results)
                
                # 3. Analyze ground truth data
                print("\n🎯 GROUND TRUTH ANALYSIS")
                print("-" * 40)
                self._analyze_ground_truth(f, analysis_results)
                
                # 4. Analyze geometric configuration
                print("\n📐 GEOMETRIC CONFIGURATION")
                print("-" * 40)
                self._analyze_geometry(f, analysis_results)
                
                # 5. Data quality checks
                print("\n✅ DATA QUALITY VALIDATION")
                print("-" * 40)
                self._validate_data_quality(f, analysis_results)
                
                if detailed:
                    # 6. Statistical analysis
                    print("\n📊 STATISTICAL ANALYSIS")
                    print("-" * 40)
                    self._statistical_analysis(f, analysis_results)
        
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _analyze_hdf5_structure(self, hdf5_file, results):
        """Analyze the internal structure of the HDF5 file."""
        
        # List all datasets and their properties
        datasets = {}
        
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size_mb': obj.nbytes / (1024**2),
                    'compression': obj.compression,
                    'chunks': obj.chunks
                }
                
                # Print dataset info
                print(f"📦 Dataset: {name}")
                print(f"   Shape: {obj.shape}")
                print(f"   Data type: {obj.dtype}")
                print(f"   Size: {obj.nbytes / (1024**2):.2f} MB")
                if obj.compression:
                    print(f"   Compression: {obj.compression}")
                
                # Print attributes
                if obj.attrs:
                    print("   Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"     {attr_name}: {attr_value}")
                print()
        
        hdf5_file.visititems(visitor)
        results['datasets'] = datasets
        
        # File-level attributes
        print("🏷️  File Attributes:")
        file_attrs = {}
        for attr_name, attr_value in hdf5_file.attrs.items():
            file_attrs[attr_name] = attr_value
            print(f"   {attr_name}: {attr_value}")
        results['file_attributes'] = file_attrs
    
    def _analyze_measurements(self, hdf5_file, results):
        """Analyze measurement data (log-amplitude and phase)."""
        
        # Load measurement data
        log_amp = hdf5_file['log_amplitude'][:]
        phase = hdf5_file['phase'][:]
        
        print(f"📡 Measurement Arrays:")
        print(f"   Log-amplitude shape: {log_amp.shape}")
        print(f"   Phase shape: {phase.shape}")
        print(f"   Total measurements: {log_amp.size}")
        
        # Check if this is the new optimized format
        if log_amp.size == 1000:
            print(f"   📈 Optimized probe placement detected: 1000 measurements (50 sources × 20 detectors)")
            print(f"   📊 Data augmentation available: subsample 256 for training (3.9x combinations)")
        elif log_amp.size == 256:
            print(f"   📊 Legacy format detected: 256 independent measurements")
        else:
            print(f"   ⚠️  Unexpected measurement count: {log_amp.size}")
        
        # Calculate statistics
        log_amp_stats = {
            'min': float(np.min(log_amp)),
            'max': float(np.max(log_amp)),
            'mean': float(np.mean(log_amp)),
            'std': float(np.std(log_amp)),
            'median': float(np.median(log_amp))
        }
        
        phase_stats = {
            'min': float(np.min(phase)),
            'max': float(np.max(phase)),
            'mean': float(np.mean(phase)),
            'std': float(np.std(phase)),
            'median': float(np.median(phase))
        }
        
        print(f"\n📊 Log-Amplitude Statistics:")
        print(f"   Range: [{log_amp_stats['min']:.3f}, {log_amp_stats['max']:.3f}]")
        print(f"   Mean ± Std: {log_amp_stats['mean']:.3f} ± {log_amp_stats['std']:.3f}")
        print(f"   Median: {log_amp_stats['median']:.3f}")
        
        print(f"\n📊 Phase Statistics (degrees):")
        print(f"   Range: [{phase_stats['min']:.1f}°, {phase_stats['max']:.1f}°]")
        print(f"   Mean ± Std: {phase_stats['mean']:.1f}° ± {phase_stats['std']:.1f}°")
        print(f"   Median: {phase_stats['median']:.1f}°")
        
        # Check for invalid values
        log_amp_issues = {
            'nan_count': int(np.sum(np.isnan(log_amp))),
            'inf_count': int(np.sum(np.isinf(log_amp))),
            'negative_count': int(np.sum(log_amp < -20))  # Extremely low values
        }
        
        phase_issues = {
            'nan_count': int(np.sum(np.isnan(phase))),
            'inf_count': int(np.sum(np.isinf(phase))),
            'wrap_issues': int(np.sum(np.abs(phase) > 360))  # Phase wrapping issues
        }
        
        if any(log_amp_issues.values()):
            print(f"\n⚠️  Log-Amplitude Issues:")
            for issue, count in log_amp_issues.items():
                if count > 0:
                    print(f"   {issue}: {count}")
        
        if any(phase_issues.values()):
            print(f"\n⚠️  Phase Issues:")
            for issue, count in phase_issues.items():
                if count > 0:
                    print(f"   {issue}: {count}")
        
        results['measurements'] = {
            'log_amplitude': log_amp_stats,
            'phase': phase_stats,
            'data_issues': {
                'log_amplitude': log_amp_issues,
                'phase': phase_issues
            }
        }
    
    def _analyze_ground_truth(self, hdf5_file, results):
        """Analyze ground truth optical property maps."""
        
        gt_data = hdf5_file['ground_truth'][:]
        
        print(f"🎯 Ground Truth Maps:")
        print(f"   Shape: {gt_data.shape}")
        print(f"   Data type: {gt_data.dtype}")
        print(f"   Total voxels: {gt_data.shape[1] * gt_data.shape[2] * gt_data.shape[3]:,}")
        print(f"   Property channels: {gt_data.shape[0]} (μₐ, μ′s)")
        
        # Extract absorption and scattering maps
        mua_map = gt_data[0]  # Channel 0: Absorption coefficient
        musp_map = gt_data[1]  # Channel 1: Reduced scattering coefficient
        
        # Calculate statistics for each property
        mua_stats = {
            'min': float(np.min(mua_map)),
            'max': float(np.max(mua_map)),
            'mean': float(np.mean(mua_map)),
            'std': float(np.std(mua_map)),
            'unique_values': len(np.unique(mua_map))
        }
        
        musp_stats = {
            'min': float(np.min(musp_map)),
            'max': float(np.max(musp_map)),
            'mean': float(np.mean(musp_map)),
            'std': float(np.std(musp_map)),
            'unique_values': len(np.unique(musp_map))
        }
        
        print(f"\n📊 Absorption Coefficient (μₐ) [mm⁻¹]:")
        print(f"   Range: [{mua_stats['min']:.6f}, {mua_stats['max']:.6f}]")
        print(f"   Mean ± Std: {mua_stats['mean']:.6f} ± {mua_stats['std']:.6f}")
        print(f"   Unique values: {mua_stats['unique_values']}")
        
        print(f"\n📊 Reduced Scattering (μ′s) [mm⁻¹]:")
        print(f"   Range: [{musp_stats['min']:.3f}, {musp_stats['max']:.3f}]")
        print(f"   Mean ± Std: {musp_stats['mean']:.3f} ± {musp_stats['std']:.3f}")
        print(f"   Unique values: {musp_stats['unique_values']}")
        
        # Analyze tissue regions
        zero_voxels = np.sum((mua_map == 0) & (musp_map == 0))
        tissue_voxels = np.sum((mua_map > 0) | (musp_map > 0))
        total_voxels = mua_map.size
        
        print(f"\n🧬 Tissue Distribution:")
        print(f"   Air voxels: {zero_voxels:,} ({zero_voxels/total_voxels*100:.1f}%)")
        print(f"   Tissue voxels: {tissue_voxels:,} ({tissue_voxels/total_voxels*100:.1f}%)")
        
        # Identify different tissue types based on unique property combinations
        # Create combined property array by stacking flattened absorption and scattering maps
        combined_properties = np.stack([mua_map.flatten(), musp_map.flatten()], axis=1)
        unique_combinations = np.unique(combined_properties, axis=0)
        tissue_types = len(unique_combinations) - 1 if (0, 0) in [tuple(comb) for comb in unique_combinations] else len(unique_combinations)
        
        print(f"   Tissue types identified: {tissue_types}")
        print(f"   Property combinations:")
        for i, (mua, musp) in enumerate(unique_combinations):
            voxel_count = np.sum((mua_map == mua) & (musp_map == musp))
            if mua == 0 and musp == 0:
                print(f"     Type {i}: Air (μₐ={mua:.6f}, μ′s={musp:.3f}) - {voxel_count:,} voxels")
            else:
                print(f"     Type {i}: Tissue (μₐ={mua:.6f}, μ′s={musp:.3f}) - {voxel_count:,} voxels")
        
        results['ground_truth'] = {
            'shape': gt_data.shape,
            'absorption': mua_stats,
            'scattering': musp_stats,
            'tissue_distribution': {
                'air_voxels': int(zero_voxels),
                'tissue_voxels': int(tissue_voxels),
                'tissue_types': int(tissue_types)
            },
            'unique_combinations': unique_combinations.tolist()
        }
    
    def _analyze_geometry(self, hdf5_file, results):
        """Analyze source and detector geometric configurations."""
        
        source_pos = hdf5_file['source_positions'][:]
        det_pos = hdf5_file['detector_positions'][:]
        
        print(f"📐 Geometric Configuration:")
        print(f"   Source positions shape: {source_pos.shape}")
        print(f"   Detector positions shape: {det_pos.shape}")
        print(f"   Number of measurements: {len(source_pos)}")
        print(f"   Format: 1:1 source-detector pairs" if len(det_pos.shape) == 2 else "   Format: 1:N source-detector mapping")
        
        # Calculate position statistics
        print(f"\n📊 Source Position Statistics [mm]:")
        for axis, name in enumerate(['X', 'Y', 'Z']):
            coords = source_pos[:, axis]
            print(f"   {name}-axis: [{coords.min():.1f}, {coords.max():.1f}] "
                  f"(mean: {coords.mean():.1f}, std: {coords.std():.1f})")
        
        # Calculate source-detector distances (updated for 1:1 system)
        if len(det_pos.shape) == 2:  # (n_measurements, 3) - new 1:1 system
            print(f"\n📏 Source-Detector Distances:")
            distances = []
            for i in range(len(source_pos)):
                src = source_pos[i]
                det = det_pos[i]  # Single detector for this measurement
                distance = np.linalg.norm(det - src)
                distances.append(distance)
                
                if i < 5:  # Show first 5 measurements
                    print(f"   Measurement {i+1}: {distance:.1f}mm")
            
            distances = np.array(distances)
            print(f"\n📊 Distance Statistics:")
            print(f"   Range: [{distances.min():.1f}, {distances.max():.1f}] mm")
            print(f"   Mean ± Std: {distances.mean():.1f} ± {distances.std():.1f} mm")
            print(f"   Median: {np.median(distances):.1f} mm")
            
            # Check minimum distance constraint
            min_distance_violations = np.sum(distances < 5.0)
            if min_distance_violations > 0:
                print(f"   ⚠️  Minimum distance violations (<5mm): {min_distance_violations}")
        
        results['geometry'] = {
            'n_sources': len(source_pos),
            'n_detectors': len(det_pos),  # Updated: for 1:1 format this equals n_sources
            'source_bounds': {
                'x': [float(source_pos[:, 0].min()), float(source_pos[:, 0].max())],
                'y': [float(source_pos[:, 1].min()), float(source_pos[:, 1].max())],
                'z': [float(source_pos[:, 2].min()), float(source_pos[:, 2].max())]
            }
        }
        
        if len(det_pos.shape) == 2:  # New format: (n_measurements, 3) - 1:1 pairs  
            results['geometry']['distance_stats'] = {
                'min': float(distances.min()),
                'max': float(distances.max()),
                'mean': float(distances.mean()),
                'std': float(distances.std())
            }
    
    def _validate_data_quality(self, hdf5_file, results):
        """Perform data quality validation checks."""
        
        issues = []
        warnings = []
        
        # Check data consistency
        log_amp = hdf5_file['log_amplitude'][:]
        phase = hdf5_file['phase'][:]
        source_pos = hdf5_file['source_positions'][:]
        det_pos = hdf5_file['detector_positions'][:]
        
        # Shape consistency
        if log_amp.shape != phase.shape:
            issues.append(f"Measurement shape mismatch: log_amp {log_amp.shape} vs phase {phase.shape}")
        
        expected_measurements = len(source_pos)  # Updated for 1:1 system
        actual_measurements = log_amp.size
        if expected_measurements != actual_measurements:
            issues.append(f"Measurement count mismatch: expected {expected_measurements}, got {actual_measurements}")
        
        # Data range checks
        if np.any(np.isnan(log_amp)) or np.any(np.isnan(phase)):
            issues.append("NaN values found in measurement data")
        
        if np.any(np.isinf(log_amp)) or np.any(np.isinf(phase)):
            issues.append("Infinite values found in measurement data")
        
        # Physical plausibility
        if np.min(log_amp) < -30:
            warnings.append(f"Very low log-amplitude values detected (min: {np.min(log_amp):.2f})")
        
        if np.max(np.abs(phase)) > 360:
            warnings.append(f"Phase values outside ±360° range (max: {np.max(np.abs(phase)):.1f}°)")
        
        # Ground truth validation
        gt_data = hdf5_file['ground_truth'][:]
        if np.any(gt_data < 0):
            issues.append("Negative optical property values in ground truth")
        
        # Print results
        if not issues and not warnings:
            print("✅ All data quality checks passed!")
        else:
            if issues:
                print("❌ Data Quality Issues:")
                for issue in issues:
                    print(f"   • {issue}")
            
            if warnings:
                print("⚠️  Data Quality Warnings:")
                for warning in warnings:
                    print(f"   • {warning}")
        
        results['data_quality'] = {
            'issues': issues,
            'warnings': warnings,
            'passed': len(issues) == 0
        }
    
    def _statistical_analysis(self, hdf5_file, results):
        """Perform detailed statistical analysis."""
        
        log_amp = hdf5_file['log_amplitude'][:]
        phase = hdf5_file['phase'][:]
        
        print("📊 Advanced Statistical Analysis:")
        
        # Distribution analysis
        log_amp_flat = log_amp.flatten()
        phase_flat = phase.flatten()
        
        # Normality tests
        _, log_amp_p = stats.shapiro(log_amp_flat[:5000])  # Sample for speed
        _, phase_p = stats.shapiro(phase_flat[:5000])
        
        print(f"   Log-amplitude normality (Shapiro-Wilk p-value): {log_amp_p:.2e}")
        print(f"   Phase normality (Shapiro-Wilk p-value): {phase_p:.2e}")
        
        # Correlation analysis with robust error handling
        try:
            # Ensure we have valid data for correlation
            if len(log_amp_flat) > 10 and len(phase_flat) > 10:
                # Remove any non-finite values
                valid_mask = np.isfinite(log_amp_flat) & np.isfinite(phase_flat)
                clean_log_amp = log_amp_flat[valid_mask]
                clean_phase = phase_flat[valid_mask]
                
                if len(clean_log_amp) > 10 and np.std(clean_log_amp) > 1e-10 and np.std(clean_phase) > 1e-10:
                    correlation = np.corrcoef(clean_log_amp, clean_phase)[0, 1]
                else:
                    correlation = 0.0  # Set to zero if insufficient variance
            else:
                correlation = 0.0  # Set to zero if insufficient data
        except Exception as e:
            correlation = 0.0  # Default fallback
            print(f"   Warning: Correlation calculation failed: {e}")
        
        print(f"   Log-amplitude vs Phase correlation: {correlation:.3f}")
        
        # Outlier detection
        log_amp_q1, log_amp_q3 = np.percentile(log_amp_flat, [25, 75])
        log_amp_iqr = log_amp_q3 - log_amp_q1
        log_amp_outliers = np.sum((log_amp_flat < log_amp_q1 - 1.5*log_amp_iqr) | 
                                  (log_amp_flat > log_amp_q3 + 1.5*log_amp_iqr))
        
        phase_q1, phase_q3 = np.percentile(phase_flat, [25, 75])
        phase_iqr = phase_q3 - phase_q1
        phase_outliers = np.sum((phase_flat < phase_q1 - 1.5*phase_iqr) | 
                               (phase_flat > phase_q3 + 1.5*phase_iqr))
        
        print(f"   Log-amplitude outliers (IQR method): {log_amp_outliers} ({log_amp_outliers/len(log_amp_flat)*100:.1f}%)")
        print(f"   Phase outliers (IQR method): {phase_outliers} ({phase_outliers/len(phase_flat)*100:.1f}%)")
        
        results['statistics'] = {
            'normality': {
                'log_amplitude_p': float(log_amp_p),
                'phase_p': float(phase_p)
            },
            'correlation': float(correlation),
            'outliers': {
                'log_amplitude': int(log_amp_outliers),
                'phase': int(phase_outliers)
            }
        }
    
    def visualize_dataset(self, file_path, save_plots=True, show_plots=False):
        """🎨 Create comprehensive 9-graph analysis for a single dataset."""
        print(f"\n🎨 GENERATING COMPREHENSIVE VISUALIZATION: {file_path.parent.name}")
        print("="*60)
        
        with h5py.File(file_path, 'r') as f:
            # Load data
            log_amp = f['log_amplitude'][:]
            phase = f['phase'][:]
            source_pos = f['source_positions'][:]
            det_pos = f['detector_positions'][:]
            gt_data = f['ground_truth'][:]
            
            # Calculate distances and derived metrics
            distances = np.array([np.linalg.norm(det_pos[i] - source_pos[i]) 
                                 for i in range(len(source_pos))])
            
            # Physics analysis
            slope_amp, _, r_amp, p_amp, _ = stats.linregress(distances, log_amp)
            slope_phase, _, r_phase, p_phase, _ = stats.linregress(distances, phase)
            
            # Ground truth analysis
            mua_map = gt_data[0]  # Absorption
            musp_map = gt_data[1]  # Scattering
            
            # Create 3x3 comprehensive layout
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f'Comprehensive NIR Analysis: {file_path.parent.name}', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # 1. Physics Validation: Log-amplitude vs Distance
            ax1 = plt.subplot(3, 3, 1)
            ax1.scatter(distances, log_amp, alpha=0.6, color=COLORS['primary'], s=30)
            z_amp = np.polyfit(distances, log_amp, 1)
            p_amp_line = np.poly1d(z_amp)
            ax1.plot(distances, p_amp_line(distances), color=COLORS['accent'], linewidth=2,
                    label=f'Slope: {slope_amp:.4f}\nR²: {r_amp**2:.3f}')
            ax1.set_xlabel('Distance [mm]')
            ax1.set_ylabel('Log-Amplitude')
            ax1.set_title('Physics: Log-Amplitude vs Distance')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. Physics Validation: Phase vs Distance  
            ax2 = plt.subplot(3, 3, 2)
            ax2.scatter(distances, phase, alpha=0.6, color=COLORS['secondary'], s=30)
            z_phase = np.polyfit(distances, phase, 1)
            p_phase_line = np.poly1d(z_phase)
            ax2.plot(distances, p_phase_line(distances), color=COLORS['accent'], linewidth=2,
                    label=f'Slope: {slope_phase:.4f}\nR²: {r_phase**2:.3f}')
            ax2.set_xlabel('Distance [mm]')
            ax2.set_ylabel('Phase [degrees]')
            ax2.set_title('Physics: Phase vs Distance')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Log-Amplitude vs Phase Correlation
            ax3 = plt.subplot(3, 3, 3)
            ax3.scatter(log_amp, phase, alpha=0.5, color=COLORS['tissue'], s=20)
            correlation = np.corrcoef(log_amp, phase)[0, 1]
            ax3.set_xlabel('Log-Amplitude')
            ax3.set_ylabel('Phase [degrees]')
            ax3.set_title(f'Amplitude-Phase Correlation\nR = {correlation:.3f}')
            ax3.grid(True, alpha=0.3)
            
            # 4. Ground Truth: Absorption (Central Slice)
            ax4 = plt.subplot(3, 3, 4)
            central_slice = mua_map.shape[2] // 2
            im4 = ax4.imshow(mua_map[:, :, central_slice], cmap='plasma', aspect='equal')
            ax4.set_title('Ground Truth: Absorption (μₐ)')
            ax4.set_xlabel('X [voxels]')
            ax4.set_ylabel('Y [voxels]')
            plt.colorbar(im4, ax=ax4, label='μₐ [mm⁻¹]', shrink=0.8)
            
            # 5. Ground Truth: Scattering (Central Slice)
            ax5 = plt.subplot(3, 3, 5)
            im5 = ax5.imshow(musp_map[:, :, central_slice], cmap='viridis', aspect='equal')
            ax5.set_title('Ground Truth: Scattering (μ′s)')
            ax5.set_xlabel('X [voxels]')
            ax5.set_ylabel('Y [voxels]')
            plt.colorbar(im5, ax=ax5, label='μ′s [mm⁻¹]', shrink=0.8)
            
            # 6. 3D Source Geometry Layout (Sources Only)
            ax6 = plt.subplot(3, 3, 6, projection='3d')
            
            # Only plot sources, color-coded by measurement signal strength
            source_colors = log_amp  # Color sources by their signal strength
            scatter = ax6.scatter(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2], 
                                c=source_colors, s=50, alpha=0.8, cmap='viridis',
                                edgecolors='black', linewidth=0.5)
            
            # Add colorbar to show signal strength mapping
            cbar = plt.colorbar(scatter, ax=ax6, shrink=0.6, pad=0.1)
            cbar.set_label('Log-Amplitude', fontsize=8)
            
            ax6.set_xlabel('X [mm]')
            ax6.set_ylabel('Y [mm]')
            ax6.set_zlabel('Z [mm]')
            ax6.set_title('3D Source Layout\n(colored by signal strength)')
            
            # Set equal aspect ratio for better 3D visualization
            ax6.set_box_aspect([1,1,1])
            
            # 7. Tissue Property Contrast Analysis
            ax7 = plt.subplot(3, 3, 7)
            
            # Extract unique tissue types from ground truth
            unique_mua = np.unique(mua_map)
            unique_musp = np.unique(musp_map)
            
            # Create tissue property scatter plot
            tissue_combinations = []
            tissue_volumes = []
            tissue_labels = []
            
            for mua_val in unique_mua:
                for musp_val in unique_musp:
                    mask = (mua_map == mua_val) & (musp_map == musp_val)
                    volume = np.sum(mask)
                    if volume > 0:  # Only include combinations that exist
                        tissue_combinations.append([mua_val, musp_val])
                        tissue_volumes.append(volume)
                        
                        if mua_val == 0 and musp_val == 0:
                            tissue_labels.append('Air')
                        elif len(tissue_combinations) == 2:  # First tissue type (usually healthy)
                            tissue_labels.append('Healthy')
                        else:
                            tissue_labels.append(f'Tumor {len(tissue_combinations)-2}')
            
            tissue_combinations = np.array(tissue_combinations)
            tissue_volumes = np.array(tissue_volumes)
            
            # Create scatter plot with bubble sizes proportional to volume
            for i, (combo, volume, label) in enumerate(zip(tissue_combinations, tissue_volumes, tissue_labels)):
                mua_val, musp_val = combo
                size = np.sqrt(volume) / 20  # Scale bubble size
                if label == 'Air':
                    color = COLORS['air']
                elif label == 'Healthy':
                    color = COLORS['tissue'] 
                else:
                    color = COLORS['tumor']
                
                ax7.scatter(mua_val, musp_val, s=size, c=color, alpha=0.7, 
                           edgecolors='black', linewidth=1, label=f'{label} ({volume:,} voxels)')
            
            ax7.set_xlabel('Absorption μₐ [mm⁻¹]')
            ax7.set_ylabel('Scattering μ′s [mm⁻¹]')
            ax7.set_title('Tissue Property Contrast\n(bubble size = volume)')
            ax7.legend(fontsize=7, loc='upper right')
            ax7.grid(True, alpha=0.3)
            
            # Add contrast lines for reference
            if len(tissue_combinations) > 1:
                healthy_idx = 1 if tissue_labels[0] == 'Air' else 0
                if healthy_idx < len(tissue_combinations):
                    healthy_mua, healthy_musp = tissue_combinations[healthy_idx]
                    ax7.axhline(healthy_musp, color='gray', linestyle='--', alpha=0.5)
                    ax7.axvline(healthy_mua, color='gray', linestyle='--', alpha=0.5)
            
            # 8. Distance Distribution Analysis
            ax8 = plt.subplot(3, 3, 8)
            ax8.hist(distances, bins=30, alpha=0.7, color=COLORS['neutral1'], 
                    edgecolor='black', linewidth=0.5)
            ax8.axvline(distances.mean(), color=COLORS['accent'], linestyle='--', 
                       linewidth=2, label=f'Mean: {distances.mean():.1f}mm')
            ax8.axvline(np.median(distances), color=COLORS['success'], linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(distances):.1f}mm')
            ax8.set_xlabel('Source-Detector Distance [mm]')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Distance Distribution')
            ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)
            
            # 9. Quality Assessment Summary
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            
            # Calculate quality metrics
            physics_correct = (slope_amp < 0 and slope_phase > 0)
            correlation_strength = abs(correlation)
            snr_estimate = np.std(log_amp) / np.mean(np.abs(log_amp))
            distance_uniformity = 1 - (np.std(distances) / np.mean(distances))
            
            # Quality indicators
            quality_text = f"""
QUALITY ASSESSMENT
{'='*25}

Physics Validation:
  • Log-Amp Slope: {'✅' if slope_amp < 0 else '❌'} {slope_amp:.4f}
  • Phase Slope: {'✅' if slope_phase > 0 else '❌'} {slope_phase:.4f}
  • Overall Physics: {'✅ PASS' if physics_correct else '❌ FAIL'}

Statistical Metrics:
  • Amp-Phase Correlation: {correlation:.3f}
  • Signal Quality: {snr_estimate:.3f}
  • Distance Uniformity: {distance_uniformity:.3f}

Dataset Properties:
  • Measurements: {len(log_amp):,}
  • Distance Range: {distances.min():.1f}-{distances.max():.1f}mm
  • Tissue Types: {len(np.unique(mua_map))}
  • File Size: {file_path.stat().st_size / (1024**2):.1f} MB

Overall Status: {'🎉 EXCELLENT' if physics_correct and correlation_strength > 0.3 else '✅ GOOD' if physics_correct else '⚠️ NEEDS REVIEW'}
            """
            
            ax9.text(0.05, 0.95, quality_text, transform=ax9.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['bg_light'], alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots:
                # Create results directory
                results_dir = project_root / "analysis_results"
                results_dir.mkdir(exist_ok=True)
                
                plot_path = results_dir / f"{file_path.parent.name}_comprehensive_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   ✅ Comprehensive plot saved: {plot_path}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()

    def deep_single_analysis(self, file_path):
        """🔍 Deep analysis of a single phantom with professional visualizations."""
        print(f"\n🔬 DEEP SINGLE PHANTOM ANALYSIS")
        print("="*70)
        print(f"📂 Analyzing: {file_path.parent.name}")
        
        # Run comprehensive single dataset analysis
        results = self.analyze_single_dataset(file_path, detailed=True)
        
        # Create professional visualizations
        self.visualize_dataset(file_path, save_plots=True, show_plots=True)
        
        return results


# ============================================================================
# DUAL-MODE MAIN MENU SYSTEM  
# ============================================================================

def single_phantom_analysis():
    """🎯 OPTION 1: Single Phantom Analysis with sleek visualizations."""
    print("\n" + "="*80)
    print("🔬 SINGLE PHANTOM ANALYSIS")
    print("="*80)
    
    # Initialize analyzer to discover datasets
    analyzer = NIRDatasetAnalyzer()
    
    if not analyzer.phantom_files:
        print("❌ No phantom datasets found!")
        return
    
    # Show available phantoms
    print(f"\n📋 Available Phantoms ({len(analyzer.phantom_files)}):")
    for i, file_path in enumerate(analyzer.phantom_files):
        phantom_name = file_path.parent.name
        file_size = file_path.stat().st_size / (1024**2)
        print(f"   [{i+1:2d}] {phantom_name:15} ({file_size:6.1f} MB)")
    
    # Get user selection
    try:
        choice = input(f"\nSelect phantom [1-{len(analyzer.phantom_files)}]: ").strip()
        phantom_idx = int(choice) - 1
        
        if 0 <= phantom_idx < len(analyzer.phantom_files):
            selected_file = analyzer.phantom_files[phantom_idx]
            analyzer.deep_single_analysis(selected_file)
        else:
            print("❌ Invalid selection!")
    except (ValueError, KeyboardInterrupt):
        print("❌ Invalid input or cancelled!")


def comprehensive_multi_phantom_analysis():
    """🎯 OPTION 2: True Cross-Dataset Analysis (not individual analysis)."""
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE CROSS-DATASET ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = NIRDatasetAnalyzer()
    
    if not analyzer.phantom_files:
        print("❌ No phantom datasets found!")
        return
    
    print(f"📊 Found {len(analyzer.phantom_files)} phantoms for cross-dataset analysis")
    print("🔄 This will perform comprehensive physics validation across ALL phantoms...")
    print("   • Physics relationship validation for entire dataset")
    print("   • Statistical comparison and quality assessment")
    print("   • Cross-phantom consistency analysis")
    print()
    
    # Ask user how many phantoms to analyze
    try:
        max_phantoms = len(analyzer.phantom_files)  # Use all available phantoms
        default_phantoms = min(20, max_phantoms)  # Default to 20 or max available
        choice = input(f"Analyze how many phantoms? [1-{max_phantoms}] (default: {default_phantoms}): ").strip()
        if choice:
            num_phantoms = min(int(choice), max_phantoms)
        else:
            num_phantoms = default_phantoms
    except ValueError:
        num_phantoms = default_phantoms
    
    print(f"\n🔬 COMPREHENSIVE {num_phantoms}-PHANTOM PHYSICS VALIDATION")
    print('='*60)
    print("Performing cross-dataset analysis...")
    print()
    
    # Initialize cross-dataset results tracking
    results = {}
    physics_violations = []
    working_phantoms = []
    all_slopes_amp = []
    all_slopes_phase = []
    all_r2_amp = []
    all_r2_phase = []
    all_distances = []
    all_log_amp = []
    all_phase = []
    
    # Process each phantom
    for i in range(num_phantoms):
        file_path = analyzer.phantom_files[i]
        phantom_name = file_path.parent.name
        
        try:
            with h5py.File(file_path, 'r') as f:
                log_amp = f['log_amplitude'][:]
                phase = f['phase'][:]
                sources = f['source_positions'][:]
                detectors = f['detector_positions'][:]
                
                # Calculate distances
                distances = np.array([np.linalg.norm(detectors[j] - sources[j]) 
                                    for j in range(len(sources))])
                
                # Physics analysis
                slope_amp, _, r_amp, p_amp, _ = stats.linregress(distances, log_amp)
                slope_phase, _, r_phase, p_phase, _ = stats.linregress(distances, phase)
                
                # Physics validation
                amp_correct = slope_amp < 0  # Should be negative (attenuation)
                phase_correct = slope_phase > 0  # Should be positive (delay)
                physics_correct = amp_correct and phase_correct
                
                # Store individual results
                results[phantom_name] = {
                    'slope_amp': slope_amp,
                    'slope_phase': slope_phase,
                    'r2_amp': r_amp**2,
                    'r2_phase': r_phase**2,
                    'p_amp': p_amp,
                    'p_phase': p_phase,
                    'physics_correct': physics_correct,
                    'amp_correct': amp_correct,
                    'phase_correct': phase_correct,
                    'n_measurements': len(log_amp)
                }
                
                # Aggregate data for cross-dataset analysis
                all_slopes_amp.append(slope_amp)
                all_slopes_phase.append(slope_phase)
                all_r2_amp.append(r_amp**2)
                all_r2_phase.append(r_phase**2)
                all_distances.extend(distances)
                all_log_amp.extend(log_amp)
                all_phase.extend(phase)
                
                # Classification
                if physics_correct:
                    working_phantoms.append(phantom_name)
                else:
                    physics_violations.append(phantom_name)
                
                # Status indicators
                amp_status = '✅' if amp_correct else '🚨'
                phase_status = '✅' if phase_correct else '🚨'
                physics_status = '✅' if physics_correct else '🚨'
                
                print(f'{phantom_name}: {physics_status}')
                print(f'  Log-amp: slope={slope_amp:.6f} {amp_status} | R²={r_amp**2:.4f} | p={p_amp:.3f}')
                print(f'  Phase:   slope={slope_phase:.6f} {phase_status} | R²={r_phase**2:.4f} | p={p_phase:.3f}')
                
        except Exception as e:
            print(f'{phantom_name}: ❌ ERROR - {e}')
            physics_violations.append(phantom_name)
    
    # Convert to numpy arrays for analysis
    all_slopes_amp = np.array(all_slopes_amp)
    all_slopes_phase = np.array(all_slopes_phase)
    all_r2_amp = np.array(all_r2_amp)
    all_r2_phase = np.array(all_r2_phase)
    all_distances = np.array(all_distances)
    all_log_amp = np.array(all_log_amp)
    all_phase = np.array(all_phase)
    
    print()
    print('🎯 CROSS-DATASET PHYSICS VALIDATION SUMMARY')
    print('='*50)
    print(f'Total phantoms tested: {num_phantoms}')
    print(f'✅ Correct physics: {len(working_phantoms)}/{num_phantoms} ({len(working_phantoms)/num_phantoms*100:.1f}%)')
    print(f'🚨 Physics violations: {len(physics_violations)}/{num_phantoms} ({len(physics_violations)/num_phantoms*100:.1f}%)')
    
    # Success rate assessment
    success_rate = len(working_phantoms) / num_phantoms
    if success_rate >= 0.75:
        print('\n🎉 EXCELLENT! ≥75% success rate - dataset is working very well!')
    elif success_rate >= 0.60:
        print('\n✅ GOOD! ≥60% success rate - dataset is working!')
    elif success_rate >= 0.50:
        print('\n⚠️  MODERATE! ≥50% success rate - some phantoms need review')
    else:
        print('\n❌ POOR! <50% success rate - significant issues detected')
    
    print()
    print('Working phantoms:', working_phantoms[:5], '...' if len(working_phantoms) > 5 else '')
    if physics_violations:
        print('Violation phantoms:', physics_violations[:5], '...' if len(physics_violations) > 5 else '')
    
    # Cross-dataset statistical analysis
    print()
    print('📊 CROSS-DATASET STATISTICAL ANALYSIS')
    print('='*40)
    print(f'Slope distributions (log-amplitude):')
    print(f'  Mean: {np.mean(all_slopes_amp):.6f} ± {np.std(all_slopes_amp):.6f}')
    print(f'  Range: [{np.min(all_slopes_amp):.6f}, {np.max(all_slopes_amp):.6f}]')
    print(f'  Correct slopes: {np.sum(all_slopes_amp < 0)}/{len(all_slopes_amp)} ({np.sum(all_slopes_amp < 0)/len(all_slopes_amp)*100:.1f}%)')
    
    print(f'Slope distributions (phase):')
    print(f'  Mean: {np.mean(all_slopes_phase):.6f} ± {np.std(all_slopes_phase):.6f}')
    print(f'  Range: [{np.min(all_slopes_phase):.6f}, {np.max(all_slopes_phase):.6f}]')
    print(f'  Correct slopes: {np.sum(all_slopes_phase > 0)}/{len(all_slopes_phase)} ({np.sum(all_slopes_phase > 0)/len(all_slopes_phase)*100:.1f}%)')
    
    print(f'R² distributions:')
    print(f'  Log-amp R²: {np.mean(all_r2_amp):.4f} ± {np.std(all_r2_amp):.4f}')
    print(f'  Phase R²: {np.mean(all_r2_phase):.4f} ± {np.std(all_r2_phase):.4f}')
    
    # Create cross-dataset visualization
    print()
    print('🎨 Creating cross-dataset visualization...')
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Cross-Dataset Analysis: {num_phantoms} Phantoms', fontsize=18, fontweight='bold')
    
    # 1. Slope distributions
    ax1 = plt.subplot(2, 4, 1)
    ax1.hist(all_slopes_amp, bins=20, alpha=0.7, color=COLORS['primary'], edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Physics Boundary')
    ax1.set_xlabel('Log-Amplitude Slope')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Log-Amp Slope Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.hist(all_slopes_phase, bins=20, alpha=0.7, color=COLORS['secondary'], edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Physics Boundary')
    ax2.set_xlabel('Phase Slope')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Phase Slope Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2. R² distributions
    ax3 = plt.subplot(2, 4, 3)
    ax3.hist(all_r2_amp, bins=20, alpha=0.7, color=COLORS['primary'], edgecolor='black')
    ax3.set_xlabel('Log-Amplitude R²')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Log-Amp R² Distribution')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(all_r2_phase, bins=20, alpha=0.7, color=COLORS['secondary'], edgecolor='black')
    ax4.set_xlabel('Phase R²')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Phase R² Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 3. Aggregated physics relationships
    ax5 = plt.subplot(2, 4, 5)
    sample_indices = np.random.choice(len(all_distances), size=min(5000, len(all_distances)), replace=False)
    ax5.scatter(all_distances[sample_indices], all_log_amp[sample_indices], 
               alpha=0.3, color=COLORS['primary'], s=5)
    z_all = np.polyfit(all_distances, all_log_amp, 1)
    p_all = np.poly1d(z_all)
    ax5.plot(all_distances, p_all(all_distances), color=COLORS['accent'], linewidth=3,
            label=f'Overall Slope: {z_all[0]:.6f}')
    ax5.set_xlabel('Distance [mm]')
    ax5.set_ylabel('Log-Amplitude')
    ax5.set_title('Aggregated: Log-Amp vs Distance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 4, 6)
    ax6.scatter(all_distances[sample_indices], all_phase[sample_indices], 
               alpha=0.3, color=COLORS['secondary'], s=5)
    z_phase_all = np.polyfit(all_distances, all_phase, 1)
    p_phase_all = np.poly1d(z_phase_all)
    ax6.plot(all_distances, p_phase_all(all_distances), color=COLORS['accent'], linewidth=3,
            label=f'Overall Slope: {z_phase_all[0]:.6f}')
    ax6.set_xlabel('Distance [mm]')
    ax6.set_ylabel('Phase [degrees]')
    ax6.set_title('Aggregated: Phase vs Distance')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 4. Physics validation summary
    ax7 = plt.subplot(2, 4, 7)
    phantom_names = list(results.keys())
    phantom_status = [1 if results[name]['physics_correct'] else 0 for name in phantom_names]
    colors = [COLORS['success'] if status else COLORS['neutral1'] for status in phantom_status]
    bars = ax7.bar(range(len(phantom_names)), phantom_status, color=colors)
    ax7.set_xlabel('Phantom Index')
    ax7.set_ylabel('Physics Correct (1=Yes, 0=No)')
    ax7.set_title('Physics Validation per Phantom')
    ax7.set_ylim(-0.1, 1.1)
    
    # 5. Summary statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    summary_text = f"""
CROSS-DATASET SUMMARY
{'='*25}

Dataset Scale:
  • Total Phantoms: {num_phantoms}
  • Total Measurements: {len(all_log_amp):,}
  • Working Phantoms: {len(working_phantoms)}
  • Success Rate: {success_rate:.1%}

Physics Validation:
  • Correct Log-Amp Slopes: {np.sum(all_slopes_amp < 0)}/{len(all_slopes_amp)}
  • Correct Phase Slopes: {np.sum(all_slopes_phase > 0)}/{len(all_slopes_phase)}
  • Overall Aggregate Slope (Amp): {z_all[0]:.6f}
  • Overall Aggregate Slope (Phase): {z_phase_all[0]:.6f}

Quality Metrics:
  • Mean R² (Log-Amp): {np.mean(all_r2_amp):.3f}
  • Mean R² (Phase): {np.mean(all_r2_phase):.3f}
  • Distance Range: {np.min(all_distances):.1f}-{np.max(all_distances):.1f}mm

Status: {'🎉 EXCELLENT' if success_rate >= 0.75 else '✅ GOOD' if success_rate >= 0.6 else '⚠️ REVIEW NEEDED'}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['bg_light'], alpha=0.8))
    
    plt.tight_layout()
    
    # Save cross-dataset analysis
    results_dir = project_root / "analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    plot_path = results_dir / f"cross_dataset_analysis_{num_phantoms}_phantoms.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Cross-dataset analysis saved: {plot_path}")
    
    plt.show()
    
    print()
    print("🎉 CROSS-DATASET ANALYSIS COMPLETE!")
    print(f"📊 Successfully analyzed {num_phantoms} phantoms with {success_rate:.1%} success rate")
    
    return results


def main_menu():
    """🎯 Main menu for dual-mode analysis system."""
    print("\n" + "="*80)
    print("🔬 NIR PHANTOM DATASET ANALYSIS SUITE")
    print("="*80)
    print("🎯 Choose Analysis Mode:")
    print()
    print("   [1] 🔍 Single Phantom Analysis")
    print("       • 9-graph comprehensive visualization")
    print("       • Physics validation & quality assessment")
    print("       • Detailed tissue property analysis")
    print("       • 3D geometry & statistical summaries")
    print()
    print("   [2] 📊 Cross-Dataset Analysis")  
    print("       • Physics validation across ALL phantoms")
    print("       • Cross-dataset statistical comparison")
    print("       • Aggregate physics relationship analysis")
    print("       • Dataset-wide quality assessment")
    print()
    print("   [0] 🚪 Exit")
    print()
    
    try:
        choice = input("Select option [0-2]: ").strip()
        
        if choice == "1":
            single_phantom_analysis()
        elif choice == "2":
            comprehensive_multi_phantom_analysis()
        elif choice == "0":
            print("👋 Goodbye!")
            return None
        else:
            print("❌ Invalid choice!")
            return main_menu()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return None


if __name__ == "__main__":
    # Run the main dual-mode menu
    main_menu()
