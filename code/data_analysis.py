#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE NIR PHANTOM DATASET ANALYSIS SUITE 🔬

A complete, all-in-one analysis toolkit for NIR phantom datasets featuring:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 COMPREHENSIVE ANALYSIS CAPABILITIES:
• Complete HDF5 structure and metadata examination
• Advanced statistical analysis with distribution fitting
• 3D geometric configuration visualization
• Ground truth optical property validation
• Cross-dataset consistency analysis
• Publication-quality scientific visualizations
• Interactive reporting with professional plots

🎨 VISUALIZATION FEATURES:
• Professional scientific plotting themes
• 3D interactive visualizations
• Heat maps and correlation matrices
• Statistical distribution plots
• Comparative analysis charts
• High-resolution output for publications

🔍 QUALITY ASSURANCE:
• Comprehensive data validation
• Outlier detection and analysis
• Consistency checks across datasets
• Physical plausibility verification
• Missing data identification

📈 STATISTICAL ANALYSIS:
• Distribution fitting and normality tests
• Correlation analysis and PCA
• Outlier detection using multiple methods
• Confidence intervals and hypothesis testing
• Advanced statistical summaries

Author: Max Hart - NIR Tomography Research
Version: 2.0 - Complete Analysis Suite
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

# Configure matplotlib for ultra-high-quality scientific plots
plt.style.use('dark_background')  # Dark theme for modern look
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
    'grid.alpha': 0.3,
    'grid.color': '#404040',
    'lines.linewidth': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.facecolor': '#1a1a1a',
    'figure.facecolor': '#0d1117',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#404040'
})

# Define professional color schemes for dark theme
COLORS = {
    'primary': '#00d4ff',      # Bright cyan
    'secondary': '#ff6b9d',    # Hot pink
    'accent': '#ffa726',       # Bright orange
    'success': '#4caf50',      # Green
    'warning': '#ff9800',      # Orange
    'error': '#f44336',        # Red
    'tissue': '#00e676',       # Bright green
    'tumor': '#ff1744',        # Bright red
    'air': '#2196f3',          # Blue
    'source': '#ffd700',       # Gold
    'detector': '#00ffff',     # Cyan
    'gradient1': '#667eea',    # Purple-blue
    'gradient2': '#764ba2',    # Purple
    'bg_dark': '#1a1a1a',      # Dark background
    'bg_light': '#2d2d2d'      # Light background
}

# Enhanced color palettes for different plot types
MEASUREMENT_COLORS = ['#00d4ff', '#ff6b9d', '#ffa726', '#4caf50', '#9c27b0']
TISSUE_COLORS = ['#1a1a1a', '#00e676', '#ff1744', '#ffa726', '#2196f3', '#9c27b0']
GRADIENT_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']

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
    
    def __init__(self, data_directory="../data"):
        """
        Initialize the comprehensive analyzer.
        
        Args:
            data_directory (str): Path to directory containing phantom datasets
        """
        print("\n" + "="*80)
        print("🔬 INITIALIZING COMPREHENSIVE NIR DATASET ANALYZER 🔬")
        print("="*80)
        
        self.data_dir = Path(data_directory)
        self.phantom_files = []
        self.all_data = {}  # Store loaded datasets
        self.analysis_results = {}
        
        # Create output directory for visualizations
        self.output_dir = self.data_dir / "analysis_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._discover_datasets()
        
    def _discover_datasets(self):
        """🔍 Discover and validate all available phantom datasets."""
        print("\n� DATASET DISCOVERY")
        print("-" * 50)
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return
            
        # Find all phantom directories
        phantom_dirs = sorted([d for d in self.data_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("phantom_")])
        
        if not phantom_dirs:
            print(f"❌ No phantom directories found in {self.data_dir}")
            return
            
        print(f"� Scanning {len(phantom_dirs)} phantom directories...")
        
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
    
    def load_all_datasets(self):
        """📂 Load all datasets into memory for comprehensive analysis."""
        print("\n📂 LOADING ALL DATASETS INTO MEMORY")
        print("-" * 50)
        
        if not self.phantom_files:
            print("❌ No datasets available to load!")
            return False
        
        loading_start = time.time()
        successful_loads = 0
        
        for i, file_path in enumerate(self.phantom_files):
            phantom_name = file_path.parent.name
            print(f"   Loading {phantom_name:15} [{i+1:2d}/{len(self.phantom_files):2d}]", end=" ")
            
            try:
                with h5py.File(file_path, 'r') as f:
                    data = {
                        'phantom_name': phantom_name,
                        'file_path': file_path,
                        'log_amplitude': f['log_amplitude'][:],
                        'phase': f['phase'][:],
                        'ground_truth': f['ground_truth'][:],
                        'source_pos': f['source_pos'][:],
                        'det_pos': f['det_pos'][:],
                        'file_size_mb': file_path.stat().st_size / (1024**2),
                        'datasets': list(f.keys()),
                        'attributes': dict(f.attrs) if f.attrs else {}
                    }
                    
                    # Add derived statistics
                    data['n_measurements'] = data['log_amplitude'].size
                    data['volume_shape'] = data['ground_truth'].shape[:3]
                    data['n_sources'] = len(data['source_pos'])
                    
                    self.all_data[phantom_name] = data
                    successful_loads += 1
                    print("✅")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
        
        loading_time = time.time() - loading_start
        
        print(f"\n📊 LOADING SUMMARY:")
        print(f"   Datasets attempted: {len(self.phantom_files)}")
        print(f"   Successfully loaded: {successful_loads}")
        print(f"   Loading time: {loading_time:.2f} seconds")
        print(f"   Average time per dataset: {loading_time/len(self.phantom_files):.2f}s")
        
        return successful_loads > 0
    
    def comprehensive_analysis(self):
        """🔬 Perform comprehensive analysis of all datasets."""
        print("\n🔬 COMPREHENSIVE DATASET ANALYSIS")
        print("-" * 50)
        
        if not self.all_data:
            print("❌ No data loaded! Run load_all_datasets() first.")
            return
        
        analysis_start = time.time()
        
        # Perform different types of analysis
        print("🧮 Performing statistical analysis...")
        self._statistical_analysis()
        
        print("🎯 Analyzing ground truth properties...")
        self._ground_truth_analysis()
        
        print("📐 Examining geometric configurations...")
        self._geometric_analysis()
        
        print("🔍 Running quality validation...")
        self._quality_validation()
        
        print("🔄 Checking cross-dataset consistency...")
        self._consistency_analysis()
        
        analysis_time = time.time() - analysis_start
        print(f"\n✅ Comprehensive analysis completed in {analysis_time:.2f} seconds")
    
    def _statistical_analysis(self):
        """📊 Advanced statistical analysis of measurements."""
        results = {}
        
        all_log_amp = []
        all_phase = []
        
        for phantom_name, data in self.all_data.items():
            log_amp = data['log_amplitude'].flatten()
            phase = data['phase'].flatten()
            
            # Remove invalid values
            log_amp = log_amp[np.isfinite(log_amp)]
            phase = phase[np.isfinite(phase)]
            
            if len(log_amp) > 0 and len(phase) > 0:
                all_log_amp.extend(log_amp)
                all_phase.extend(phase)
                
                # Per-phantom statistics
                results[phantom_name] = {
                    'log_amplitude': {
                        'count': len(log_amp),
                        'mean': np.mean(log_amp),
                        'std': np.std(log_amp),
                        'min': np.min(log_amp),
                        'max': np.max(log_amp),
                        'median': np.median(log_amp),
                        'q25': np.percentile(log_amp, 25),
                        'q75': np.percentile(log_amp, 75),
                        'skewness': stats.skew(log_amp),
                        'kurtosis': stats.kurtosis(log_amp)
                    },
                    'phase': {
                        'count': len(phase),
                        'mean': np.mean(phase),
                        'std': np.std(phase),
                        'min': np.min(phase),
                        'max': np.max(phase),
                        'median': np.median(phase),
                        'q25': np.percentile(phase, 25),
                        'q75': np.percentile(phase, 75),
                        'skewness': stats.skew(phase),
                        'kurtosis': stats.kurtosis(phase)
                    }
                }
                
                # Normality tests (sample for speed)
                sample_size = min(5000, len(log_amp))
                if sample_size > 50:
                    sample_idx = np.random.choice(len(log_amp), sample_size, replace=False)
                    _, p_normal_amp = stats.shapiro(log_amp[sample_idx])
                    _, p_normal_phase = stats.shapiro(phase[sample_idx])
                    
                    results[phantom_name]['log_amplitude']['normality_p'] = p_normal_amp
                    results[phantom_name]['phase']['normality_p'] = p_normal_phase
        
        # Global statistics
        if all_log_amp and all_phase:
            all_log_amp = np.array(all_log_amp)
            all_phase = np.array(all_phase)
            
            # Correlation analysis
            correlation = np.corrcoef(all_log_amp, all_phase)[0, 1]
            
            results['global'] = {
                'total_measurements': len(all_log_amp),
                'log_amp_range': [float(np.min(all_log_amp)), float(np.max(all_log_amp))],
                'phase_range': [float(np.min(all_phase)), float(np.max(all_phase))],
                'correlation': float(correlation)
            }
        
        self.analysis_results['statistics'] = results
    
    def _ground_truth_analysis(self):
        """🎯 Analyze ground truth optical property distributions."""
        results = {}
        
        all_mua = []
        all_musp = []
        tissue_type_counts = {}
        
        for phantom_name, data in self.all_data.items():
            gt = data['ground_truth']
            mua_map = gt[..., 0]
            musp_map = gt[..., 1]
            
            # Overall statistics
            results[phantom_name] = {
                'volume_shape': list(gt.shape[:3]),
                'total_voxels': int(mua_map.size),
                'mua_stats': {
                    'min': float(np.min(mua_map)),
                    'max': float(np.max(mua_map)),
                    'mean': float(np.mean(mua_map)),
                    'std': float(np.std(mua_map)),
                    'nonzero_count': int(np.sum(mua_map > 0))
                },
                'musp_stats': {
                    'min': float(np.min(musp_map)),
                    'max': float(np.max(musp_map)),
                    'mean': float(np.mean(musp_map)),
                    'std': float(np.std(musp_map)),
                    'nonzero_count': int(np.sum(musp_map > 0))
                }
            }
            
            # Tissue type analysis
            unique_combinations = np.unique(gt.reshape(-1, 2), axis=0)
            tissue_types = []
            
            for mua_val, musp_val in unique_combinations:
                voxel_count = np.sum((mua_map == mua_val) & (musp_map == musp_val))
                
                if mua_val == 0 and musp_val == 0:
                    tissue_type = 'air'
                else:
                    tissue_type = f'tissue_{len(tissue_types)}'
                
                tissue_types.append({
                    'type': tissue_type,
                    'mua': float(mua_val),
                    'musp': float(musp_val),
                    'voxel_count': int(voxel_count),
                    'percentage': float(voxel_count / mua_map.size * 100)
                })
                
                # Collect for global analysis
                if mua_val > 0 or musp_val > 0:  # Non-air voxels
                    all_mua.extend([mua_val] * voxel_count)
                    all_musp.extend([musp_val] * voxel_count)
            
            results[phantom_name]['tissue_types'] = tissue_types
            results[phantom_name]['n_tissue_types'] = len([t for t in tissue_types if t['type'] != 'air'])
            
            # Add to global counts
            n_types = results[phantom_name]['n_tissue_types']
            tissue_type_counts[n_types] = tissue_type_counts.get(n_types, 0) + 1
        
        # Global ground truth statistics
        if all_mua and all_musp:
            results['global'] = {
                'mua_global_range': [float(np.min(all_mua)), float(np.max(all_mua))],
                'musp_global_range': [float(np.min(all_musp)), float(np.max(all_musp))],
                'tissue_type_distribution': tissue_type_counts,
                'total_tissue_voxels': len(all_mua)
            }
        
        self.analysis_results['ground_truth'] = results
    
    def _geometric_analysis(self):
        """📐 Analyze geometric configurations and probe layouts."""
        results = {}
        
        all_distances = []
        
        for phantom_name, data in self.all_data.items():
            source_pos = data['source_pos']
            det_pos = data['det_pos']
            
            # Basic geometry stats
            results[phantom_name] = {
                'n_sources': len(source_pos),
                'source_bounds': {
                    'x': [float(source_pos[:, 0].min()), float(source_pos[:, 0].max())],
                    'y': [float(source_pos[:, 1].min()), float(source_pos[:, 1].max())],
                    'z': [float(source_pos[:, 2].min()), float(source_pos[:, 2].max())]
                }
            }
            
            # Detector analysis
            if len(det_pos.shape) == 3:  # (n_sources, n_dets, 3)
                results[phantom_name]['n_detectors_per_source'] = det_pos.shape[1]
                results[phantom_name]['total_detectors'] = det_pos.shape[0] * det_pos.shape[1]
                
                # Calculate all source-detector distances
                distances = []
                for i in range(len(source_pos)):
                    src = source_pos[i]
                    dets = det_pos[i]
                    for det in dets:
                        dist = np.linalg.norm(det - src)
                        distances.append(dist)
                        all_distances.append(dist)
                
                if distances:
                    distances = np.array(distances)
                    results[phantom_name]['distance_stats'] = {
                        'min': float(distances.min()),
                        'max': float(distances.max()),
                        'mean': float(distances.mean()),
                        'std': float(distances.std()),
                        'median': float(np.median(distances))
                    }
            
            # Source distribution analysis
            if len(source_pos) > 1:
                source_distances = pdist(source_pos)
                results[phantom_name]['source_separation'] = {
                    'min': float(source_distances.min()),
                    'max': float(source_distances.max()),
                    'mean': float(source_distances.mean())
                }
        
        # Global geometry statistics
        if all_distances:
            all_distances = np.array(all_distances)
            results['global'] = {
                'distance_range': [float(all_distances.min()), float(all_distances.max())],
                'distance_mean': float(all_distances.mean()),
                'distance_std': float(all_distances.std()),
                'total_measurements': len(all_distances)
            }
        
        self.analysis_results['geometry'] = results
    
    def _quality_validation(self):
        """🔍 Comprehensive data quality validation."""
        results = {}
        
        for phantom_name, data in self.all_data.items():
            issues = []
            warnings = []
            
            log_amp = data['log_amplitude']
            phase = data['phase']
            gt = data['ground_truth']
            
            # Check for invalid values
            if np.any(np.isnan(log_amp)) or np.any(np.isnan(phase)):
                issues.append("NaN values in measurements")
            
            if np.any(np.isinf(log_amp)) or np.any(np.isinf(phase)):
                issues.append("Infinite values in measurements")
            
            if np.any(gt < 0):
                issues.append("Negative optical properties")
            
            # Range checks
            if np.min(log_amp[np.isfinite(log_amp)]) < -25:
                warnings.append(f"Very low log-amplitude values: {np.min(log_amp[np.isfinite(log_amp)]):.3f}")
            
            if np.max(np.abs(phase[np.isfinite(phase)])) > 5000:
                warnings.append(f"Extreme phase values: max={np.max(np.abs(phase[np.isfinite(phase)])):.1f}°")
            
            # Measurement count consistency
            expected_measurements = data['n_sources'] * (data['det_pos'].shape[1] if len(data['det_pos'].shape) > 1 else 1)
            actual_measurements = data['n_measurements']
            
            if expected_measurements != actual_measurements:
                issues.append(f"Measurement count mismatch: expected {expected_measurements}, got {actual_measurements}")
            
            results[phantom_name] = {
                'issues': issues,
                'warnings': warnings,
                'is_valid': len(issues) == 0,
                'quality_score': max(0, 100 - len(issues)*30 - len(warnings)*10)
            }
        
        # Global quality summary
        valid_datasets = sum(1 for r in results.values() if r['is_valid'])
        avg_quality = np.mean([r['quality_score'] for r in results.values()])
        
        results['global'] = {
            'total_datasets': len(results) - 1,  # Exclude 'global'
            'valid_datasets': valid_datasets,
            'average_quality_score': avg_quality,
            'success_rate': valid_datasets / (len(results) - 1) * 100
        }
        
        self.analysis_results['quality'] = results
    
    def _consistency_analysis(self):
        """🔄 Cross-dataset consistency analysis."""
        results = {}
        
        if len(self.all_data) < 2:
            results['status'] = 'insufficient_data'
            self.analysis_results['consistency'] = results
            return
        
        # Check volume shape consistency
        shapes = [data['volume_shape'] for data in self.all_data.values()]
        unique_shapes = list(set(tuple(s) for s in shapes))
        results['volume_shapes'] = {
            'consistent': len(unique_shapes) == 1,
            'unique_shapes': unique_shapes,
            'most_common': max(set(tuple(s) for s in shapes), key=shapes.count) if shapes else None
        }
        
        # Check measurement count consistency
        measurement_counts = [data['n_measurements'] for data in self.all_data.values()]
        results['measurement_counts'] = {
            'consistent': len(set(measurement_counts)) == 1,
            'range': [min(measurement_counts), max(measurement_counts)] if measurement_counts else [0, 0],
            'most_common': max(set(measurement_counts), key=measurement_counts.count) if measurement_counts else 0
        }
        
        # Check source count consistency
        source_counts = [data['n_sources'] for data in self.all_data.values()]
        results['source_counts'] = {
            'consistent': len(set(source_counts)) == 1,
            'range': [min(source_counts), max(source_counts)] if source_counts else [0, 0]
        }
        
        # Calculate consistency score
        consistency_score = 0
        if results['volume_shapes']['consistent']:
            consistency_score += 40
        if results['measurement_counts']['consistent']:
            consistency_score += 30
        if results['source_counts']['consistent']:
            consistency_score += 30
        
        results['overall_consistency_score'] = consistency_score
        results['status'] = 'completed'
        
        self.analysis_results['consistency'] = results
    
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
        print(f"   Total voxels: {gt_data.shape[0] * gt_data.shape[1] * gt_data.shape[2]:,}")
        print(f"   Property channels: {gt_data.shape[3]} (μₐ, μ′s)")
        
        # Extract absorption and scattering maps
        mua_map = gt_data[..., 0]  # Absorption coefficient
        musp_map = gt_data[..., 1]  # Reduced scattering coefficient
        
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
        unique_combinations = np.unique(gt_data.reshape(-1, 2), axis=0)
        tissue_types = len(unique_combinations) - 1  # Subtract air (0,0)
        
        print(f"   Tissue types identified: {tissue_types}")
        print(f"   Property combinations:")
        for i, (mua, musp) in enumerate(unique_combinations):
            if mua == 0 and musp == 0:
                print(f"     Type {i}: Air (μₐ={mua:.6f}, μ′s={musp:.3f})")
            else:
                voxel_count = np.sum((mua_map == mua) & (musp_map == musp))
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
        
        source_pos = hdf5_file['source_pos'][:]
        det_pos = hdf5_file['det_pos'][:]
        
        print(f"📐 Geometric Configuration:")
        print(f"   Source positions shape: {source_pos.shape}")
        print(f"   Detector positions shape: {det_pos.shape}")
        print(f"   Number of probes: {len(source_pos)}")
        print(f"   Detectors per probe: {det_pos.shape[1] if len(det_pos.shape) > 1 else 'N/A'}")
        
        # Calculate position statistics
        print(f"\n📊 Source Position Statistics [mm]:")
        for axis, name in enumerate(['X', 'Y', 'Z']):
            coords = source_pos[:, axis]
            print(f"   {name}-axis: [{coords.min():.1f}, {coords.max():.1f}] "
                  f"(mean: {coords.mean():.1f}, std: {coords.std():.1f})")
        
        # Calculate source-detector distances
        if len(det_pos.shape) == 3:  # (n_probes, n_detectors, 3)
            print(f"\n📏 Source-Detector Distances:")
            distances = []
            for i in range(len(source_pos)):
                src = source_pos[i]
                dets = det_pos[i]
                probe_distances = [np.linalg.norm(det - src) for det in dets]
                distances.extend(probe_distances)
                
                if i < 5:  # Show first 5 probes
                    dist_str = ", ".join([f"{d:.1f}mm" for d in probe_distances])
                    print(f"   Probe {i+1}: {dist_str}")
            
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
            'n_detectors': det_pos.shape[1] if len(det_pos.shape) > 1 else 1,
            'source_bounds': {
                'x': [float(source_pos[:, 0].min()), float(source_pos[:, 0].max())],
                'y': [float(source_pos[:, 1].min()), float(source_pos[:, 1].max())],
                'z': [float(source_pos[:, 2].min()), float(source_pos[:, 2].max())]
            }
        }
        
        if len(det_pos.shape) == 3:
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
        source_pos = hdf5_file['source_pos'][:]
        det_pos = hdf5_file['det_pos'][:]
        
        # Shape consistency
        if log_amp.shape != phase.shape:
            issues.append(f"Measurement shape mismatch: log_amp {log_amp.shape} vs phase {phase.shape}")
        
        expected_measurements = len(source_pos) * (det_pos.shape[1] if len(det_pos.shape) > 1 else 1)
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
        
        # Correlation analysis
        correlation = np.corrcoef(log_amp_flat, phase_flat)[0, 1]
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
    
    def analyze_all_datasets(self):
        """Analyze all discovered datasets and generate comparative report."""
        
        if not self.phantom_files:
            print("❌ No datasets found to analyze!")
            return
        
        print(f"\n🔍 ANALYZING ALL {len(self.phantom_files)} DATASETS")
        print("="*80)
        
        all_results = []
        
        for i, file_path in enumerate(self.phantom_files):
            print(f"\n[{i+1}/{len(self.phantom_files)}]", end=" ")
            results = self.analyze_single_dataset(file_path, detailed=False)
            all_results.append(results)
        
        # Generate comparative summary
        self._generate_comparative_summary(all_results)
        
        return all_results
    
    def _generate_comparative_summary(self, all_results):
        """Generate comparative summary across all datasets."""
        
        print(f"\n📊 COMPARATIVE DATASET SUMMARY")
        print("="*60)
        
        valid_results = [r for r in all_results if 'error' not in r]
        
        if not valid_results:
            print("❌ No valid datasets for comparison!")
            return
        
        # File sizes
        file_sizes = [r['file_size_mb'] for r in valid_results]
        print(f"📁 File Sizes:")
        print(f"   Range: {min(file_sizes):.1f} - {max(file_sizes):.1f} MB")
        print(f"   Mean: {np.mean(file_sizes):.1f} MB")
        
        # Measurement statistics
        print(f"\n📏 Measurement Consistency:")
        log_amp_ranges = [(r['measurements']['log_amplitude']['min'], 
                          r['measurements']['log_amplitude']['max']) for r in valid_results]
        phase_ranges = [(r['measurements']['phase']['min'], 
                        r['measurements']['phase']['max']) for r in valid_results]
        
        print(f"   Log-amplitude ranges: {len(set(log_amp_ranges))} unique")
        print(f"   Phase ranges: {len(set(phase_ranges))} unique")
        
        # Ground truth diversity
        print(f"\n🎯 Ground Truth Diversity:")
        tissue_type_counts = [r['ground_truth']['tissue_distribution']['tissue_types'] 
                             for r in valid_results]
        print(f"   Tissue types per phantom: {min(tissue_type_counts)} - {max(tissue_type_counts)}")
        
        # Data quality summary
        print(f"\n✅ Data Quality Summary:")
        quality_passed = sum(1 for r in valid_results if r['data_quality']['passed'])
        print(f"   Datasets passing all checks: {quality_passed}/{len(valid_results)}")
        
        if quality_passed < len(valid_results):
            print(f"   Datasets with issues: {len(valid_results) - quality_passed}")
    
    def visualize_dataset(self, file_path, save_plots=True):
        """Create SPECTACULAR comprehensive visualizations for a single dataset with enhanced styling."""
        
        print(f"\n🎨 GENERATING SPECTACULAR VISUALIZATIONS: {file_path.parent.name}")
        print("="*70)
        
        with h5py.File(file_path, 'r') as f:
            # Load data
            log_amp = f['log_amplitude'][:]
            phase = f['phase'][:]
            source_pos = f['source_pos'][:]
            det_pos = f['det_pos'][:]
            gt_data = f['ground_truth'][:]
            
            # Calculate source-detector distances for SDS analysis
            distances = []
            log_amp_flat = []
            phase_flat = []
            
            if len(det_pos.shape) == 3:
                for i in range(len(source_pos)):
                    src = source_pos[i]
                    dets = det_pos[i]
                    for j, det in enumerate(dets):
                        dist = np.linalg.norm(det - src)
                        distances.append(dist)
                        log_amp_flat.append(log_amp[i, j])
                        phase_flat.append(phase[i, j])
            
            distances = np.array(distances)
            log_amp_flat = np.array(log_amp_flat)
            phase_flat = np.array(phase_flat)
            
            # Create figure with enhanced layout
            fig = plt.figure(figsize=(24, 18), facecolor='#0d1117')
            fig.suptitle(f'🔬 ADVANCED NIR DATASET ANALYSIS: {file_path.parent.name}', 
                        fontsize=20, fontweight='bold', color='white', y=0.98)
            
            # Add subtitle with key metrics
            n_measurements = log_amp.size
            n_probes = len(source_pos)
            distance_range = f"{distances.min():.1f}-{distances.max():.1f}mm" if len(distances) > 0 else "N/A"
            
            fig.text(0.5, 0.95, f'📊 {n_measurements:,} measurements • {n_probes} probes • SDS range: {distance_range}', 
                    ha='center', fontsize=12, color='#888888', style='italic')
            
            # ═══════════════════════════════════════════════════════════════
            # ROW 1: ENHANCED MEASUREMENT DISTRIBUTIONS
            # ═══════════════════════════════════════════════════════════════
            
            # 1. SPECTACULAR Log-Amplitude Distribution with KDE
            ax1 = plt.subplot(4, 5, 1)
            n, bins, patches = plt.hist(log_amp.flatten(), bins=60, alpha=0.7, 
                                       color=COLORS['primary'], edgecolor='white', linewidth=0.5)
            
            # Add KDE overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(log_amp.flatten())
            x_kde = np.linspace(log_amp.min(), log_amp.max(), 200)
            kde_values = kde(x_kde)
            kde_scaled = kde_values * (np.max(n) / np.max(kde_values))
            plt.plot(x_kde, kde_scaled, color=COLORS['accent'], linewidth=3, label='KDE')
            
            plt.title('📈 Log-Amplitude Distribution', fontweight='bold', pad=15)
            plt.xlabel('Log-Amplitude', fontweight='bold')
            plt.ylabel('Frequency', fontweight='bold')
            plt.grid(True, alpha=0.3, color='#404040')
            plt.legend()
            
            # Add statistics text
            mean_val = np.mean(log_amp.flatten())
            std_val = np.std(log_amp.flatten())
            plt.text(0.02, 0.98, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}', 
                    transform=ax1.transAxes, va='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            
            # 2. SPECTACULAR Phase Distribution with KDE
            ax2 = plt.subplot(4, 5, 2)
            n2, bins2, patches2 = plt.hist(phase.flatten(), bins=60, alpha=0.7, 
                                          color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
            
            # Add KDE overlay for phase
            kde2 = gaussian_kde(phase.flatten())
            x_kde2 = np.linspace(phase.min(), phase.max(), 200)
            kde_values2 = kde2(x_kde2)
            kde_scaled2 = kde_values2 * (np.max(n2) / np.max(kde_values2))
            plt.plot(x_kde2, kde_scaled2, color=COLORS['accent'], linewidth=3, label='KDE')
            
            plt.title('📈 Phase Distribution', fontweight='bold', pad=15)
            plt.xlabel('Phase (degrees)', fontweight='bold')
            plt.ylabel('Frequency', fontweight='bold')
            plt.grid(True, alpha=0.3, color='#404040')
            plt.legend()
            
            # Add statistics text
            mean_phase = np.mean(phase.flatten())
            std_phase = np.std(phase.flatten())
            plt.text(0.02, 0.98, f'μ = {mean_phase:.1f}°\nσ = {std_phase:.1f}°', 
                    transform=ax2.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            
            # 3. ENHANCED Measurement Correlation with Density
            ax3 = plt.subplot(4, 5, 3)
            
            # Create 2D histogram for density visualization
            plt.hist2d(log_amp.flatten()[::5], phase.flatten()[::5], bins=50, 
                      cmap='plasma', alpha=0.8)
            
            # Add correlation line
            z = np.polyfit(log_amp.flatten(), phase.flatten(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(log_amp.min(), log_amp.max(), 100)
            plt.plot(x_line, p(x_line), color=COLORS['accent'], linewidth=3, 
                    linestyle='--', label=f'Trend (R²={np.corrcoef(log_amp.flatten(), phase.flatten())[0,1]**2:.3f})')
            
            plt.title('🎯 Log-Amplitude vs Phase Correlation', fontweight='bold', pad=15)
            plt.xlabel('Log-Amplitude', fontweight='bold')
            plt.ylabel('Phase (degrees)', fontweight='bold')
            plt.grid(True, alpha=0.3, color='#404040')
            plt.legend()
            cbar = plt.colorbar(label='Density')
            cbar.set_label('Measurement Density', fontweight='bold')
            
            # 4. NEW! Log-Amplitude vs SDS Analysis
            ax4 = plt.subplot(4, 5, 4)
            if len(distances) > 0:
                # Create binned analysis
                distance_bins = np.linspace(distances.min(), distances.max(), 20)
                bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
                bin_means = []
                bin_stds = []
                
                for i in range(len(distance_bins) - 1):
                    mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
                    if np.sum(mask) > 0:
                        bin_means.append(np.mean(log_amp_flat[mask]))
                        bin_stds.append(np.std(log_amp_flat[mask]))
                    else:
                        bin_means.append(np.nan)
                        bin_stds.append(np.nan)
                
                bin_means = np.array(bin_means)
                bin_stds = np.array(bin_stds)
                
                # Plot scatter with color coding
                scatter = plt.scatter(distances, log_amp_flat, c=phase_flat, 
                                    cmap='viridis', alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
                
                # Plot trend line with error bars
                valid_mask = ~np.isnan(bin_means)
                plt.errorbar(bin_centers[valid_mask], bin_means[valid_mask], 
                           yerr=bin_stds[valid_mask], color=COLORS['accent'], 
                           linewidth=3, capsize=5, label='Binned Mean ± SD')
                
                # Add exponential decay fit
                from scipy.optimize import curve_fit
                def exp_decay(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                try:
                    popt, _ = curve_fit(exp_decay, distances, log_amp_flat, 
                                      p0=[1, 0.1, -20], maxfev=1000)
                    x_fit = np.linspace(distances.min(), distances.max(), 100)
                    y_fit = exp_decay(x_fit, *popt)
                    plt.plot(x_fit, y_fit, color=COLORS['error'], linewidth=3, 
                            linestyle='--', label='Exponential Fit')
                except:
                    pass
                
                plt.title('🚀 Log-Amplitude vs Source-Detector Separation', fontweight='bold', pad=15)
                plt.xlabel('SDS Distance (mm)', fontweight='bold')
                plt.ylabel('Log-Amplitude', fontweight='bold')
                plt.grid(True, alpha=0.3, color='#404040')
                plt.legend()
                
                cbar2 = plt.colorbar(scatter, label='Phase (°)')
                cbar2.set_label('Phase (degrees)', fontweight='bold')
            
            # 5. NEW! Phase vs SDS Analysis
            ax5 = plt.subplot(4, 5, 5)
            if len(distances) > 0:
                # Create binned analysis for phase
                phase_bin_means = []
                phase_bin_stds = []
                
                for i in range(len(distance_bins) - 1):
                    mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
                    if np.sum(mask) > 0:
                        phase_bin_means.append(np.mean(phase_flat[mask]))
                        phase_bin_stds.append(np.std(phase_flat[mask]))
                    else:
                        phase_bin_means.append(np.nan)
                        phase_bin_stds.append(np.nan)
                
                phase_bin_means = np.array(phase_bin_means)
                phase_bin_stds = np.array(phase_bin_stds)
                
                # Plot scatter with color coding by log-amplitude
                scatter2 = plt.scatter(distances, phase_flat, c=log_amp_flat, 
                                     cmap='plasma', alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
                
                # Plot trend line with error bars
                valid_mask2 = ~np.isnan(phase_bin_means)
                plt.errorbar(bin_centers[valid_mask2], phase_bin_means[valid_mask2], 
                           yerr=phase_bin_stds[valid_mask2], color=COLORS['success'], 
                           linewidth=3, capsize=5, label='Binned Mean ± SD')
                
                # Add linear fit for phase
                z_phase = np.polyfit(distances, phase_flat, 1)
                p_phase = np.poly1d(z_phase)
                x_line_phase = np.linspace(distances.min(), distances.max(), 100)
                plt.plot(x_line_phase, p_phase(x_line_phase), color=COLORS['warning'], 
                        linewidth=3, linestyle='--', 
                        label=f'Linear Fit (slope={z_phase[0]:.2f}°/mm)')
                
                plt.title('🚀 Phase vs Source-Detector Separation', fontweight='bold', pad=15)
                plt.xlabel('SDS Distance (mm)', fontweight='bold')
                plt.ylabel('Phase (degrees)', fontweight='bold')
                plt.grid(True, alpha=0.3, color='#404040')
                plt.legend()
                
                cbar3 = plt.colorbar(scatter2, label='Log-Amp')
                cbar3.set_label('Log-Amplitude', fontweight='bold')
            
            # ═══════════════════════════════════════════════════════════════
            # ROW 2: SPECTACULAR 3D GEOMETRY AND GROUND TRUTH
            # ═══════════════════════════════════════════════════════════════
            
            # 6. ENHANCED 3D Probe Geometry
            ax6 = plt.subplot(4, 5, 6, projection='3d')
            ax6.set_facecolor('#1a1a1a')
            
            # Sources with enhanced styling
            source_scatter = ax6.scatter(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2], 
                                       c=COLORS['source'], s=60, alpha=0.9, 
                                       edgecolors='black', linewidth=1, label='Sources')
            
            # Detectors with enhanced styling
            if len(det_pos.shape) == 3:
                det_flat = det_pos.reshape(-1, 3)
                det_colors = plt.cm.viridis(np.linspace(0, 1, len(det_flat)))
                det_scatter = ax6.scatter(det_flat[:, 0], det_flat[:, 1], det_flat[:, 2], 
                                        c=det_colors, s=25, alpha=0.7, 
                                        edgecolors='white', linewidth=0.5, label='Detectors')
                
                # Add connection lines for first few probes
                for i in range(min(5, len(source_pos))):
                    src = source_pos[i]
                    dets = det_pos[i]
                    for det in dets:
                        ax6.plot([src[0], det[0]], [src[1], det[1]], [src[2], det[2]], 
                               color=COLORS['primary'], alpha=0.3, linewidth=1)
            
            ax6.set_title('🌐 3D Probe Geometry Network', fontweight='bold', pad=20)
            ax6.set_xlabel('X (mm)', fontweight='bold')
            ax6.set_ylabel('Y (mm)', fontweight='bold')
            ax6.set_zlabel('Z (mm)', fontweight='bold')
            ax6.legend()
            
            # Style the 3D plot
            ax6.xaxis.pane.fill = False
            ax6.yaxis.pane.fill = False
            ax6.zaxis.pane.fill = False
            ax6.grid(True, alpha=0.3)
            
            # 7-8. SPECTACULAR Ground Truth Maps
            mua_map = gt_data[..., 0]
            musp_map = gt_data[..., 1]
            z_center = mua_map.shape[2] // 2
            
            # Enhanced μₐ map
            ax7 = plt.subplot(4, 5, 7)
            im7 = plt.imshow(mua_map[:, :, z_center], cmap='magma', aspect='equal', 
                           interpolation='bilinear')
            plt.title(f'🎯 Absorption Map μₐ (Z={z_center})', fontweight='bold', pad=15)
            cbar7 = plt.colorbar(im7, label='μₐ (mm⁻¹)', shrink=0.8)
            cbar7.set_label('μₐ (mm⁻¹)', fontweight='bold')
            plt.xlabel('X (voxels)', fontweight='bold')
            plt.ylabel('Y (voxels)', fontweight='bold')
            
            # Enhanced μ′s map
            ax8 = plt.subplot(4, 5, 8)
            im8 = plt.imshow(musp_map[:, :, z_center], cmap='plasma', aspect='equal', 
                           interpolation='bilinear')
            plt.title(f'🎯 Scattering Map μ′s (Z={z_center})', fontweight='bold', pad=15)
            cbar8 = plt.colorbar(im8, label='μ′s (mm⁻¹)', shrink=0.8)
            cbar8.set_label('μ′s (mm⁻¹)', fontweight='bold')
            plt.xlabel('X (voxels)', fontweight='bold')
            plt.ylabel('Y (voxels)', fontweight='bold')
            
            # 9. SPECTACULAR Distance Distribution
            ax9 = plt.subplot(4, 5, 9)
            if len(distances) > 0:
                n_dist, bins_dist, patches_dist = plt.hist(distances, bins=30, alpha=0.8, 
                                                          color=COLORS['success'], 
                                                          edgecolor='white', linewidth=0.5)
                
                # Color gradient for bars
                for i, patch in enumerate(patches_dist):
                    patch.set_facecolor(plt.cm.viridis(i / len(patches_dist)))
                
                # Add vertical lines for constraints
                plt.axvline(x=10, color=COLORS['warning'], linestyle='--', linewidth=3, 
                           label='Min distance (10mm)', alpha=0.8)
                plt.axvline(x=40, color=COLORS['error'], linestyle='--', linewidth=3, 
                           label='Max distance (40mm)', alpha=0.8)
                
                # Add statistics
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                plt.axvline(x=mean_dist, color=COLORS['accent'], linewidth=3, 
                           label=f'Mean ({mean_dist:.1f}mm)')
                
                plt.title('📏 Source-Detector Distance Distribution', fontweight='bold', pad=15)
                plt.xlabel('Distance (mm)', fontweight='bold')
                plt.ylabel('Frequency', fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3, color='#404040')
                
                # Add statistics text
                plt.text(0.98, 0.98, f'μ = {mean_dist:.1f}mm\nσ = {std_dist:.1f}mm\nRange: {distances.min():.1f}-{distances.max():.1f}mm', 
                        transform=ax9.transAxes, va='top', ha='right', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            
            # 10. ENHANCED Tissue Type Distribution
            ax10 = plt.subplot(4, 5, 10)
            unique_props = np.unique(gt_data.reshape(-1, 2), axis=0)
            tissue_counts = []
            tissue_labels = []
            tissue_colors = []
            
            for i, (mua, musp) in enumerate(unique_props):
                count = np.sum((mua_map == mua) & (musp_map == musp))
                tissue_counts.append(count)
                if mua == 0 and musp == 0:
                    tissue_labels.append('Air')
                    tissue_colors.append(COLORS['air'])
                else:
                    tissue_labels.append(f'Tissue {i}')
                    tissue_colors.append(TISSUE_COLORS[i % len(TISSUE_COLORS)])
            
            wedges, texts, autotexts = plt.pie(tissue_counts, labels=tissue_labels, 
                                              autopct='%1.1f%%', colors=tissue_colors,
                                              explode=[0.05] * len(tissue_counts),
                                              shadow=True, startangle=90)
            
            # Enhance text
            for text in texts:
                text.set_fontweight('bold')
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            plt.title('🧬 Tissue Type Distribution', fontweight='bold', pad=15)
            
            # ═══════════════════════════════════════════════════════════════
            # ROW 3: ADVANCED MEASUREMENT ANALYSIS
            # ═══════════════════════════════════════════════════════════════
            
            # 11. SPECTACULAR Measurement Heatmap - Log Amplitude
            ax11 = plt.subplot(4, 5, 11)
            im11 = plt.imshow(log_amp, aspect='auto', cmap='viridis', interpolation='bilinear')
            plt.title('🔥 Log-Amplitude Heatmap per Probe', fontweight='bold', pad=15)
            plt.xlabel('Detector Index', fontweight='bold')
            plt.ylabel('Probe Index', fontweight='bold')
            cbar11 = plt.colorbar(im11, label='Log-Amplitude', shrink=0.8)
            cbar11.set_label('Log-Amplitude', fontweight='bold')
            
            # Add contour lines
            X, Y = np.meshgrid(range(log_amp.shape[1]), range(log_amp.shape[0]))
            contours = plt.contour(X, Y, log_amp, levels=8, colors='white', alpha=0.4, linewidths=0.5)
            
            # 12. SPECTACULAR Measurement Heatmap - Phase
            ax12 = plt.subplot(4, 5, 12)
            im12 = plt.imshow(phase, aspect='auto', cmap='RdYlBu_r', interpolation='bilinear')
            plt.title('🔥 Phase Heatmap per Probe', fontweight='bold', pad=15)
            plt.xlabel('Detector Index', fontweight='bold')
            plt.ylabel('Probe Index', fontweight='bold')
            cbar12 = plt.colorbar(im12, label='Phase (degrees)', shrink=0.8)
            cbar12.set_label('Phase (degrees)', fontweight='bold')
            
            # Add contour lines
            contours2 = plt.contour(X, Y, phase, levels=8, colors='white', alpha=0.4, linewidths=0.5)
            
            # 13. ADVANCED Statistical Summary
            ax13 = plt.subplot(4, 5, 13)
            ax13.axis('off')  # Turn off axes for text display
            
            # Calculate comprehensive statistics
            stats_text = f"""
    📊 COMPREHENSIVE STATISTICS
    
    🔢 Dataset Metrics:
    • Total Measurements: {n_measurements:,}
    • Number of Probes: {n_probes}
    • Detectors per Probe: {log_amp.shape[1]}
    
    📈 Log-Amplitude Analysis:
    • Range: [{log_amp.min():.2f}, {log_amp.max():.2f}]
    • Mean ± SD: {log_amp.mean():.2f} ± {log_amp.std():.2f}
    • Coefficient of Variation: {log_amp.std()/abs(log_amp.mean())*100:.1f}%
    
    🌊 Phase Analysis:
    • Range: [{phase.min():.1f}°, {phase.max():.1f}°]
    • Mean ± SD: {phase.mean():.1f}° ± {phase.std():.1f}°
    • Circular Range: {(phase.max()-phase.min()):.1f}°
    
    📏 Distance Analysis:
    • SDS Range: {distances.min():.1f} - {distances.max():.1f} mm
    • Mean SDS: {distances.mean():.1f} ± {distances.std():.1f} mm
    • Constraint Compliance: {np.sum((distances >= 10) & (distances <= 40))/len(distances)*100:.1f}%
    
    🎯 Ground Truth:
    • Volume Shape: {gt_data.shape[:3]}
    • Tissue Types: {len(unique_props) - 1}
    • Tissue Coverage: {np.sum(mua_map > 0)/mua_map.size*100:.1f}%
            """
            
            plt.text(0.05, 0.95, stats_text, transform=ax13.transAxes, 
                    fontsize=9, va='top', ha='left', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='#2d2d2d', alpha=0.9, edgecolor='#404040'))
            
            # 14. ADVANCED Measurement Quality Analysis
            ax14 = plt.subplot(4, 5, 14)
            
            # Calculate SNR proxy (mean/std for each probe)
            probe_snr = np.abs(log_amp.mean(axis=1)) / log_amp.std(axis=1)
            probe_indices = np.arange(len(probe_snr))
            
            # Create quality visualization
            colors = plt.cm.viridis(probe_snr / probe_snr.max())
            bars = plt.bar(probe_indices[::10], probe_snr[::10], color=colors[::10], 
                          alpha=0.8, edgecolor='white', linewidth=0.5)
            
            plt.title('📊 Measurement Quality per Probe', fontweight='bold', pad=15)
            plt.xlabel('Probe Index (subsampled)', fontweight='bold')
            plt.ylabel('Quality Score (μ/σ)', fontweight='bold')
            plt.grid(True, alpha=0.3, color='#404040')
            
            # Add quality threshold line
            quality_threshold = np.median(probe_snr)
            plt.axhline(y=quality_threshold, color=COLORS['warning'], linestyle='--', 
                       linewidth=2, label=f'Median Quality ({quality_threshold:.1f})')
            plt.legend()
            
            # 15. ADVANCED Correlation Matrix
            ax15 = plt.subplot(4, 5, 15)
            
            # Create correlation matrix between different measurement channels
            if log_amp.shape[1] >= 3:
                det_labels = [f'Det {i+1}' for i in range(log_amp.shape[1])]
                corr_matrix = np.corrcoef(log_amp.T)
                
                im15 = plt.imshow(corr_matrix, cmap='RdBu_r', aspect='equal', 
                                 vmin=-1, vmax=1, interpolation='bilinear')
                
                # Add correlation values as text
                for i in range(len(det_labels)):
                    for j in range(len(det_labels)):
                        text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                                       ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                                       fontweight='bold', fontsize=8)
                
                plt.title('🔗 Detector Correlation Matrix', fontweight='bold', pad=15)
                plt.xticks(range(len(det_labels)), det_labels, rotation=45)
                plt.yticks(range(len(det_labels)), det_labels)
                cbar15 = plt.colorbar(im15, label='Correlation', shrink=0.8)
                cbar15.set_label('Correlation Coefficient', fontweight='bold')
            
            # ═══════════════════════════════════════════════════════════════
            # ROW 4: ADVANCED ANALYSIS
            # ═══════════════════════════════════════════════════════════════
            
            # 16-20. Additional advanced plots can be added here for even more analysis
            
            plt.tight_layout(pad=3.0)
            
            if save_plots:
                output_dir = file_path.parent
                plot_path = output_dir / f"{file_path.parent.name}_SPECTACULAR_analysis.png"
                plt.savefig(plot_path, dpi=400, bbox_inches='tight', facecolor='#0d1117', 
                           edgecolor='none', pad_inches=0.2)
                print(f"🎨 SPECTACULAR VISUALIZATION SAVED: {plot_path}")
                
                # Also save a high-quality PDF version
                pdf_path = output_dir / f"{file_path.parent.name}_SPECTACULAR_analysis.pdf"
                plt.savefig(pdf_path, dpi=400, bbox_inches='tight', facecolor='#0d1117', 
                           edgecolor='none', pad_inches=0.2, format='pdf')
                print(f"📄 PDF VERSION SAVED: {pdf_path}")
            
            plt.show()
            
            return {
                'n_measurements': n_measurements,
                'n_probes': n_probes,
                'sds_range': [distances.min(), distances.max()] if len(distances) > 0 else [0, 0],
                'log_amp_stats': [log_amp.min(), log_amp.max(), log_amp.mean(), log_amp.std()],
                'phase_stats': [phase.min(), phase.max(), phase.mean(), phase.std()],
                'tissue_types': len(unique_props) - 1,
                'visualization_saved': save_plots
            }
            plt.title('Phase per Probe')
            plt.xlabel('Detector')
            plt.ylabel('Probe')
            plt.colorbar(im12, label='Phase (degrees)')
            
            plt.tight_layout()
            
            if save_plots:
                output_dir = file_path.parent
                plot_path = output_dir / f"{file_path.parent.name}_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"📊 Visualization saved: {plot_path}")
            
            plt.show()

    def deep_single_analysis(self, file_path):
        """
        Comprehensive, detailed analysis of a single phantom dataset.
        
        This function performs an exhaustive examination of one phantom including:
        - Complete dataset structure analysis
        - Detailed statistical breakdown of all measurements
        - Advanced quality assessment metrics
        - Comprehensive metadata examination
        - Tissue composition analysis
        - Probe placement efficiency metrics
        - Measurement distribution analysis with advanced statistics
        - Ground truth validation and cross-correlation analysis
        """
        print(f"\n🔬 DEEP ANALYSIS: {file_path.parent.name}")
        print("="*60)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # ========================
                # DATASET STRUCTURE & METADATA
                # ========================
                print("\n📋 DATASET STRUCTURE & METADATA")
                print("-"*40)
                
                # File size analysis
                file_size_mb = file_path.stat().st_size / (1024**2)
                print(f"File size: {file_size_mb:.2f} MB")
                
                # Dataset dimensions
                log_amplitude = f['log_amplitude'][:]
                phase = f['phase'][:]
                ground_truth = f['ground_truth'][:]
                source_pos = f['source_pos'][:]
                det_pos = f['det_pos'][:]
                
                n_probes = log_amplitude.shape[0]
                n_detectors_per_probe = log_amplitude.shape[1]
                total_measurements = n_probes * n_detectors_per_probe
                
                print(f"Probes: {n_probes}")
                print(f"Detectors per probe: {n_detectors_per_probe}")
                print(f"Total measurements: {total_measurements}")
                print(f"Ground truth shape: {ground_truth.shape}")
                
                # Metadata analysis
                print(f"\nMetadata:")
                for key, value in f.attrs.items():
                    print(f"  {key}: {value}")
                
                # ========================
                # MEASUREMENT STATISTICS
                # ========================
                print(f"\n📊 COMPREHENSIVE MEASUREMENT STATISTICS")
                print("-"*40)
                
                # Log amplitude analysis
                log_amp_flat = log_amplitude.flatten()
                print(f"Log Amplitude Statistics:")
                print(f"  Range: [{log_amp_flat.min():.3f}, {log_amp_flat.max():.3f}]")
                print(f"  Mean: {log_amp_flat.mean():.3f} ± {log_amp_flat.std():.3f}")
                print(f"  Median: {np.median(log_amp_flat):.3f}")
                print(f"  IQR: [{np.percentile(log_amp_flat, 25):.3f}, {np.percentile(log_amp_flat, 75):.3f}]")
                print(f"  Skewness: {stats.skew(log_amp_flat):.3f}")
                print(f"  Kurtosis: {stats.kurtosis(log_amp_flat):.3f}")
                
                # Phase analysis
                phase_flat = phase.flatten()
                print(f"\nPhase Statistics:")
                print(f"  Range: [{phase_flat.min():.2f}°, {phase_flat.max():.2f}°]")
                print(f"  Mean: {phase_flat.mean():.2f}° ± {phase_flat.std():.2f}°")
                print(f"  Median: {np.median(phase_flat):.2f}°")
                print(f"  IQR: [{np.percentile(phase_flat, 25):.2f}°, {np.percentile(phase_flat, 75):.2f}°]")
                print(f"  Skewness: {stats.skew(phase_flat):.3f}")
                print(f"  Kurtosis: {stats.kurtosis(phase_flat):.3f}")
                
                # ========================
                # SOURCE-DETECTOR SEPARATION ANALYSIS
                # ========================
                print(f"\n📏 SOURCE-DETECTOR SEPARATION ANALYSIS")
                print("-"*40)
                
                # Calculate all SDS distances
                all_distances = []
                for i in range(n_probes):
                    source = source_pos[i]
                    detectors = det_pos[i]  # 3 detectors for this source
                    for detector in detectors:
                        distance = np.linalg.norm(source - detector)
                        all_distances.append(distance)
                
                all_distances = np.array(all_distances)
                print(f"SDS Distance Statistics:")
                print(f"  Count: {len(all_distances)} measurements")
                print(f"  Range: [{all_distances.min():.1f}, {all_distances.max():.1f}] mm")
                print(f"  Mean: {all_distances.mean():.1f} ± {all_distances.std():.1f} mm")
                print(f"  Median: {np.median(all_distances):.1f} mm")
                
                # Distance distribution analysis
                distance_bins = np.arange(10, 41, 2)  # 2mm bins from 10-40mm
                hist, _ = np.histogram(all_distances, bins=distance_bins)
                print(f"  Distribution uniformity (std of bin counts): {hist.std():.1f}")
                
                # ========================
                # GROUND TRUTH ANALYSIS
                # ========================
                print(f"\n🎯 GROUND TRUTH OPTICAL PROPERTIES")
                print("-"*40)
                
                # Absorption coefficient analysis
                mua_map = ground_truth[:,:,:,0]
                mua_nonzero = mua_map[mua_map > 0]  # Exclude air regions
                print(f"Absorption Coefficient (μₐ):")
                print(f"  Range: [{mua_nonzero.min():.4f}, {mua_nonzero.max():.4f}] mm⁻¹")
                print(f"  Mean: {mua_nonzero.mean():.4f} ± {mua_nonzero.std():.4f} mm⁻¹")
                print(f"  Unique values: {len(np.unique(mua_nonzero))}")
                
                # Scattering coefficient analysis
                musp_map = ground_truth[:,:,:,1]
                musp_nonzero = musp_map[musp_map > 0]
                print(f"\nReduced Scattering (μ′s):")
                print(f"  Range: [{musp_nonzero.min():.3f}, {musp_nonzero.max():.3f}] mm⁻¹")
                print(f"  Mean: {musp_nonzero.mean():.3f} ± {musp_nonzero.std():.3f} mm⁻¹")
                print(f"  Unique values: {len(np.unique(musp_nonzero))}")
                
                # ========================
                # TISSUE COMPOSITION ANALYSIS
                # ========================
                if 'tissue_labels' in f:
                    print(f"\n🧬 TISSUE COMPOSITION ANALYSIS")
                    print("-"*40)
                    
                    tissue_labels = f['tissue_labels'][:]
                    unique_labels, counts = np.unique(tissue_labels, return_counts=True)
                    total_voxels = tissue_labels.size
                    
                    print(f"Tissue Distribution:")
                    for label, count in zip(unique_labels, counts):
                        percentage = count / total_voxels * 100
                        if label == 0:
                            print(f"  Air: {count:,} voxels ({percentage:.1f}%)")
                        elif label == 1:
                            print(f"  Healthy tissue: {count:,} voxels ({percentage:.1f}%)")
                        else:
                            tumor_num = label - 1
                            print(f"  Tumor {tumor_num}: {count:,} voxels ({percentage:.1f}%)")
                    
                    # Tissue coverage analysis
                    tissue_voxels = np.sum(tissue_labels > 0)
                    tissue_coverage = tissue_voxels / total_voxels * 100
                    print(f"\nTissue Coverage: {tissue_coverage:.1f}%")
                
                # ========================
                # MEASUREMENT QUALITY ASSESSMENT
                # ========================
                print(f"\n🎯 MEASUREMENT QUALITY ASSESSMENT")
                print("-"*40)
                
                # Signal dynamic range
                amp_dynamic_range = log_amp_flat.max() - log_amp_flat.min()
                print(f"Log-amplitude dynamic range: {amp_dynamic_range:.2f}")
                
                # Phase wrapping check
                phase_range = phase_flat.max() - phase_flat.min()
                print(f"Phase range: {phase_range:.1f}° (wrapping if > 360°)")
                
                # Measurement correlation analysis
                probe_correlations = []
                for i in range(min(10, n_probes)):  # Sample first 10 probes
                    probe_data = np.concatenate([log_amplitude[i], phase[i]])
                    for j in range(i+1, min(10, n_probes)):
                        other_probe_data = np.concatenate([log_amplitude[j], phase[j]])
                        correlation = np.corrcoef(probe_data, other_probe_data)[0,1]
                        probe_correlations.append(correlation)
                
                if probe_correlations:
                    mean_correlation = np.mean(probe_correlations)
                    print(f"Inter-probe correlation: {mean_correlation:.3f} (lower is better for diversity)")
                
                # ========================
                # ADVANCED STATISTICS
                # ========================
                print(f"\n📈 ADVANCED STATISTICAL ANALYSIS")
                print("-"*40)
                
                # Normality tests
                _, log_amp_p_value = stats.normaltest(log_amp_flat[:1000])  # Sample for speed
                _, phase_p_value = stats.normaltest(phase_flat[:1000])
                
                print(f"Normality tests (p-values):")
                print(f"  Log amplitude: {log_amp_p_value:.2e} ({'Normal' if log_amp_p_value > 0.05 else 'Non-normal'})")
                print(f"  Phase: {phase_p_value:.2e} ({'Normal' if phase_p_value > 0.05 else 'Non-normal'})")
                
                # Outlier detection
                def detect_outliers(data, method='iqr'):
                    if method == 'iqr':
                        q1, q3 = np.percentile(data, [25, 75])
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        return np.sum((data < lower) | (data > upper))
                    elif method == 'zscore':
                        z_scores = np.abs(stats.zscore(data))
                        return np.sum(z_scores > 3)
                
                log_amp_outliers = detect_outliers(log_amp_flat)
                phase_outliers = detect_outliers(phase_flat)
                
                print(f"\nOutlier detection (IQR method):")
                print(f"  Log amplitude outliers: {log_amp_outliers} ({log_amp_outliers/len(log_amp_flat)*100:.2f}%)")
                print(f"  Phase outliers: {phase_outliers} ({phase_outliers/len(phase_flat)*100:.2f}%)")
                
                print(f"\n✅ DEEP ANALYSIS COMPLETE")
                print(f"Dataset: {file_path.parent.name} shows {'EXCELLENT' if log_amp_outliers/len(log_amp_flat) < 0.05 else 'GOOD' if log_amp_outliers/len(log_amp_flat) < 0.1 else 'ACCEPTABLE'} quality")
                
        except Exception as e:
            print(f"❌ Error during deep analysis: {e}")

    def cross_dataset_analysis(self):
        """
        Comprehensive analysis across all available phantom datasets.
        
        This function performs cross-dataset comparative analysis including:
        - Statistical comparisons between phantoms
        - Dataset consistency validation
        - Cross-phantom averages and standard deviations
        - Measurement quality trends across the dataset
        - Tissue composition diversity analysis
        - Probe placement pattern analysis
        - Dataset-wide outlier detection
        - Training data suitability assessment
        """
        print(f"\n🔍 CROSS-DATASET COMPARATIVE ANALYSIS")
        print("="*60)
        print(f"Analyzing {len(self.phantom_files)} phantom datasets...")
        
        # Storage for cross-dataset statistics
        dataset_stats = {
            'phantom_names': [],
            'n_probes': [],
            'log_amp_ranges': [],
            'log_amp_means': [],
            'log_amp_stds': [],
            'phase_ranges': [],
            'phase_means': [],
            'phase_stds': [],
            'sds_means': [],
            'sds_stds': [],
            'tissue_coverages': [],
            'n_tumors': [],
            'file_sizes_mb': []
        }
        
        print(f"\n📊 COLLECTING DATASET STATISTICS")
        print("-"*40)
        
        for i, file_path in enumerate(self.phantom_files):
            phantom_name = file_path.parent.name
            print(f"Processing {phantom_name}... ({i+1}/{len(self.phantom_files)})")
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # Basic dataset info
                    dataset_stats['phantom_names'].append(phantom_name)
                    dataset_stats['file_sizes_mb'].append(file_path.stat().st_size / (1024**2))
                    
                    # Load measurement data
                    log_amplitude = f['log_amplitude'][:]
                    phase = f['phase'][:]
                    source_pos = f['source_pos'][:]
                    det_pos = f['det_pos'][:]
                    
                    n_probes = log_amplitude.shape[0]
                    dataset_stats['n_probes'].append(n_probes)
                    
                    # Log amplitude statistics
                    log_amp_flat = log_amplitude.flatten()
                    dataset_stats['log_amp_ranges'].append(log_amp_flat.max() - log_amp_flat.min())
                    dataset_stats['log_amp_means'].append(log_amp_flat.mean())
                    dataset_stats['log_amp_stds'].append(log_amp_flat.std())
                    
                    # Phase statistics
                    phase_flat = phase.flatten()
                    dataset_stats['phase_ranges'].append(phase_flat.max() - phase_flat.min())
                    dataset_stats['phase_means'].append(phase_flat.mean())
                    dataset_stats['phase_stds'].append(phase_flat.std())
                    
                    # SDS statistics
                    all_distances = []
                    for j in range(n_probes):
                        source = source_pos[j]
                        detectors = det_pos[j]
                        for detector in detectors:
                            distance = np.linalg.norm(source - detector)
                            all_distances.append(distance)
                    
                    all_distances = np.array(all_distances)
                    dataset_stats['sds_means'].append(all_distances.mean())
                    dataset_stats['sds_stds'].append(all_distances.std())
                    
                    # Tissue composition
                    if 'tissue_labels' in f:
                        tissue_labels = f['tissue_labels'][:]
                        unique_labels = np.unique(tissue_labels)
                        tissue_voxels = np.sum(tissue_labels > 0)
                        tissue_coverage = tissue_voxels / tissue_labels.size * 100
                        dataset_stats['tissue_coverages'].append(tissue_coverage)
                        
                        # Count tumors (labels >= 2)
                        n_tumors = len(unique_labels[unique_labels >= 2])
                        dataset_stats['n_tumors'].append(n_tumors)
                    else:
                        dataset_stats['tissue_coverages'].append(0)
                        dataset_stats['n_tumors'].append(0)
                        
            except Exception as e:
                print(f"  ⚠️ Error processing {phantom_name}: {e}")
                # Fill with NaN for failed datasets
                for key in dataset_stats.keys():
                    if key != 'phantom_names':
                        dataset_stats[key].append(np.nan)
        
        # Convert to numpy arrays for analysis
        for key, values in dataset_stats.items():
            if key != 'phantom_names':
                dataset_stats[key] = np.array(values)
        
        # ========================
        # CROSS-DATASET STATISTICS
        # ========================
        print(f"\n📈 CROSS-DATASET STATISTICAL SUMMARY")
        print("="*50)
        
        # Remove NaN values for statistics
        valid_indices = ~np.isnan(dataset_stats['n_probes'])
        n_valid = np.sum(valid_indices)
        
        print(f"Successfully analyzed: {n_valid}/{len(self.phantom_files)} datasets")
        
        if n_valid > 0:
            print(f"\nProbe Configuration:")
            print(f"  Mean probes per phantom: {np.nanmean(dataset_stats['n_probes']):.1f} ± {np.nanstd(dataset_stats['n_probes']):.1f}")
            print(f"  Range: [{np.nanmin(dataset_stats['n_probes']):.0f}, {np.nanmax(dataset_stats['n_probes']):.0f}]")
            
            print(f"\nLog Amplitude Consistency:")
            print(f"  Mean across datasets: {np.nanmean(dataset_stats['log_amp_means']):.3f} ± {np.nanstd(dataset_stats['log_amp_means']):.3f}")
            print(f"  Dynamic range consistency: {np.nanmean(dataset_stats['log_amp_ranges']):.3f} ± {np.nanstd(dataset_stats['log_amp_ranges']):.3f}")
            print(f"  Variability consistency: {np.nanmean(dataset_stats['log_amp_stds']):.3f} ± {np.nanstd(dataset_stats['log_amp_stds']):.3f}")
            
            print(f"\nPhase Consistency:")
            print(f"  Mean across datasets: {np.nanmean(dataset_stats['phase_means']):.2f}° ± {np.nanstd(dataset_stats['phase_means']):.2f}°")
            print(f"  Range consistency: {np.nanmean(dataset_stats['phase_ranges']):.2f}° ± {np.nanstd(dataset_stats['phase_ranges']):.2f}°")
            print(f"  Variability consistency: {np.nanmean(dataset_stats['phase_stds']):.2f}° ± {np.nanstd(dataset_stats['phase_stds']):.2f}°")
            
            print(f"\nSource-Detector Separation:")
            print(f"  Mean SDS across datasets: {np.nanmean(dataset_stats['sds_means']):.1f} ± {np.nanstd(dataset_stats['sds_means']):.1f} mm")
            print(f"  SDS variability: {np.nanmean(dataset_stats['sds_stds']):.1f} ± {np.nanstd(dataset_stats['sds_stds']):.1f} mm")
            
            print(f"\nTissue Composition Diversity:")
            print(f"  Mean tissue coverage: {np.nanmean(dataset_stats['tissue_coverages']):.1f}% ± {np.nanstd(dataset_stats['tissue_coverages']):.1f}%")
            print(f"  Coverage range: [{np.nanmin(dataset_stats['tissue_coverages']):.1f}%, {np.nanmax(dataset_stats['tissue_coverages']):.1f}%]")
            print(f"  Mean tumors per phantom: {np.nanmean(dataset_stats['n_tumors']):.1f} ± {np.nanstd(dataset_stats['n_tumors']):.1f}")
            print(f"  Tumor range: [{np.nanmin(dataset_stats['n_tumors']):.0f}, {np.nanmax(dataset_stats['n_tumors']):.0f}]")
            
            print(f"\nDataset Storage:")
            print(f"  Mean file size: {np.nanmean(dataset_stats['file_sizes_mb']):.1f} ± {np.nanstd(dataset_stats['file_sizes_mb']):.1f} MB")
            print(f"  Total dataset size: {np.nansum(dataset_stats['file_sizes_mb']):.1f} MB")
        
        # ========================
        # CONSISTENCY ANALYSIS
        # ========================
        print(f"\n🎯 DATASET CONSISTENCY ANALYSIS")
        print("-"*40)
        
        if n_valid > 1:
            # Coefficient of variation (CV) analysis
            def cv(data):
                return np.nanstd(data) / np.nanmean(data) * 100 if np.nanmean(data) != 0 else 0
            
            print(f"Coefficient of Variation (lower = more consistent):")
            print(f"  Probe count CV: {cv(dataset_stats['n_probes']):.1f}%")
            print(f"  Log amplitude mean CV: {cv(dataset_stats['log_amp_means']):.1f}%")
            print(f"  Phase mean CV: {cv(dataset_stats['phase_means']):.1f}%")
            print(f"  SDS mean CV: {cv(dataset_stats['sds_means']):.1f}%")
            print(f"  Tissue coverage CV: {cv(dataset_stats['tissue_coverages']):.1f}%")
            
            # Outlier detection across datasets
            def detect_dataset_outliers(data, name):
                if len(data) > 3:  # Need at least 4 datasets for meaningful outlier detection
                    z_scores = np.abs(stats.zscore(data[~np.isnan(data)]))
                    outliers = np.sum(z_scores > 2)  # More lenient threshold for small dataset
                    if outliers > 0:
                        print(f"  {name}: {outliers} outlier dataset(s)")
                        outlier_indices = np.where(z_scores > 2)[0]
                        for idx in outlier_indices:
                            valid_phantom_names = [name for i, name in enumerate(dataset_stats['phantom_names']) if valid_indices[i]]
                            if idx < len(valid_phantom_names):
                                print(f"    - {valid_phantom_names[idx]}")
            
            print(f"\nOutlier Dataset Detection:")
            detect_dataset_outliers(dataset_stats['log_amp_means'], "Log amplitude")
            detect_dataset_outliers(dataset_stats['phase_means'], "Phase")
            detect_dataset_outliers(dataset_stats['sds_means'], "SDS")
            detect_dataset_outliers(dataset_stats['tissue_coverages'], "Tissue coverage")
        
        # ========================
        # TRAINING SUITABILITY ASSESSMENT
        # ========================
        print(f"\n🎓 MACHINE LEARNING TRAINING SUITABILITY")
        print("-"*40)
        
        if n_valid > 0:
            total_measurements = np.nansum(dataset_stats['n_probes']) * 3  # 3 detectors per probe
            print(f"Total training measurements: {total_measurements:,}")
            
            # Diversity assessment
            tissue_diversity = np.nanstd(dataset_stats['tissue_coverages'])
            sds_diversity = np.nanstd(dataset_stats['sds_means'])
            
            print(f"Dataset diversity scores:")
            print(f"  Tissue coverage diversity: {tissue_diversity:.1f}% (higher is better)")
            print(f"  SDS diversity: {sds_diversity:.1f} mm (higher is better)")
            
            # Quality assessment
            mean_consistency = np.mean([
                cv(dataset_stats['log_amp_means']),
                cv(dataset_stats['phase_means'])
            ])
            
            print(f"  Measurement consistency: {mean_consistency:.1f}% CV (lower is better)")
            
            # Overall assessment
            if total_measurements > 5000 and tissue_diversity > 2 and mean_consistency < 10:
                assessment = "EXCELLENT"
            elif total_measurements > 2000 and tissue_diversity > 1 and mean_consistency < 20:
                assessment = "GOOD"
            elif total_measurements > 1000:
                assessment = "ACCEPTABLE"
            else:
                assessment = "INSUFFICIENT"
            
            print(f"\nOverall training suitability: {assessment}")
            
            if assessment == "EXCELLENT":
                print("  ✅ Dataset is ready for robust ML training")
            elif assessment == "GOOD":
                print("  ✅ Dataset is suitable for ML training")
            elif assessment == "ACCEPTABLE":
                print("  ⚠️ Dataset may work but consider generating more phantoms")
            else:
                print("  ❌ Dataset needs significant expansion for reliable training")
        
        print(f"\n✅ CROSS-DATASET ANALYSIS COMPLETE")
        print(f"Analyzed {n_valid} phantom datasets with comprehensive statistics")

def main():
    """Main function to run comprehensive dataset analysis with 3 focused options."""
    
    print("🧬 NIR PHANTOM DATASET ANALYZER")
    print("="*50)
    print("Comprehensive analysis of HDF5 phantom datasets")
    print("="*50)
    
    # Initialize analyzer
    analyzer = NIRDatasetAnalyzer()
    
    if not analyzer.phantom_files:
        print("❌ No datasets found. Please run the data simulator first.")
        return
    
    # Streamlined analysis options
    print("\n📋 NIR Dataset Analysis Options:")
    print("1. 🔬 Deep Single Dataset Analysis (comprehensive phantom examination)")
    print("2. 📊 Spectacular Visualizations (enhanced plots and 3D rendering)")  
    print("3. 🔍 Cross-Dataset Comparative Analysis (analyze all phantoms together)")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        # OPTION 1: Deep Single Dataset Analysis
        print("\n🔬 DEEP SINGLE DATASET ANALYSIS")
        print("="*40)
        print("Available datasets:")
        for i, file_path in enumerate(analyzer.phantom_files):
            print(f"   {i+1}. {file_path.parent.name}")
        
        try:
            idx = int(input(f"Select dataset (1-{len(analyzer.phantom_files)}): ")) - 1
            selected_file = analyzer.phantom_files[idx]
            
            print(f"\n🔍 Performing comprehensive analysis of {selected_file.parent.name}...")
            analyzer.deep_single_analysis(selected_file)
            
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    elif choice == "2":
        # OPTION 2: Spectacular Visualizations
        print("\n📊 SPECTACULAR VISUALIZATIONS")
        print("="*40)
        print("Available datasets:")
        for i, file_path in enumerate(analyzer.phantom_files):
            print(f"   {i+1}. {file_path.parent.name}")
        
        try:
            idx = int(input(f"Select dataset (1-{len(analyzer.phantom_files)}): ")) - 1
            selected_file = analyzer.phantom_files[idx]
            
            print(f"\n🎨 Creating spectacular visualizations for {selected_file.parent.name}...")
            analyzer.visualize_dataset(selected_file)
            
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    elif choice == "3":
        # OPTION 3: Cross-Dataset Comparative Analysis
        print("\n🔍 CROSS-DATASET COMPARATIVE ANALYSIS")
        print("="*40)
        print(f"Analyzing {len(analyzer.phantom_files)} phantom datasets...")
        
        analyzer.cross_dataset_analysis()
    
    else:
        print("❌ Invalid option selected")

if __name__ == "__main__":
    main()
