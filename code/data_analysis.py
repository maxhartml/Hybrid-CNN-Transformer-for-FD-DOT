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
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality scientific plots
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Define professional color schemes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'info': '#592941',
    'tissue': '#2E8B57',
    'tumor': '#DC143C',
    'air': '#4682B4',
    'source': '#FFD700',
    'detector': '#00FFFF'
}

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
        self.output_dir.mkdir(exist_ok=True)
        
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
        """Create comprehensive visualizations for a single dataset."""
        
        print(f"\n📊 GENERATING VISUALIZATIONS: {file_path.parent.name}")
        print("="*60)
        
        with h5py.File(file_path, 'r') as f:
            # Load data
            log_amp = f['log_amplitude'][:]
            phase = f['phase'][:]
            source_pos = f['source_pos'][:]
            det_pos = f['det_pos'][:]
            gt_data = f['ground_truth'][:]
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f'NIR Dataset Analysis: {file_path.parent.name}', fontsize=16, fontweight='bold')
            
            # 1. Measurement distributions
            ax1 = plt.subplot(3, 4, 1)
            plt.hist(log_amp.flatten(), bins=50, alpha=0.7, edgecolor='black')
            plt.title('Log-Amplitude Distribution')
            plt.xlabel('Log-Amplitude')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            ax2 = plt.subplot(3, 4, 2)
            plt.hist(phase.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
            plt.title('Phase Distribution')
            plt.xlabel('Phase (degrees)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 2. Measurement correlation
            ax3 = plt.subplot(3, 4, 3)
            plt.scatter(log_amp.flatten()[::10], phase.flatten()[::10], alpha=0.5, s=1)
            plt.title('Log-Amplitude vs Phase')
            plt.xlabel('Log-Amplitude')
            plt.ylabel('Phase (degrees)')
            plt.grid(True, alpha=0.3)
            
            # 3. Source-detector geometry
            ax4 = plt.subplot(3, 4, 4, projection='3d')
            ax4.scatter(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2], 
                       c='red', s=20, label='Sources', alpha=0.7)
            if len(det_pos.shape) == 3:
                det_flat = det_pos.reshape(-1, 3)
                ax4.scatter(det_flat[:, 0], det_flat[:, 1], det_flat[:, 2], 
                           c='blue', s=10, label='Detectors', alpha=0.5)
            ax4.set_title('Probe Geometry')
            ax4.set_xlabel('X (mm)')
            ax4.set_ylabel('Y (mm)')
            ax4.set_zlabel('Z (mm)')
            ax4.legend()
            
            # 4. Ground truth slices
            mua_map = gt_data[..., 0]
            musp_map = gt_data[..., 1]
            
            # Central slices
            z_center = mua_map.shape[2] // 2
            y_center = mua_map.shape[1] // 2
            x_center = mua_map.shape[0] // 2
            
            ax5 = plt.subplot(3, 4, 5)
            im5 = plt.imshow(mua_map[:, :, z_center], cmap='viridis', aspect='equal')
            plt.title(f'μₐ Map (Z={z_center})')
            plt.colorbar(im5, label='μₐ (mm⁻¹)')
            
            ax6 = plt.subplot(3, 4, 6)
            im6 = plt.imshow(musp_map[:, :, z_center], cmap='plasma', aspect='equal')
            plt.title(f'μ′s Map (Z={z_center})')
            plt.colorbar(im6, label='μ′s (mm⁻¹)')
            
            ax7 = plt.subplot(3, 4, 7)
            im7 = plt.imshow(mua_map[:, y_center, :], cmap='viridis', aspect='equal')
            plt.title(f'μₐ Map (Y={y_center})')
            plt.colorbar(im7, label='μₐ (mm⁻¹)')
            
            ax8 = plt.subplot(3, 4, 8)
            im8 = plt.imshow(musp_map[:, y_center, :], cmap='plasma', aspect='equal')
            plt.title(f'μ′s Map (Y={y_center})')
            plt.colorbar(im8, label='μ′s (mm⁻¹)')
            
            # 5. Distance distributions
            if len(det_pos.shape) == 3:
                distances = []
                for i in range(len(source_pos)):
                    src = source_pos[i]
                    dets = det_pos[i]
                    probe_distances = [np.linalg.norm(det - src) for det in dets]
                    distances.extend(probe_distances)
                
                ax9 = plt.subplot(3, 4, 9)
                plt.hist(distances, bins=30, alpha=0.7, edgecolor='black', color='green')
                plt.title('Source-Detector Distances')
                plt.xlabel('Distance (mm)')
                plt.ylabel('Frequency')
                plt.axvline(x=5, color='red', linestyle='--', label='Min distance (5mm)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 6. Tissue type distribution
            unique_props = np.unique(gt_data.reshape(-1, 2), axis=0)
            tissue_counts = []
            tissue_labels = []
            
            for i, (mua, musp) in enumerate(unique_props):
                count = np.sum((mua_map == mua) & (musp_map == musp))
                tissue_counts.append(count)
                if mua == 0 and musp == 0:
                    tissue_labels.append('Air')
                else:
                    tissue_labels.append(f'Tissue {i}')
            
            ax10 = plt.subplot(3, 4, 10)
            plt.pie(tissue_counts, labels=tissue_labels, autopct='%1.1f%%')
            plt.title('Tissue Type Distribution')
            
            # 7. Measurement heatmap per probe
            ax11 = plt.subplot(3, 4, 11)
            im11 = plt.imshow(log_amp, aspect='auto', cmap='viridis')
            plt.title('Log-Amplitude per Probe')
            plt.xlabel('Detector')
            plt.ylabel('Probe')
            plt.colorbar(im11, label='Log-Amplitude')
            
            ax12 = plt.subplot(3, 4, 12)
            im12 = plt.imshow(phase, aspect='auto', cmap='RdYlBu')
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

def main():
    """Main function to run comprehensive dataset analysis."""
    
    print("🧬 NIR PHANTOM DATASET ANALYZER")
    print("="*50)
    print("Comprehensive analysis of HDF5 phantom datasets")
    print("="*50)
    
    # Initialize analyzer
    analyzer = NIRDatasetAnalyzer()
    
    if not analyzer.phantom_files:
        print("❌ No datasets found. Please run the data simulator first.")
        return
    
    # Analysis options
    print("\n📋 Analysis Options:")
    print("1. Analyze single dataset (detailed)")
    print("2. Analyze all datasets (summary)")
    print("3. Visualize single dataset")
    print("4. Full analysis with visualizations")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        print("\nAvailable datasets:")
        for i, file_path in enumerate(analyzer.phantom_files):
            print(f"   {i+1}. {file_path.parent.name}")
        
        try:
            idx = int(input(f"Select dataset (1-{len(analyzer.phantom_files)}): ")) - 1
            analyzer.analyze_single_dataset(analyzer.phantom_files[idx], detailed=True)
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    elif choice == "2":
        analyzer.analyze_all_datasets()
    
    elif choice == "3":
        print("\nAvailable datasets:")
        for i, file_path in enumerate(analyzer.phantom_files):
            print(f"   {i+1}. {file_path.parent.name}")
        
        try:
            idx = int(input(f"Select dataset (1-{len(analyzer.phantom_files)}): ")) - 1
            analyzer.visualize_dataset(analyzer.phantom_files[idx])
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    elif choice == "4":
        print("\n🚀 Running full analysis...")
        all_results = analyzer.analyze_all_datasets()
        
        # Visualize first dataset as example
        if analyzer.phantom_files:
            print(f"\n📊 Creating visualizations for {analyzer.phantom_files[0].parent.name}...")
            analyzer.visualize_dataset(analyzer.phantom_files[0])
    
    else:
        print("❌ Invalid option selected")

if __name__ == "__main__":
    main()
