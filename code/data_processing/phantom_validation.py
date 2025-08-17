#!/usr/bin/env python3
"""
🔍 MASTER PHANTOM DATASET VALIDATOR 🔍

Comprehensive validation suite for NIR phantom datasets with full quality assurance:

✅ CORE VALIDATIONS:
• Data integrity (NaN/Inf detection across all phantoms)
• Measurement quality (SNR, amplitude/phase ranges)
• Tissue composition analysis
• Outlier detection and flagging
• File corruption checks

🔧 ADVANCED CHECKS:
• Statistical distribution analysis
• Cross-phantom consistency validation
• Quality score computation per phantom
• Detailed reporting with flagged issues
• Performance benchmarking

🎯 USAGE:
Perfect for validating large datasets (100-5000 phantoms) before training.
Provides confidence that your dataset is clean and ready for ML.

Author: Max Hart - NIR Tomography Research
Version: 2.0 - Master Validator
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import zscore
import warnings
import time
import os
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Quality thresholds for validation
QUALITY_THRESHOLDS = {
    'min_snr_db': 40.0,           # Minimum acceptable SNR
    'max_snr_db': 50.0,           # Maximum expected SNR
    'target_snr_db': 46.0,        # Target SNR (0.5% noise)
    'max_log_amp_range': 15.0,    # Max reasonable log amplitude range
    'min_log_amp_range': 5.0,     # Min reasonable log amplitude range
    'max_phase_range': 180.0,     # Max reasonable phase range
    'min_tissue_percentage': 15.0, # Min tissue coverage
    'max_tissue_percentage': 50.0, # Max tissue coverage
    'outlier_z_threshold': 3.0,   # Z-score threshold for outlier detection
    'max_measurement_count': 1000, # Expected measurement count
    'min_measurement_count': 1000  # Expected measurement count
}

def analyze_phantom_dataset(data_directory, output_dir="analysis_results", quick_mode=False):
    """
    Master validation suite for phantom datasets with comprehensive quality assurance.
    
    Args:
        data_directory (str): Path to directory containing phantom HDF5 files
        output_dir (str): Directory to save analysis results and reports
        quick_mode (bool): If True, sample subset for faster analysis
    
    Returns:
        dict: Comprehensive validation results and quality metrics
    """
    
    print("🔍 MASTER PHANTOM DATASET VALIDATOR")
    print("=" * 80)
    
    data_path = Path(data_directory)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all phantom files
    phantom_files = list(data_path.glob("phantom_*/phantom_*_scan.h5"))
    n_phantoms = len(phantom_files)
    
    print(f"📊 Found {n_phantoms} phantom files to validate")
    
    if quick_mode and n_phantoms > 100:
        phantom_files = phantom_files[:100]
        n_phantoms = 100
        print(f"🚀 Quick mode: Analyzing first {n_phantoms} phantoms")
    
    print("-" * 80)
    
    if n_phantoms == 0:
        print("❌ No phantom files found! Check the data directory path.")
        return {'status': 'failed', 'reason': 'no_files_found'}
    
    # Initialize comprehensive data collection
    validation_data = {
        'phantom_id': [], 'file_path': [], 'file_size_mb': [],
        'log_amp_min': [], 'log_amp_max': [], 'log_amp_mean': [], 'log_amp_std': [], 'log_amp_range': [],
        'phase_min': [], 'phase_max': [], 'phase_mean': [], 'phase_std': [], 'phase_range': [],
        'n_measurements': [], 'expected_measurements': [],
        'tissue_voxels': [], 'tumor_voxels': [], 'air_voxels': [], 'total_voxels': [],
        'tissue_percentage': [], 'tumor_percentage': [], 'air_percentage': [],
        'n_tumors': [], 'max_tumor_size': [], 'min_tumor_size': [], 'avg_tumor_size': [],
        'mua_healthy_min': [], 'mua_healthy_max': [], 'mua_healthy_mean': [],
        'musp_healthy_min': [], 'musp_healthy_max': [], 'musp_healthy_mean': [],
        'has_nan_log_amp': [], 'has_inf_log_amp': [], 'has_nan_phase': [], 'has_inf_phase': [],
        'amplitude_snr_db': [], 'noise_amplitude_std': [], 'noise_phase_std': [],
        'processing_time_ms': [], 'quality_score': [], 'issues': []
    }
    
    print("📈 Processing phantoms with comprehensive quality checks...")
    start_time = time.time()
    
    failed_phantoms = []
    issue_summary = {
        'file_corruption': 0, 'nan_inf_values': 0, 'snr_issues': 0, 
        'measurement_count_wrong': 0, 'extreme_outliers': 0, 'tissue_issues': 0
    }
    
    # Process each phantom with detailed validation
    for i, phantom_file in enumerate(phantom_files):
        phantom_start = time.time()
        phantom_id = phantom_file.parent.name
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_phantoms - i - 1) / rate if rate > 0 else 0
            print(f"   Progress: {i+1}/{n_phantoms} ({100*(i+1)/n_phantoms:.1f}%) | ETA: {eta:.0f}s")
        
        try:
            phantom_issues = []
            
            # Check file accessibility and size
            file_size_mb = phantom_file.stat().st_size / (1024**2)
            
            with h5py.File(phantom_file, 'r') as f:
                # Load all measurement data
                log_amplitude = f['log_amplitude'][:]
                phase = f['phase'][:]
                ground_truth = f['ground_truth'][:]  # Shape: (2, Nx, Ny, Nz)
                phantom_volume = f['tissue_labels'][:]
                
                # === BASIC DATA VALIDATION ===
                validation_data['phantom_id'].append(phantom_id)
                validation_data['file_path'].append(str(phantom_file))
                validation_data['file_size_mb'].append(file_size_mb)
                
                # === MEASUREMENT STATISTICS ===
                log_amp_min, log_amp_max = float(np.min(log_amplitude)), float(np.max(log_amplitude))
                log_amp_mean, log_amp_std = float(np.mean(log_amplitude)), float(np.std(log_amplitude))
                log_amp_range = log_amp_max - log_amp_min
                
                phase_min, phase_max = float(np.min(phase)), float(np.max(phase))
                phase_mean, phase_std = float(np.mean(phase)), float(np.std(phase))
                phase_range = phase_max - phase_min
                
                validation_data['log_amp_min'].append(log_amp_min)
                validation_data['log_amp_max'].append(log_amp_max)
                validation_data['log_amp_mean'].append(log_amp_mean)
                validation_data['log_amp_std'].append(log_amp_std)
                validation_data['log_amp_range'].append(log_amp_range)
                
                validation_data['phase_min'].append(phase_min)
                validation_data['phase_max'].append(phase_max)
                validation_data['phase_mean'].append(phase_mean)
                validation_data['phase_std'].append(phase_std)
                validation_data['phase_range'].append(phase_range)
                
                n_measurements = len(log_amplitude)
                expected_measurements = QUALITY_THRESHOLDS['max_measurement_count']
                validation_data['n_measurements'].append(n_measurements)
                validation_data['expected_measurements'].append(expected_measurements)
                
                # === TISSUE COMPOSITION ANALYSIS ===
                air_voxels = int(np.sum(phantom_volume == 0))
                tissue_voxels = int(np.sum(phantom_volume == 1))
                tumor_voxels = int(np.sum(phantom_volume >= 2))
                total_voxels = phantom_volume.size
                
                tissue_percentage = 100 * (tissue_voxels + tumor_voxels) / total_voxels
                tumor_percentage = 100 * tumor_voxels / total_voxels if tumor_voxels > 0 else 0
                air_percentage = 100 * air_voxels / total_voxels
                
                validation_data['air_voxels'].append(air_voxels)
                validation_data['tissue_voxels'].append(tissue_voxels)
                validation_data['tumor_voxels'].append(tumor_voxels)
                validation_data['total_voxels'].append(total_voxels)
                validation_data['tissue_percentage'].append(tissue_percentage)
                validation_data['tumor_percentage'].append(tumor_percentage)
                validation_data['air_percentage'].append(air_percentage)
                
                # === TUMOR ANALYSIS ===
                unique_labels = np.unique(phantom_volume)
                tumor_labels = [label for label in unique_labels if label >= 2]
                n_tumors = len(tumor_labels)
                
                if tumor_labels:
                    tumor_sizes = [int(np.sum(phantom_volume == label)) for label in tumor_labels]
                    max_tumor_size = max(tumor_sizes)
                    min_tumor_size = min(tumor_sizes)
                    avg_tumor_size = np.mean(tumor_sizes)
                else:
                    max_tumor_size = min_tumor_size = avg_tumor_size = 0
                
                validation_data['n_tumors'].append(n_tumors)
                validation_data['max_tumor_size'].append(max_tumor_size)
                validation_data['min_tumor_size'].append(min_tumor_size)
                validation_data['avg_tumor_size'].append(avg_tumor_size)
                
                # === OPTICAL PROPERTY ANALYSIS ===
                mua_map = ground_truth[0]  # Absorption map
                musp_map = ground_truth[1]  # Scattering map
                
                # Get healthy tissue optical properties
                healthy_mask = (phantom_volume == 1)
                if np.any(healthy_mask):
                    healthy_mua = mua_map[healthy_mask]
                    healthy_musp = musp_map[healthy_mask]
                    
                    validation_data['mua_healthy_min'].append(float(np.min(healthy_mua)))
                    validation_data['mua_healthy_max'].append(float(np.max(healthy_mua)))
                    validation_data['mua_healthy_mean'].append(float(np.mean(healthy_mua)))
                    validation_data['musp_healthy_min'].append(float(np.min(healthy_musp)))
                    validation_data['musp_healthy_max'].append(float(np.max(healthy_musp)))
                    validation_data['musp_healthy_mean'].append(float(np.mean(healthy_musp)))
                else:
                    validation_data['mua_healthy_min'].append(0)
                    validation_data['mua_healthy_max'].append(0)
                    validation_data['mua_healthy_mean'].append(0)
                    validation_data['musp_healthy_min'].append(0)
                    validation_data['musp_healthy_max'].append(0)
                    validation_data['musp_healthy_mean'].append(0)
                
                # === DATA INTEGRITY CHECKS ===
                has_nan_log_amp = bool(np.any(np.isnan(log_amplitude)))
                has_inf_log_amp = bool(np.any(np.isinf(log_amplitude)))
                has_nan_phase = bool(np.any(np.isnan(phase)))
                has_inf_phase = bool(np.any(np.isinf(phase)))
                
                validation_data['has_nan_log_amp'].append(has_nan_log_amp)
                validation_data['has_inf_log_amp'].append(has_inf_log_amp)
                validation_data['has_nan_phase'].append(has_nan_phase)
                validation_data['has_inf_phase'].append(has_inf_phase)
                
                # === SNR CALCULATION ===
                if 'noise_amplitude_std' in f.attrs:
                    noise_amplitude_std = f.attrs['noise_amplitude_std']
                    # CORRECT: mean(exp(log_amp)) not exp(mean(log_amp))
                    amplitude = np.exp(log_amplitude)
                    signal_mean = np.mean(amplitude)
                    snr_db = 20 * np.log10(signal_mean / noise_amplitude_std) if noise_amplitude_std > 0 else 0
                    validation_data['amplitude_snr_db'].append(float(snr_db))
                    validation_data['noise_amplitude_std'].append(float(noise_amplitude_std))
                else:
                    validation_data['amplitude_snr_db'].append(0)
                    validation_data['noise_amplitude_std'].append(0)
                
                if 'noise_phase_std' in f.attrs:
                    validation_data['noise_phase_std'].append(float(f.attrs['noise_phase_std']))
                else:
                    validation_data['noise_phase_std'].append(0)
                
                # === QUALITY ASSESSMENT ===
                quality_score = 100.0  # Start with perfect score
                
                # Check for critical issues
                if has_nan_log_amp or has_inf_log_amp or has_nan_phase or has_inf_phase:
                    phantom_issues.append("NaN/Inf values detected")
                    quality_score -= 50
                    issue_summary['nan_inf_values'] += 1
                
                if n_measurements != expected_measurements:
                    phantom_issues.append(f"Wrong measurement count: {n_measurements} vs {expected_measurements}")
                    quality_score -= 30
                    issue_summary['measurement_count_wrong'] += 1
                
                # SNR validation
                snr_db = validation_data['amplitude_snr_db'][-1]
                if not (QUALITY_THRESHOLDS['min_snr_db'] <= snr_db <= QUALITY_THRESHOLDS['max_snr_db']):
                    phantom_issues.append(f"SNR out of range: {snr_db:.1f} dB")
                    quality_score -= 20
                    issue_summary['snr_issues'] += 1
                
                # Measurement range validation
                if log_amp_range < QUALITY_THRESHOLDS['min_log_amp_range'] or log_amp_range > QUALITY_THRESHOLDS['max_log_amp_range']:
                    phantom_issues.append(f"Log amplitude range unusual: {log_amp_range:.1f}")
                    quality_score -= 15
                
                if phase_range > QUALITY_THRESHOLDS['max_phase_range']:
                    phantom_issues.append(f"Phase range too large: {phase_range:.1f}°")
                    quality_score -= 10
                
                # Tissue composition validation
                if not (QUALITY_THRESHOLDS['min_tissue_percentage'] <= tissue_percentage <= QUALITY_THRESHOLDS['max_tissue_percentage']):
                    phantom_issues.append(f"Tissue percentage unusual: {tissue_percentage:.1f}%")
                    quality_score -= 10
                    issue_summary['tissue_issues'] += 1
                
                validation_data['quality_score'].append(max(0, quality_score))
                validation_data['issues'].append("; ".join(phantom_issues) if phantom_issues else "None")
                
                # Processing time
                processing_time_ms = (time.time() - phantom_start) * 1000
                validation_data['processing_time_ms'].append(processing_time_ms)
                
        except Exception as e:
            print(f"⚠️  Error processing {phantom_file}: {e}")
            failed_phantoms.append((phantom_id, str(e)))
            issue_summary['file_corruption'] += 1
            continue
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(validation_data)
    n_valid = len(df)
    total_time = time.time() - start_time
    
    print(f"✅ Successfully processed {n_valid}/{n_phantoms} phantoms in {total_time:.1f}s")
    if failed_phantoms:
        print(f"❌ Failed to process {len(failed_phantoms)} phantoms:")
        for phantom_id, error in failed_phantoms[:5]:  # Show first 5 errors
            print(f"   • {phantom_id}: {error}")
        if len(failed_phantoms) > 5:
            print(f"   ... and {len(failed_phantoms)-5} more")
    
    print("-" * 80)
    
    if n_valid == 0:
        return {'status': 'failed', 'reason': 'no_valid_phantoms'}
    
    # =================================================================
    # COMPREHENSIVE OUTLIER DETECTION
    # =================================================================
    
    print("\n🔍 ADVANCED OUTLIER DETECTION")
    print("=" * 80)
    
    # Define metrics to check for outliers
    outlier_metrics = [
        'log_amp_mean', 'log_amp_std', 'log_amp_range',
        'phase_mean', 'phase_std', 'phase_range',
        'amplitude_snr_db', 'tissue_percentage', 'tumor_percentage',
        'n_tumors', 'avg_tumor_size', 'quality_score'
    ]
    
    outlier_results = {}
    total_outliers = 0
    
    for metric in outlier_metrics:
        if metric in df.columns and df[metric].nunique() > 1:  # Skip if all values are the same
            # Calculate z-scores
            z_scores = np.abs(zscore(df[metric], nan_policy='omit'))
            outlier_mask = z_scores > QUALITY_THRESHOLDS['outlier_z_threshold']
            outlier_indices = df.index[outlier_mask].tolist()
            
            if len(outlier_indices) > 0:
                outlier_phantoms = df.loc[outlier_indices, 'phantom_id'].tolist()
                outlier_values = df.loc[outlier_indices, metric].tolist()
                
                outlier_results[metric] = {
                    'count': len(outlier_indices),
                    'phantoms': outlier_phantoms,
                    'values': outlier_values,
                    'z_scores': z_scores[outlier_mask].tolist(),
                    'mean': df[metric].mean(),
                    'std': df[metric].std()
                }
                
                total_outliers += len(outlier_indices)
                
                print(f"📊 {metric}: {len(outlier_indices)} outliers detected")
                print(f"   Mean ± Std: {df[metric].mean():.3f} ± {df[metric].std():.3f}")
                if len(outlier_indices) <= 5:
                    for idx, (phantom, value, z) in enumerate(zip(outlier_phantoms, outlier_values, z_scores[outlier_mask])):
                        print(f"   • {phantom}: {value:.3f} (z={z:.1f})")
                else:
                    print(f"   • {outlier_phantoms[0]}: {outlier_values[0]:.3f} (z={z_scores[outlier_mask][0]:.1f}) ... and {len(outlier_indices)-1} more")
    
    if total_outliers == 0:
        print("✅ No statistical outliers detected - Excellent consistency!")
    else:
        print(f"⚠️  Total outlier detections: {total_outliers} (some phantoms may appear in multiple categories)")
    
    # Flag extreme outliers for detailed review
    extreme_outliers = []
    for phantom_idx, quality_score in enumerate(df['quality_score']):
        if quality_score < 70:  # Low quality score
            phantom_id = df.loc[phantom_idx, 'phantom_id']
            issues = df.loc[phantom_idx, 'issues']
            extreme_outliers.append({'phantom_id': phantom_id, 'quality_score': quality_score, 'issues': issues})
    
    if extreme_outliers:
        issue_summary['extreme_outliers'] = len(extreme_outliers)
        print(f"\n🚨 EXTREME OUTLIERS REQUIRING REVIEW: {len(extreme_outliers)}")
        for outlier in extreme_outliers[:5]:  # Show first 5
            print(f"   • {outlier['phantom_id']}: Score {outlier['quality_score']:.0f}/100 - {outlier['issues']}")
        if len(extreme_outliers) > 5:
            print(f"   ... and {len(extreme_outliers)-5} more")
    
    # =================================================================
    # COMPREHENSIVE VALIDATION REPORT
    # =================================================================
    
    print(f"\n🎯 MASTER VALIDATION REPORT")
    print("=" * 80)
    
    # 1. DATA INTEGRITY ASSESSMENT
    print("\n1️⃣ DATA INTEGRITY ASSESSMENT:")
    nan_log_amp = df['has_nan_log_amp'].sum()
    inf_log_amp = df['has_inf_log_amp'].sum()
    nan_phase = df['has_nan_phase'].sum()
    inf_phase = df['has_inf_phase'].sum()
    
    total_integrity_issues = nan_log_amp + inf_log_amp + nan_phase + inf_phase
    if total_integrity_issues == 0:
        print("✅ PERFECT DATA INTEGRITY - No NaN/Inf values detected")
        integrity_score = 100
    else:
        print(f"❌ Data integrity issues found in {total_integrity_issues} instances:")
        print(f"   • NaN log amplitude: {nan_log_amp} phantoms")
        print(f"   • Inf log amplitude: {inf_log_amp} phantoms")
        print(f"   • NaN phase: {nan_phase} phantoms")
        print(f"   • Inf phase: {inf_phase} phantoms")
        integrity_score = max(0, 100 - (total_integrity_issues / n_valid) * 100)
    
    # 2. MEASUREMENT QUALITY ASSESSMENT
    print("\n2️⃣ MEASUREMENT QUALITY ASSESSMENT:")
    
    # SNR Analysis
    avg_snr = df['amplitude_snr_db'].mean()
    snr_std = df['amplitude_snr_db'].std()
    snr_in_range = np.sum((df['amplitude_snr_db'] >= QUALITY_THRESHOLDS['min_snr_db']) & 
                          (df['amplitude_snr_db'] <= QUALITY_THRESHOLDS['max_snr_db']))
    snr_score = (snr_in_range / n_valid) * 100
    
    print(f"📡 Signal-to-Noise Ratio:")
    print(f"   • Average SNR: {avg_snr:.1f} ± {snr_std:.1f} dB")
    print(f"   • Target: {QUALITY_THRESHOLDS['target_snr_db']:.1f} dB (0.5% noise)")
    print(f"   • Phantoms in acceptable range: {snr_in_range}/{n_valid} ({snr_score:.1f}%)")
    
    if snr_score >= 95:
        print("   ✅ SNR Excellent")
    elif snr_score >= 90:
        print("   ⚠️  SNR Good with minor deviations")
    else:
        print("   ❌ SNR Issues detected")
    
    # Measurement Ranges
    print(f"\n� Measurement Ranges:")
    print(f"   • Log Amplitude: [{df['log_amp_min'].min():.2f}, {df['log_amp_max'].max():.2f}]")
    print(f"     Mean ± Std: {df['log_amp_mean'].mean():.2f} ± {df['log_amp_std'].mean():.2f}")
    print(f"   • Phase: [{df['phase_min'].min():.1f}°, {df['phase_max'].max():.1f}°]")
    print(f"     Mean ± Std: {df['phase_mean'].mean():.1f}° ± {df['phase_std'].mean():.1f}°")
    
    # Check if ranges are reasonable
    reasonable_log_range = (df['log_amp_min'].min() >= -25) and (df['log_amp_max'].max() <= 5)
    reasonable_phase_range = (df['phase_min'].min() >= 0) and (df['phase_max'].max() <= 360)
    
    if reasonable_log_range and reasonable_phase_range:
        print("   ✅ All measurement ranges physically reasonable")
        range_score = 100
    else:
        print("   ⚠️  Some measurement ranges may be unusual")
        range_score = 80
    
    # 3. PHANTOM COMPOSITION ANALYSIS
    print("\n3️⃣ PHANTOM COMPOSITION ANALYSIS:")
    print(f"🧬 Tissue Coverage:")
    print(f"   • Average: {df['tissue_percentage'].mean():.1f}% ± {df['tissue_percentage'].std():.1f}%")
    print(f"   • Range: [{df['tissue_percentage'].min():.1f}%, {df['tissue_percentage'].max():.1f}%]")
    
    tissue_in_range = np.sum((df['tissue_percentage'] >= QUALITY_THRESHOLDS['min_tissue_percentage']) & 
                            (df['tissue_percentage'] <= QUALITY_THRESHOLDS['max_tissue_percentage']))
    tissue_score = (tissue_in_range / n_valid) * 100
    print(f"   • Phantoms with reasonable tissue coverage: {tissue_in_range}/{n_valid} ({tissue_score:.1f}%)")
    
    print(f"\n🎯 Tumor Analysis:")
    tumor_phantoms = df[df['n_tumors'] > 0]
    n_with_tumors = len(tumor_phantoms)
    print(f"   • Phantoms with tumors: {n_with_tumors}/{n_valid} ({100*n_with_tumors/n_valid:.1f}%)")
    if n_with_tumors > 0:
        print(f"   • Average tumors per phantom: {df['n_tumors'].mean():.1f}")
        print(f"   • Average tumor coverage: {df['tumor_percentage'].mean():.2f}%")
        print(f"   • Tumor size range: {df['min_tumor_size'].min():.0f} - {df['max_tumor_size'].max():.0f} voxels")
    
    # 4. CONSISTENCY AND VARIABILITY
    print("\n4️⃣ DATASET CONSISTENCY ANALYSIS:")
    
    # Check measurement count consistency
    measurement_consistency = df['n_measurements'].nunique() == 1
    expected_count = QUALITY_THRESHOLDS['max_measurement_count']
    correct_count = np.sum(df['n_measurements'] == expected_count)
    count_score = (correct_count / n_valid) * 100
    
    print(f"� Measurement Count Consistency:")
    print(f"   • Expected per phantom: {expected_count}")
    print(f"   • Phantoms with correct count: {correct_count}/{n_valid} ({count_score:.1f}%)")
    
    if count_score == 100:
        print("   ✅ Perfect measurement count consistency")
    else:
        print("   ⚠️  Measurement count inconsistencies detected")
    
    # Processing performance
    avg_processing_time = df['processing_time_ms'].mean()
    print(f"\n⚡ Processing Performance:")
    print(f"   • Average processing time: {avg_processing_time:.1f} ms per phantom")
    print(f"   • Total processing time: {total_time:.1f} seconds")
    print(f"   • Processing rate: {n_valid/total_time:.1f} phantoms/second")
    
    # 5. OVERALL QUALITY ASSESSMENT
    print("\n5️⃣ OVERALL QUALITY ASSESSMENT:")
    
    # Calculate composite quality metrics
    avg_quality_score = df['quality_score'].mean()
    quality_distribution = {
        'excellent': np.sum(df['quality_score'] >= 95),
        'good': np.sum((df['quality_score'] >= 85) & (df['quality_score'] < 95)),
        'acceptable': np.sum((df['quality_score'] >= 70) & (df['quality_score'] < 85)),
        'poor': np.sum(df['quality_score'] < 70)
    }
    
    print(f"📊 Quality Score Distribution:")
    print(f"   • Average quality score: {avg_quality_score:.1f}/100")
    print(f"   • Excellent (≥95): {quality_distribution['excellent']}/{n_valid} ({100*quality_distribution['excellent']/n_valid:.1f}%)")
    print(f"   • Good (85-94): {quality_distribution['good']}/{n_valid} ({100*quality_distribution['good']/n_valid:.1f}%)")
    print(f"   • Acceptable (70-84): {quality_distribution['acceptable']}/{n_valid} ({100*quality_distribution['acceptable']/n_valid:.1f}%)")
    print(f"   • Poor (<70): {quality_distribution['poor']}/{n_valid} ({100*quality_distribution['poor']/n_valid:.1f}%)")
    
    # Overall dataset score
    overall_score = np.mean([integrity_score, snr_score, range_score, tissue_score, count_score, avg_quality_score])
    
    print(f"\n📈 COMPOSITE DATASET SCORE: {overall_score:.1f}/100")
    
    if overall_score >= 95:
        print("🏆 EXCELLENT - Dataset ready for production ML training")
        dataset_verdict = "EXCELLENT"
    elif overall_score >= 85:
        print("✅ GOOD - Dataset suitable for ML training with minor monitoring")
        dataset_verdict = "GOOD"
    elif overall_score >= 70:
        print("⚠️  ACCEPTABLE - Dataset usable but may need some cleanup")
        dataset_verdict = "ACCEPTABLE"
    else:
        print("❌ POOR - Dataset needs significant cleanup before training")
        dataset_verdict = "POOR"
    
    # =================================================================
    # GENERATE ENHANCED VISUALIZATION PLOTS
    # =================================================================
    
    print(f"\n📊 Generating comprehensive validation plots...")
    
    # Create enhanced validation plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Master Phantom Dataset Validation Report ({n_valid} phantoms)', fontsize=16, fontweight='bold')
    
    # Plot 1: Quality Score Distribution
    axes[0,0].hist(df['quality_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Quality Score Distribution')
    axes[0,0].set_xlabel('Quality Score')
    axes[0,0].set_ylabel('Count')
    axes[0,0].axvline(df['quality_score'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {df["quality_score"].mean():.1f}')
    axes[0,0].legend()
    
    # Plot 2: SNR Distribution
    axes[0,1].hist(df['amplitude_snr_db'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title('SNR Distribution')
    axes[0,1].set_xlabel('SNR [dB]')
    axes[0,1].set_ylabel('Count')
    axes[0,1].axvline(QUALITY_THRESHOLDS['target_snr_db'], color='red', linestyle='--', 
                     label=f'Target: {QUALITY_THRESHOLDS["target_snr_db"]} dB')
    axes[0,1].legend()
    
    # Plot 3: Log Amplitude Range Distribution
    axes[0,2].hist(df['log_amp_range'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[0,2].set_title('Log Amplitude Range Distribution')
    axes[0,2].set_xlabel('Log Amplitude Range')
    axes[0,2].set_ylabel('Count')
    
    # Plot 4: Phase Range Distribution
    axes[1,0].hist(df['phase_range'], bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_title('Phase Range Distribution')
    axes[1,0].set_xlabel('Phase Range [°]')
    axes[1,0].set_ylabel('Count')
    
    # Plot 5: Tissue Coverage
    axes[1,1].hist(df['tissue_percentage'], bins=20, alpha=0.7, color='brown', edgecolor='black')
    axes[1,1].set_title('Tissue Coverage Distribution')
    axes[1,1].set_xlabel('Tissue Coverage [%]')
    axes[1,1].set_ylabel('Count')
    
    # Plot 6: Tumor Count Distribution
    max_tumors = int(df['n_tumors'].max()) + 1
    axes[1,2].hist(df['n_tumors'], bins=range(max_tumors + 1), alpha=0.7, color='red', edgecolor='black', align='left')
    axes[1,2].set_title('Tumor Count Distribution')
    axes[1,2].set_xlabel('Number of Tumors')
    axes[1,2].set_ylabel('Count')
    axes[1,2].set_xticks(range(max_tumors))
    
    # Plot 7: Processing Time Distribution
    axes[2,0].hist(df['processing_time_ms'], bins=20, alpha=0.7, color='teal', edgecolor='black')
    axes[2,0].set_title('Processing Time Distribution')
    axes[2,0].set_xlabel('Processing Time [ms]')
    axes[2,0].set_ylabel('Count')
    
    # Plot 8: File Size Distribution
    axes[2,1].hist(df['file_size_mb'], bins=20, alpha=0.7, color='gold', edgecolor='black')
    axes[2,1].set_title('File Size Distribution')
    axes[2,1].set_xlabel('File Size [MB]')
    axes[2,1].set_ylabel('Count')
    
    # Plot 9: Quality vs SNR Scatter
    scatter = axes[2,2].scatter(df['amplitude_snr_db'], df['quality_score'], 
                               c=df['tissue_percentage'], alpha=0.6, cmap='viridis')
    axes[2,2].set_title('Quality Score vs SNR')
    axes[2,2].set_xlabel('SNR [dB]')
    axes[2,2].set_ylabel('Quality Score')
    plt.colorbar(scatter, ax=axes[2,2], label='Tissue %')
    
    # Plot 10: Issue Summary Bar Chart
    issue_counts = [
        issue_summary['file_corruption'],
        issue_summary['nan_inf_values'], 
        issue_summary['snr_issues'],
        issue_summary['measurement_count_wrong'],
        issue_summary['tissue_issues'],
        issue_summary['extreme_outliers']
    ]
    issue_labels = ['File\nCorruption', 'NaN/Inf\nValues', 'SNR\nIssues', 
                   'Wrong\nCount', 'Tissue\nIssues', 'Extreme\nOutliers']
    
    colors = ['red' if count > 0 else 'green' for count in issue_counts]
    bars = axes[3,0].bar(issue_labels, issue_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[3,0].set_title('Issue Summary')
    axes[3,0].set_ylabel('Count')
    axes[3,0].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, issue_counts):
        if count > 0:
            axes[3,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                          str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 11: Optical Properties Scatter
    if 'mua_healthy_mean' in df.columns and df['mua_healthy_mean'].sum() > 0:
        axes[3,1].scatter(df['mua_healthy_mean'], df['musp_healthy_mean'], 
                         alpha=0.6, color='navy', edgecolors='white')
        axes[3,1].set_title('Optical Properties Distribution')
        axes[3,1].set_xlabel('μₐ [mm⁻¹]')
        axes[3,1].set_ylabel('μ′s [mm⁻¹]')
    else:
        axes[3,1].text(0.5, 0.5, 'No optical\nproperties data', ha='center', va='center', 
                      transform=axes[3,1].transAxes)
        axes[3,1].set_title('Optical Properties')
    
    # Plot 12: Dataset Health Summary
    health_metrics = {
        'Data Integrity': integrity_score,
        'SNR Quality': snr_score,
        'Range Validity': range_score,
        'Tissue Coverage': tissue_score,
        'Count Consistency': count_score
    }
    
    metric_names = list(health_metrics.keys())
    metric_values = list(health_metrics.values())
    colors = ['green' if v >= 95 else 'orange' if v >= 80 else 'red' for v in metric_values]
    
    bars = axes[3,2].bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    axes[3,2].set_title('Dataset Health Metrics')
    axes[3,2].set_ylabel('Score [%]')
    axes[3,2].set_ylim(0, 100)
    axes[3,2].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, value in zip(bars, metric_values):
        axes[3,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                      f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'master_phantom_validation_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =================================================================
    # SAVE COMPREHENSIVE RESULTS
    # =================================================================
    
    # Save detailed statistics
    df.to_csv(output_path / 'master_phantom_validation_data.csv', index=False)
    
    # Save outlier analysis
    if outlier_results:
        outlier_df_data = []
        for metric, data in outlier_results.items():
            for phantom, value, z_score in zip(data['phantoms'], data['values'], data['z_scores']):
                outlier_df_data.append({
                    'phantom_id': phantom,
                    'metric': metric,
                    'value': value,
                    'z_score': z_score,
                    'metric_mean': data['mean'],
                    'metric_std': data['std']
                })
        
        outlier_df = pd.DataFrame(outlier_df_data)
        outlier_df.to_csv(output_path / 'outlier_analysis.csv', index=False)
    
    # Save summary report
    summary_report = {
        'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_path': str(data_path),
        'phantoms_processed': int(n_valid),
        'phantoms_failed': len(failed_phantoms),
        'processing_time_seconds': float(total_time),
        'overall_score': float(overall_score),
        'dataset_verdict': dataset_verdict,
        'integrity_score': float(integrity_score),
        'snr_score': float(snr_score),
        'range_score': float(range_score),
        'tissue_score': float(tissue_score),
        'count_score': float(count_score),
        'issue_summary': {k: int(v) for k, v in issue_summary.items()},
        'outlier_count': int(total_outliers),
        'extreme_outlier_count': len(extreme_outliers),
        'quality_distribution': {k: int(v) for k, v in quality_distribution.items()}
    }
    
    import json
    with open(output_path / 'validation_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    # =================================================================
    # FINAL VALIDATION VERDICT
    # =================================================================
    
    print(f"\n🎯 FINAL MASTER VALIDATION VERDICT")
    print("=" * 80)
    
    # List any critical issues that need attention
    critical_issues = []
    
    if integrity_score < 95:
        critical_issues.append("Data integrity problems")
    
    if snr_score < 90:
        critical_issues.append("SNR consistency issues")
    
    if count_score < 95:
        critical_issues.append("Measurement count problems")
    
    if len(extreme_outliers) > n_valid * 0.05:  # More than 5% extreme outliers
        critical_issues.append("Too many extreme outliers")
    
    if len(critical_issues) == 0:
        print("🎉 VALIDATION PASSED WITH FLYING COLORS!")
        print("✅ Dataset is production-ready for ML training")
        print("✅ All quality metrics meet or exceed expectations")
        print("✅ No critical issues detected")
        print("✅ Ready for large-scale model training")
        status = "EXCELLENT"
    else:
        print("⚠️  VALIDATION COMPLETED WITH ISSUES:")
        for issue in critical_issues:
            print(f"   • {issue}")
        
        if overall_score >= 70:
            print("\n✅ Dataset is still usable but monitor the flagged issues")
            status = "USABLE_WITH_MONITORING"
        else:
            print("\n❌ Recommend addressing issues before training")
            status = "NEEDS_ATTENTION"
    
    print(f"\n📊 Comprehensive analysis complete!")
    print(f"📈 Master validation report: {output_path}/master_phantom_validation_report.png")
    print(f"📄 Detailed data: {output_path}/master_phantom_validation_data.csv")
    print(f"🔍 Outlier analysis: {output_path}/outlier_analysis.csv")
    print(f"📋 Summary report: {output_path}/validation_summary.json")
    
    return {
        'status': status,
        'overall_score': overall_score,
        'dataset_verdict': dataset_verdict,
        'phantoms_processed': n_valid,
        'critical_issues': critical_issues,
        'summary': summary_report
    }
