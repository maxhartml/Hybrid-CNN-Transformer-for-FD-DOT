# Tissue Patch Journey Documentation

## Overview
Complete implementation of tissue patch extraction and integration for enhanced NIR-DOT reconstruction. This document traces the complete journey of tissue patches from extraction to final model output.

## 🎯 The Complete Tissue Patch Journey

### **Stage 1: Data Generation & Storage**
- **Location**: `data_simulator.py` 
- **Action**: Phantoms generated with tissue structures
- **Output**: HDF5 files with ground truth volumes (2, 64, 64, 64) - [μ_a, μ_s]

### **Stage 2: Data Loading & Patch Extraction** ✅ **NEWLY IMPLEMENTED**
- **Location**: `data_loader.py` → `extract_tissue_patches_from_measurements()`
- **Input**: 
  - Ground truth volume: (2, 64, 64, 64)
  - Source positions: (1000, 3) in mm
  - Detector positions: (1000, 3) in mm  
  - Measurement indices: (256,) subsampled
- **Process**: 
  1. Convert mm positions to voxel coordinates 
  2. Extract 16×16×16 patches around each source/detector
  3. Handle edge cases with padding
  4. Flatten and interleave channels: [μ_a[0], μ_s[0], μ_a[1], μ_s[1], ...]
- **Output**: `tissue_patches` (256, 2, 8192) where 8192 = 16³ × 2 channels

### **Stage 3: Training Data Flow** ✅ **ENHANCED**
- **Location**: `stage2_trainer.py`
- **Input**: Batch with keys: `['nir_measurements', 'ground_truth', 'tissue_patches', 'phantom_id']`
- **Shape**: `tissue_patches` (batch_size, 256, 2, 8192)
- **Process**: Move to device, validate shapes, pass to model

### **Stage 4: NIR Processing** ✅ **READY**
- **Location**: `nir_processor.py` → `PerMeasurementTissueEncoder`
- **Input**: (batch, 2, 8192) - interleaved tissue patches
- **Process**:
  1. Reshape interleaved data → (batch, 2, 2, 16, 16, 16)
  2. 3D CNN encoding → (batch×2, 4) features per patch
  3. Concatenate → (batch, 8) total tissue context
- **Output**: 8D tissue context vector per measurement

### **Stage 5: Enhanced NIR Features** ✅ **READY**
- **Location**: `nir_processor.py` → `SimplifiedNIRProcessor.forward()`
- **Input**: 
  - NIR measurements: (batch, 8) 
  - Tissue patches: (batch, 2, 8192)
- **Process**: Project NIR features (8D) + tissue context (8D) → spatial features (256D)
- **Output**: Enhanced features (batch, 256) with tissue context

### **Stage 6: Transformer Enhancement** ✅ **READY**
- **Location**: `transformer_encoder.py`
- **Input**: Enhanced features (batch, 256) with tissue context already integrated
- **Process**: Multi-head attention and spatial modeling
- **Output**: Transformer-enhanced features (batch, 256)

### **Stage 7: CNN Reconstruction** ✅ **READY**  
- **Location**: `cnn_autoencoder.py` → decode()
- **Input**: Enhanced features (batch, 256)
- **Process**: Frozen CNN decoder (trained in Stage 1)
- **Output**: Final reconstruction (batch, 2, 64, 64, 64)

## 🔧 Implementation Details

### **Patch Extraction Specifications**
- **Patch Size**: 16×16×16 voxels (16mm³ at 1mm resolution)
- **Channels**: 2 (absorption μ_a, scattering μ_s)
- **Format**: Interleaved [μ_a[0], μ_s[0], μ_a[1], μ_s[1], ...] for CNN compatibility
- **Patches per Measurement**: 2 (source location + detector location)
- **Edge Handling**: Zero-padding when patches extend beyond volume boundaries

### **Data Flow Shapes**
```
Raw Ground Truth:     (2, 64, 64, 64)
Source/Det Positions: (1000, 3) mm → converted to voxel indices
Extracted Patches:    (256, 2, 8192) - 256 measurements × 2 patches × interleaved data
Tissue Encoding:      (256, 8) - 256 measurements × 8D context
Enhanced Features:    (256, 256) - with tissue context integrated
Final Reconstruction: (batch, 2, 64, 64, 64)
```

### **Key Functions Implemented**
1. **`extract_tissue_patches_from_measurements()`** - Core extraction logic
2. **Enhanced `NIRPhantomDataset.__getitem__()`** - Returns tissue patches in batch
3. **Enhanced Stage2Trainer logging** - Detailed tissue patch debugging
4. **Ready NIR processor integration** - Handles interleaved format correctly

## 🎊 Benefits of Implementation

### **Enhanced Spatial Modeling**
- **Local Context**: 16×16×16 patches provide detailed tissue properties around each measurement
- **Anatomical Constraints**: Real tissue structure guides reconstruction
- **Measurement-Specific**: Each NIR measurement gets its own tissue context

### **Improved Performance**
- **Baseline vs Enhanced**: Easy A/B testing with `use_tissue_patches` flag
- **Data Augmentation**: Different measurement subsets provide varied tissue contexts
- **Gradient Flow**: Tissue context enables better spatial feature learning

### **Production Ready**
- **Memory Efficient**: Patches extracted on-demand during data loading
- **Hardware Optimized**: Interleaved format optimized for CNN processing
- **Robust Edge Handling**: Handles boundary cases gracefully
- **Comprehensive Logging**: Full debugging and monitoring support

## 🚀 Ready for Training

The complete tissue patch pipeline is now implemented and ready for:

1. **Stage 1 Training**: CNN autoencoder (tissue patches not used)
2. **Stage 2 Enhanced Training**: Full pipeline with tissue context
3. **Stage 2 Baseline Training**: Set `USE_TISSUE_PATCHES_STAGE2 = False` for comparison

All tissue patch extraction, processing, and integration is complete and production-ready!

---

**Date**: August 7, 2025  
**Status**: ✅ COMPLETE - Ready for Production Training
**Next Step**: Begin Stage 1 training to prepare for enhanced Stage 2 evaluation
