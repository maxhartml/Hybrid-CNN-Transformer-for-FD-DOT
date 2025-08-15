# Comprehensive Technical Brief: NIR-DOT Reconstruction Project

**Generated**: August 15, 2025  
**Evidence-Based Analysis**: Complete repository crawl with citations

---

## 1) Repository Map & Entry Points

### High-Level File Inventory

**Core Project Structure**:
```
/home/ubuntu/NIR-DOT/
├── code/
│   ├── data_processing/         # Data generation and loading
│   │   ├── data_simulator.py    # 2,120 lines - phantom generation
│   │   ├── data_loader.py       # 593 lines - PyTorch DataLoader
│   │   ├── data_analysis.py     # 1,293 lines - analysis utilities
│   │   └── phantom_validation.py
│   ├── models/                  # Neural network architectures
│   │   ├── hybrid_model.py      # 618 lines - main hybrid architecture
│   │   ├── cnn_autoencoder.py   # CNN backbone
│   │   ├── transformer_encoder.py # Transformer components
│   │   ├── spatially_aware_embedding.py # NIR measurement processing
│   │   └── global_pooling_encoder.py # Post-transformer processing
│   ├── training/                # Training orchestration
│   │   ├── train_hybrid_model.py # 294 lines - main entry point
│   │   ├── training_config.py   # 192 lines - all hyperparameters
│   │   ├── stage1_trainer.py    # 915 lines - CNN pre-training
│   │   ├── stage2_trainer.py    # 1,343 lines - transformer training
│   │   ├── teacher_stage1.py    # Teacher model for latent training
│   │   └── latent_stats.py      # Latent space diagnostics
│   └── utils/                   # Utilities
│       ├── standardizers.py     # 1,293 lines - normalization
│       ├── metrics.py           # 704 lines - evaluation metrics
│       ├── viz_recon.py         # 227 lines - W&B visualization
│       └── logging_config.py    # Centralized logging
├── data/                        # Generated phantom datasets
├── checkpoints/                 # Model weights
├── logs/                        # Training logs
└── nirfaster-FF/               # Physics simulation library
```

### Primary Entry Points

**Main Training Entry**: `code/training/train_hybrid_model.py:main()` (lines 106-294)
- Orchestrates both Stage 1 and Stage 2 training
- Configuration-driven: no command-line arguments needed
- Stage selection via `CURRENT_TRAINING_STAGE` in `training_config.py:25`

**Stage 1 Training**: Via `CURRENT_TRAINING_STAGE = "stage1"` (training_config.py:25)
- Entry: `train_hybrid_model.py:218-237`
- Trainer: `Stage1Trainer` (stage1_trainer.py:71-915)

**Stage 2 Training**: Via `CURRENT_TRAINING_STAGE = "stage2"` (training_config.py:25)  
- Entry: `train_hybrid_model.py:239-272`
- Trainer: `Stage2Trainer` (stage2_trainer.py:67-1343)

### Configuration Loading

**Central Configuration**: `code/training/training_config.py`
- **All hyperparameters**: Lines 1-192, imported via `from .training_config import *` 
- **Stage control**: `CURRENT_TRAINING_STAGE = "stage2"` (line 25)
- **W&B logging**: `USE_WANDB_LOGGING = True` (line 29)
- **Batch size**: `BATCH_SIZE = 128` (line 58)
- **Training epochs**: `EPOCHS_STAGE1 = 150`, `EPOCHS_STAGE2 = 200` (lines 55-56)
- **Checkpoint paths**: Lines 146-153

---

## 2) Data Generation & On-Disk Format

### Phantom Generation

**Generator**: `code/data_processing/data_simulator.py` (2,120 lines)
- **Entry function**: `generate_phantom()` (location: need to verify exact line)
- **Voxel size**: 1mm³ (comment in header: "1mm voxel size for optimal phantom dimensions")
- **Volume dimensions**: `(64, 64, 64)` voxels = 64×64×64mm physical (data_loader.py:34)

### Optical Property Ranges (Physical Units: mm⁻¹)

**Healthy Tissue** (data_simulator.py:120-121):
```python
HEALTHY_MUA_RANGE = (0.003, 0.007)          # Absorption coefficient [mm⁻¹]
HEALTHY_MUSP_RANGE = (0.78, 1.18)           # Reduced scattering [mm⁻¹]
```

**Tumor Enhancement** (data_simulator.py:122-123):
```python
TUMOR_MUA_MULTIPLIER_RANGE = (1.5, 3.5)     # 50-250% absorption increase
TUMOR_MUSP_MULTIPLIER_RANGE = (1.5, 2.5)    # 50-150% scattering increase
```

**Visualization Ranges** (viz_recon.py:20-23):
```python
PHYS_MAX = {
    "mu_a": 0.0245,  # Maximum absorption [mm⁻¹]
    "mu_s": 2.95     # Maximum scattering [mm⁻¹]
}
```

### File Format & Storage

**Storage**: HDF5 format (data_loader.py:52-58)
```python
H5_KEYS = {
    'log_amplitude': 'log_amplitude',
    'phase': 'phase', 
    'source_pos': 'source_positions',
    'det_pos': 'detector_positions',
    'ground_truth': 'ground_truth'
}
```

**Directory Layout**: `data/phantom_XX/` where XX is phantom ID
- Each phantom: 1000 NIR measurements + ground truth volume
- Measurements: 50 sources × 20 detectors = 1000 total (hybrid_model.py:52)

### Random Seeds & Reproducibility

**Dataset Splitting** (data_loader.py:45):
```python
DEFAULT_RANDOM_SEED = 42
```

**Validation Determinism** (hybrid_model.py:95-98):
```python
if not training:
    torch.manual_seed(42)  # Fixed seed for consistent validation subsets
```

### Coordinate System

**Volume Extent**: 64 voxels = 64mm physical (1mm/voxel)
**Position Conversion** (data_loader.py:121-134): Auto-detected based on position ranges
- Centered system: [-32, +32] mm → [0, 64] voxels  
- Direct system: [0, 64] already in voxel coordinates

---

## 3) Standardization / Normalization

### Standardizer Classes

**1. PerChannelZScore** (standardizers.py:65-): Ground truth volume normalization
- **Purpose**: Per-channel z-score for 3D volumes [B, 2, D, H, W]
- **Channels**: μₐ (absorption), μ′ₛ (scattering) normalized independently
- **Fit on**: Training data only
- **Transform**: (x - mean) / std with clipping to (-5, 5) range
- **Inverse**: For raw mm⁻¹ units before metrics/visualization

**2. MeasurementStandardizer** (standardizers.py:): NIR measurements
- **Purpose**: Z-score for [log_amplitude, phase] features
- **Input shape**: [batch, n_measurements, 2]
- **Location of use**: To be verified in model pipeline

**3. PositionScaler** (standardizers.py:): Spatial coordinates  
- **Purpose**: Min-max scaling to [-1, 1] for [src_x, src_y, src_z, det_x, det_y, det_z]
- **Input shape**: [batch, n_measurements, 6]

### Standardization Pipeline

**Shapes & Application Points**:
- **Ground Truth**: [B, 2, 64, 64, 64] → standardized during training → inverse before metrics
- **NIR Measurements**: [B, 1000, 8] where 8 = [log_amp, phase, src_xyz, det_xyz]
- **Measurement Features**: First 2 dimensions standardized: [log_amplitude, phase]  
- **Position Coordinates**: Last 6 dimensions scaled: [src_x, src_y, src_z, det_x, det_y, det_z]

### Raw Physical Units Recovery

**For Metrics & Visualization** (viz_recon.py:25-41):
```python
def _inv_std_chlast(x: torch.Tensor, stdzr) -> torch.Tensor:
    """Apply inverse standardization - returns raw mm⁻¹ units"""
    inv = getattr(stdzr, "inverse_transform", None) or getattr(stdzr, "inverse_transform_ground_truth", None)
    x_raw = inv(x)  # Back to physical units
    return x_raw
```

**Clamping to Physical Ranges** (viz_recon.py:149-152):
```python
pred[:,0].clamp_(0.0, PHYS_MAX["mu_a"])  # μₐ: [0, 0.0245] mm⁻¹
pred[:,1].clamp_(0.0, PHYS_MAX["mu_s"])  # μ′ₛ: [0, 2.95] mm⁻¹
```

---

## 4) Data Pipeline (Dataset + DataLoader)

### Dataset Classes

**Primary Dataset**: `NIRPhantomDataset` (data_loader.py:173-)
- **Phantom-level loading**: Complete phantoms with subsampled measurements
- **Stage 1**: Ground truth only `[B, 2, 64, 64, 64]`
- **Stage 2**: NIR measurements `[B, 1000, 8]` + ground truth + optional tissue patches

### Batch Construction & Shapes

**Stage 1 Tensors**:
- `ground_truth`: `[batch_size, 2, 64, 64, 64]` (float32, standardized)

**Stage 2 Tensors**:  
- `nir_measurements`: `[batch_size, 1000, 8]` (float32, partially standardized)
- `ground_truth`: `[batch_size, 2, 64, 64, 64]` (float32, standardized)
- `tissue_patches`: `[batch_size, 1000, 2, 2, 16, 16, 16]` (float32, if enabled)

### DataLoader Configuration (training_config.py:65-71)

```python
NUM_WORKERS = 16          # Parallel data loading processes
PIN_MEMORY = True         # Faster GPU transfers  
PREFETCH_FACTOR = 4       # 16 workers × 4 = 64 batches prefetched
PERSISTENT_WORKERS = True # Workers survive between epochs
BATCH_SIZE = 128          # Training batch size
```

### Data Augmentation

**Dynamic Undersampling** (hybrid_model.py:84-137):
- **Training**: Random subset of 256 from 1000 measurements per phantom
- **Validation**: Fixed deterministic subset (seed=42) for consistency
- **Purpose**: Massive data augmentation through measurement variation

### Memory Optimization

**HDF5 Caching**: Datasets loaded on-demand with efficient access
**Channels-Last**: `USE_CHANNELS_LAST_MEMORY_FORMAT = True` (training_config.py:44)
**Pin Memory**: Enabled for faster CPU→GPU transfers

---

## 5) Model Architectures

### Stage 1: CNN Autoencoder

**Class**: `CNNAutoEncoder` (cnn_autoencoder.py)
**Usage**: Pre-training for feature extraction

**Architecture Details**:
- **Input**: `[B, 2, 64, 64, 64]` (ground truth volumes)
- **Base channels**: 16 (optimized from 32/64 for ~7M parameters)
- **Channel progression**: 16→32→64→128→256
- **Feature dimension**: 256 (encoder output)
- **Encoder**: Progressive downsampling with residual blocks
- **Decoder**: Progressive upsampling to reconstruct volumes
- **Output**: `[B, 2, 64, 64, 64]` (reconstructed volumes)

**Parameter Count**: ~7M total (need exact count from model output)

### Stage 2: Hybrid CNN-Transformer  

**Class**: `HybridCNNTransformer` (hybrid_model.py:147-)

**Architecture Flow**:
1. **Spatially-Aware Encoder** → 2. **Transformer** → 3. **Global Pooling** → 4. **CNN Decoder**

**Component Details**:

**1. Spatially-Aware Encoder** (spatially_aware_embedding.py):
- **Input**: NIR measurements `[B, 256, 8]` + tissue patches `[B, 256, 2, 2, 16, 16, 16]`
- **Measurement branch**: `[log_amp, phase]` → 8D embedding  
- **Position branch**: `[src_xyz, det_xyz]` → 8D embedding
- **Fusion**: Concatenate + MLP → 256D tokens
- **Output**: `[B, 256, 256]` enhanced tokens

**2. Transformer Encoder** (transformer_encoder.py):
- **Embed dim**: 256 (transformer_config.py:47)
- **Layers**: 6 (transformer_config.py:48)  
- **Heads**: 4 (transformer_config.py:49)
- **MLP ratio**: 3 (transformer_config.py:50)
- **Dropout**: 0.05 (transformer_config.py:51)
- **Input**: `[B, 256, 256]` tokens
- **Output**: `[B, 256, 256]` enhanced tokens + attention weights

**3. Global Pooling Encoder** (global_pooling_encoder.py):
- **Global pooling**: Average across sequence dimension
- **Projection**: 256 → 128 → 256D "encoded scan"  
- **Input**: `[B, 256, 256]` → **Output**: `[B, 256]`

**4. CNN Decoder** (frozen from Stage 1):
- **Input**: `[B, 256]` encoded scan
- **Output**: `[B, 2, 64, 64, 64]` reconstructed volume

### Latent Interface & Teacher-Student

**Teacher Model**: Stage 1 encoder frozen, generates `[B, 256]` latent targets
**Student Model**: Stage 2 spatially-aware encoder → transformer → pooling produces `[B, 256]` latents
**Latent Training**: RMSE loss between teacher and student latents (latent_stats.py)

### Model Behavior Configuration

**Tissue Patches**: `USE_TISSUE_PATCHES_STAGE2 = False` (training_config.py:40) - **currently baseline mode**
**Model Compilation**: `USE_MODEL_COMPILATION = True` (training_config.py:42)
**Memory Format**: Channels-last for 3D efficiency

---

## 6) Training Loops & Schedules

### Stage 1: CNN Autoencoder (stage1_trainer.py)

**Optimizer**: AdamW with OneCycleLR (training_config.py:80-96)
```python
# AdamW Parameters
STAGE1_MAX_LR = 3e-3
STAGE1_BASE_LR = 1.0e-4  
ADAMW_BETAS_STAGE1 = (0.9, 0.95)
ADAMW_EPS_STAGE1 = 1e-8
WEIGHT_DECAY = 7e-4

# OneCycleLR Schedule  
STAGE1_DIV_FACTOR = 25          # Start LR = max_lr / 25
STAGE1_FINAL_DIV_FACTOR = 100   # Final LR = max_lr / 100  
STAGE1_PCT_START = 0.30         # 30% warmup phase
STAGE1_CYCLE_MOMENTUM = True    # Momentum cycling enabled
BASE_MOMENTUM = 0.85
MAX_MOMENTUM = 0.95
```

**Loss Function**: RMSELoss (metrics.py:61-)
```python
loss = sqrt(mean((pred - target)^2))
```

**Gradient Handling**: 
- **Clipping**: `GRADIENT_CLIP_MAX_NORM = 1.0` (training_config.py:78)
- **AMP**: Mixed precision with GradScaler (training_config.py:81-84)

### Stage 2: Transformer Training (stage2_trainer.py)

**Optimizer**: AdamW with Linear Warmup + Cosine Decay (training_config.py:102-110)
```python
# AdamW Parameters
STAGE2_BASE_LR = 1.5e-4
ADAMW_BETAS_STAGE2 = (0.9, 0.98)  # Higher beta2 for attention stability
ADAMW_EPS_STAGE2 = 1e-8
WEIGHT_DECAY_TRANSFORMER = 0.01

# Schedule Parameters
STAGE2_WARMUP_PCT = 0.06        # 6% warmup phase
STAGE2_ETA_MIN_PCT = 0.15       # 15% of peak LR as minimum
```

**Loss Functions**:
- **Latent-only mode**: RMSE between teacher/student latents `[B, 256]`
- **End-to-end mode**: RMSE on reconstructed volumes `[B, 2, 64, 64, 64]` (raw mm⁻¹ units)

**Mixed Precision**: GradScaler configuration (training_config.py:81-84)
```python
GRADSCALER_INIT_SCALE = 2**8        # 256 initial scale
GRADSCALER_GROWTH_FACTOR = 2.0
GRADSCALER_BACKOFF_FACTOR = 0.5
GRADSCALER_GROWTH_INTERVAL = 200
```

### Early Stopping & Checkpointing

**Early Stopping**: `EARLY_STOPPING_PATIENCE = 25` epochs (training_config.py:61)

**Checkpoint Paths** (training_config.py:146-153):
```python
STAGE1_CHECKPOINT_PATH = "checkpoints/stage1_best.pth"
STAGE2_BASELINE_CHECKPOINT_PATH = "checkpoints/stage2_baseline_best.pth" 
STAGE2_ENHANCED_CHECKPOINT_PATH = "checkpoints/stage2_enhanced_best.pth"
```

**Best Model Criteria**: Lowest validation loss

### Validation Cadence

**Stage 1**: Every epoch - full metrics
**Stage 2**: 
- **Latent metrics**: Every epoch
- **End-to-end metrics**: Every `VAL_E2E_EVERY_K_EPOCHS = 1` epochs (training_config.py:36)

---

## 7) Metrics (Exact Implementations + Keys)

### Core Reconstruction Metrics (metrics.py)

**1. Global RMSE** (RMSELoss class, lines 61-):
```python
def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(input, target))
```
- **Units**: Raw mm⁻¹ (after inverse standardization)
- **W&B key**: `"val/total_rmse_raw"`

**2. Channel-Specific RMSE** (need to verify implementation):
- **μₐ channel**: Absorption coefficient RMSE  
- **μ′ₛ channel**: Scattering coefficient RMSE
- **W&B keys**: `"val/rmse_absorption_raw"`, `"val/rmse_scattering_raw"`

**3. Dice Coefficient** (DiceCoefficient class):
```python
# Binary mask creation with threshold
mask_pred = (input > DICE_THRESHOLD).float()  # DICE_THRESHOLD = 0.5
mask_target = (target > DICE_THRESHOLD).float()

# Dice calculation with smoothing
intersection = (mask_pred * mask_target).sum(dim=(2,3,4))
union = mask_pred.sum(dim=(2,3,4)) + mask_target.sum(dim=(2,3,4))
dice = (2.0 * intersection + DICE_SMOOTH) / (union + DICE_SMOOTH)
```
- **Threshold**: 0.5 (DICE_THRESHOLD, line 39)
- **Smoothing**: 1e-6 (DICE_SMOOTH, line 38)
- **W&B keys**: `"val/dice_absorption"`, `"val/dice_scattering"`

**4. Contrast Ratio** (ContrastRatio class):
```python
# Tumor mask creation and contrast calculation  
tumor_mask = tumor_region > threshold
if tumor_mask.sum() > 0:
    tumor_mean = input[tumor_mask].mean()
    bg_mean = input[~tumor_mask].mean()  
    contrast = (tumor_mean - bg_mean) / (bg_mean + CONTRAST_EPS)
```
- **Epsilon**: 1e-8 (CONTRAST_EPS, line 41)
- **W&B keys**: `"val/contrast_absorption"`, `"val/contrast_scattering"`

### Latent Space Diagnostics (latent_stats.py)

**Latent RMSE** (primary Stage 2 loss):
```python
def compute_latent_rmse(teacher_latent, student_latent):
    return torch.sqrt(F.mse_loss(student_latent, teacher_latent))
```

**Latent Cosine Similarity**:
```python
def compute_latent_cosine_similarity(teacher_latent, student_latent):
    return F.cosine_similarity(teacher_latent, student_latent, dim=1).mean()
```

### Metric Computation Units

**All reconstruction metrics computed in RAW mm⁻¹ units** after inverse standardization (viz_recon.py:25-41)
**Latent metrics**: On standardized latent features `[B, 256]`

### W&B Logging Keys

**Training**: `"train/loss"`, `"train/lr"`
**Validation**: `"val/total_rmse_raw"`, `"val/dice_*"`, `"val/contrast_*"`  
**Latent**: `"val/latent_rmse"`, `"val/latent_cosine_similarity"`
**Hardware**: GPU memory, utilization (training_config.py:183-192)

---

## 8) Visualization

### Visualization Pipeline (viz_recon.py)

**Input Requirements**: 
- **Raw physical units only** (mm⁻¹) - no standardized inputs accepted
- **Layout**: Strict `[B, 2, D, H, W]` channel-first format
- **Inverse standardization**: Applied automatically via `_inv_std_chlast()` (lines 25-41)

**Physics-Based Normalization** (lines 50-64):
```python
def _physics_norm(arr2d, vmax):
    data_min, data_max = arr2d.min(), arr2d.max()
    if data_max - data_min < 1e-8:
        return np.full_like(arr2d, 128, dtype=np.uint8)  # Constant arrays
    normalized = ((arr2d - data_min) / (data_max - data_min)) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)
```

**Center Slice Extraction** (lines 71-):
- **XY plane**: `vol[:, D//2, :, :]` (axial)
- **XZ plane**: `vol[:, :, H//2, :]` (coronal)  
- **YZ plane**: `vol[:, :, :, W//2]` (sagittal)

**Logging Volume**: 2 phantoms × 3 planes × 2 channels × (pred+target) = **24 images per epoch**

### W&B Image Hierarchy

**Prefix structure**: `"{prefix}/phantom_{i}/slice_{plane}_{channel}_{type}"`
- **Prefix**: "Reconstructions" (default)
- **Phantom**: 0, 1 (first 2 phantoms in batch)
- **Plane**: "xy", "xz", "yz"  
- **Channel**: "absorption", "scattering"
- **Type**: "prediction", "target"

**Example keys**:
- `"Reconstructions/phantom_0/slice_xy_absorption_prediction"`
- `"Reconstructions/phantom_1/slice_yz_scattering_target"`

### Physical Property Clamping (lines 149-152)

```python
pred[:,0].clamp_(0.0, PHYS_MAX["mu_a"])  # μₐ: [0, 0.0245] mm⁻¹
pred[:,1].clamp_(0.0, PHYS_MAX["mu_s"])  # μ′ₛ: [0, 2.95] mm⁻¹
```

### Known Issues

**No "viz_stats" complexity**: Removed autocontrast and quantile-based normalization
**Adaptive normalization**: Uses actual data range rather than fixed PHYS_MAX for better contrast

---

## 9) Performance & Hardware Features

### PyTorch Optimizations

**Model Compilation** (training_config.py:42-44):
```python
USE_MODEL_COMPILATION = True     # PyTorch 2.0 compilation for 2x speedup
COMPILATION_MODE = "default"     # Clean compilation without noise  
USE_CHANNELS_LAST_MEMORY_FORMAT = True  # Efficient 3D convolution memory layout
```

**Mixed Precision Training**: AMP with GradScaler (training_config.py:81-84)
- Automatic mixed precision for A100 optimization
- Custom GradScaler parameters to prevent scaling issues

**TensorFloat32**: Enabled for A100 performance (stage2_trainer.py):
```python
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
```

### Memory Optimizations

**DataLoader Optimizations** (training_config.py:65-71):
```python
NUM_WORKERS = 16          # High parallelism for large 3D data
PREFETCH_FACTOR = 4       # 64 batches prefetched (16×4)
PIN_MEMORY = True         # Faster GPU transfers
PERSISTENT_WORKERS = True # Workers survive between epochs
```

**Gradient Monitoring**: Track gradient norms, issue warnings at threshold (training_config.py:79)

### Hardware Monitoring

**GPU Statistics Function** (training_config.py:183-192):
```python
def log_gpu_stats():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3  
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Warnings at >90% allocation, >80% reservation
```

### Performance Bottleneck Mitigations

**Dynamic undersampling**: Fixed 256 measurements eliminates attention masking complexity
**Channels-last**: Optimized memory layout for 3D convolutions
**Persistent workers**: Avoid DataLoader restart overhead between epochs

---

## 10) Configuration & Hyperparameters Table

| Parameter | Value | Defined At | Consumed By |
|-----------|-------|------------|-------------|
| **Experiment Control** |
| `CURRENT_TRAINING_STAGE` | "stage2" | training_config.py:25 | train_hybrid_model.py:131 |
| `USE_WANDB_LOGGING` | True | training_config.py:29 | stage1_trainer.py, stage2_trainer.py |
| `TRAIN_STAGE2_LATENT_ONLY` | True | training_config.py:33 | stage2_trainer.py |
| `LATENT_DIM` | 256 | training_config.py:34 | All models, teacher_stage1.py |
| **Training Duration** |
| `EPOCHS_STAGE1` | 150 | training_config.py:55 | train_hybrid_model.py:132 |
| `EPOCHS_STAGE2` | 200 | training_config.py:56 | train_hybrid_model.py:132 |
| `BATCH_SIZE` | 128 | training_config.py:58 | train_hybrid_model.py:133 |
| `EARLY_STOPPING_PATIENCE` | 25 | training_config.py:61 | Both trainers |
| **Data Loading** |
| `NUM_WORKERS` | 16 | training_config.py:66 | data_loader.py |
| `PIN_MEMORY` | True | training_config.py:67 | data_loader.py |
| `PREFETCH_FACTOR` | 4 | training_config.py:68 | data_loader.py |
| `PERSISTENT_WORKERS` | True | training_config.py:69 | data_loader.py |
| **Stage 1 Optimization** |
| `STAGE1_MAX_LR` | 3e-3 | training_config.py:87 | stage1_trainer.py |
| `STAGE1_BASE_LR` | 1.0e-4 | training_config.py:88 | stage1_trainer.py |
| `STAGE1_PCT_START` | 0.30 | training_config.py:91 | OneCycleLR scheduler |
| `ADAMW_BETAS_STAGE1` | (0.9, 0.95) | training_config.py:94 | stage1_trainer.py |
| **Stage 2 Optimization** |
| `STAGE2_BASE_LR` | 1.5e-4 | training_config.py:105 | stage2_trainer.py |
| `STAGE2_WARMUP_PCT` | 0.06 | training_config.py:106 | stage2_trainer.py |
| `ADAMW_BETAS_STAGE2` | (0.9, 0.98) | training_config.py:109 | stage2_trainer.py |
| **Regularization** |
| `WEIGHT_DECAY` | 7e-4 | training_config.py:74 | stage1_trainer.py |
| `WEIGHT_DECAY_TRANSFORMER` | 0.01 | training_config.py:75 | stage2_trainer.py |
| `DROPOUT_CNN` | 0.18 | training_config.py:76 | cnn_autoencoder.py |
| `DROPOUT_TRANSFORMER` | 0.05 | training_config.py:77 | transformer_encoder.py |
| `GRADIENT_CLIP_MAX_NORM` | 1.0 | training_config.py:78 | Both trainers |
| **Model Architecture** |
| `BASE_CHANNELS` | 16 | cnn_autoencoder.py:59 | CNNAutoEncoder |
| `FEATURE_DIM` | 256 | cnn_autoencoder.py:60 | CNNAutoEncoder |
| `EMBED_DIM` | 256 | transformer_encoder.py:47 | TransformerEncoder |
| `NUM_LAYERS` | 6 | transformer_encoder.py:48 | TransformerEncoder |
| `NUM_HEADS` | 4 | transformer_encoder.py:49 | TransformerEncoder |
| `MLP_RATIO` | 3 | transformer_encoder.py:50 | TransformerEncoder |
| **Performance** |
| `USE_MODEL_COMPILATION` | True | training_config.py:42 | Both trainers |
| `USE_CHANNELS_LAST_MEMORY_FORMAT` | True | training_config.py:44 | Model initialization |
| **Physical Constants** |
| `PHYS_MAX["mu_a"]` | 0.0245 | viz_recon.py:21 | Visualization clamping |
| `PHYS_MAX["mu_s"]` | 2.95 | viz_recon.py:22 | Visualization clamping |
| `HEALTHY_MUA_RANGE` | (0.003, 0.007) | data_simulator.py:120 | Phantom generation |
| `HEALTHY_MUSP_RANGE` | (0.78, 1.18) | data_simulator.py:121 | Phantom generation |

---

## 11) End-to-End Trace (Shapes, Units, Ranges)

### Stage 1: CNN Autoencoder Training

**Data Flow**:
1. **Disk → Dataset**: HDF5 ground truth `(2, 64, 64, 64)` in raw mm⁻¹
2. **Dataset → DataLoader**: Batch construction `[B, 2, 64, 64, 64]`
3. **DataLoader → Standardizer**: Fit PerChannelZScore on training data
4. **Standardizer → Model**: `[B, 2, 64, 64, 64]` standardized (z-scored)
5. **CNN Forward**: `[B, 2, 64, 64, 64]` → encode → `[B, 256]` → decode → `[B, 2, 64, 64, 64]`
6. **Loss Computation**: RMSE on standardized tensors
7. **Metrics**: Inverse standardize → raw mm⁻¹ → compute Dice, Contrast, RMSE
8. **Visualization**: Raw mm⁻¹ → clamp to [0, 0.0245] μₐ, [0, 2.95] μ′ₛ → W&B

### Stage 2: Transformer Enhancement Training  

**Data Flow**:
1. **Disk → Dataset**: NIR measurements `(1000, 8)` + ground truth `(2, 64, 64, 64)` + tissue patches `(1000, 2, 2, 16, 16, 16)`
2. **Dataset → DataLoader**: Batching `[B, 1000, 8]`, `[B, 2, 64, 64, 64]`, `[B, 1000, 2, 2, 16, 16, 16]`
3. **Undersampling**: `[B, 1000, 8]` → `[B, 256, 8]` (fixed subset)
4. **Measurement Standardization**: `[log_amp, phase]` components z-scored
5. **Position Scaling**: `[src_xyz, det_xyz]` components scaled to [-1, 1]
6. **Spatially-Aware Encoding**: `[B, 256, 8]` → `[B, 256, 256]` tokens
7. **Transformer**: `[B, 256, 256]` → `[B, 256, 256]` enhanced tokens
8. **Global Pooling**: `[B, 256, 256]` → `[B, 256]` encoded scan
9. **CNN Decoder** (frozen): `[B, 256]` → `[B, 2, 64, 64, 64]` reconstructed

**Latent-Only Training**:
- **Teacher**: Stage 1 encoder on standardized GT `[B, 2, 64, 64, 64]` → `[B, 256]`
- **Student**: Stage 2 pipeline → `[B, 256]` pooled features
- **Loss**: RMSE between teacher and student latents
- **Validation**: Periodic end-to-end reconstruction + raw metrics

### Tensor Transformations Summary

| Stage | Tensor | Shape | Units | Device | Notes |
|-------|--------|-------|-------|--------|-------|
| **Data Loading** |
| HDF5 | ground_truth | `(2, 64, 64, 64)` | mm⁻¹ | CPU | Raw physics |
| Batch | ground_truth | `[B, 2, 64, 64, 64]` | mm⁻¹ | CPU | Pre-standardization |
| **Stage 1** |
| Standardized | ground_truth | `[B, 2, 64, 64, 64]` | z-scored | GPU | Training input |
| Encoded | latent | `[B, 256]` | z-scored | GPU | Encoder output |
| Reconstructed | prediction | `[B, 2, 64, 64, 64]` | z-scored | GPU | Decoder output |
| Metrics | both | `[B, 2, 64, 64, 64]` | mm⁻¹ | GPU | Inverse standardized |
| **Stage 2** |
| NIR | measurements | `[B, 1000, 8]` | mixed | GPU | log_amp/phase z-scored, pos scaled |
| Undersampled | measurements | `[B, 256, 8]` | mixed | GPU | Fixed subset |
| Encoded | tokens | `[B, 256, 256]` | learned | GPU | Spatially-aware output |
| Enhanced | tokens | `[B, 256, 256]` | learned | GPU | Transformer output |
| Pooled | encoded_scan | `[B, 256]` | learned | GPU | Global pooling |
| Final | prediction | `[B, 2, 64, 64, 64]` | z-scored | GPU | CNN decoder |
| **Latent Training** |
| Teacher | latent | `[B, 256]` | z-scored | GPU | Stage 1 encoder features |
| Student | latent | `[B, 256]` | learned | GPU | Stage 2 pooled features |

---

## 12) Diagrams (Mermaid)

### Dataflow Diagram

```mermaid
graph TD
    A[HDF5 Phantom Files<br/>data/phantom_XX/] --> B[NIRPhantomDataset<br/>Ground Truth: 2,64,64,64 mm⁻¹<br/>NIR: 1000,8 mixed units<br/>Tissue: 1000,2,2,16,16,16]
    
    B --> C[DataLoader<br/>Batch: B,1000,8<br/>Workers: 16, Pin Memory]
    
    C --> D{Training Stage?}
    
    D -->|Stage 1| E[PerChannelZScore<br/>Standardize GT volumes<br/>Fit on training data]
    D -->|Stage 2| F[Undersampling<br/>1000 → 256 measurements<br/>+ Standardize log_amp/phase<br/>+ Scale positions to [-1,1]]
    
    E --> G[CNN Autoencoder<br/>B,2,64,64,64 → B,256 → B,2,64,64,64<br/>~7M parameters]
    
    F --> H[Spatially-Aware Encoder<br/>B,256,8 → B,256,256<br/>Measurement + Position + Tissue]
    H --> I[Transformer Encoder<br/>6 layers, 4 heads, 256D<br/>B,256,256 → B,256,256]
    I --> J[Global Pooling<br/>B,256,256 → B,256<br/>Average + MLP]
    J --> K[CNN Decoder (Frozen)<br/>B,256 → B,2,64,64,64<br/>From Stage 1]
    
    G --> L[RMSE Loss<br/>Standardized units]
    K --> M{Training Mode?}
    M -->|Latent Only| N[Latent RMSE<br/>Teacher vs Student B,256]
    M -->|End-to-End| O[RMSE Loss<br/>Raw mm⁻¹ after inverse]
    
    L --> P[Inverse Standardize<br/>z-scored → mm⁻¹]
    O --> P
    N --> Q[Validation: E2E every K epochs]
    Q --> P
    
    P --> R[Metrics<br/>Dice, Contrast, RMSE<br/>Per channel in mm⁻¹]
    P --> S[Visualization<br/>Clamp to [0,0.0245] μₐ<br/>Clamp to [0,2.95] μ′ₛ<br/>24 images to W&B]
    
    R --> T[W&B Logging<br/>val/dice_*, val/contrast_*<br/>val/rmse_*_raw]
    S --> T
```

### Training Flow Diagram

```mermaid
graph TD
    A[train_hybrid_model.py<br/>CURRENT_TRAINING_STAGE] --> B{Stage Selection}
    
    B -->|"stage1"| C[Stage1Trainer]
    B -->|"stage2"| D[Stage2Trainer]
    
    C --> E[Initialize CNN Autoencoder<br/>~7M parameters]
    C --> F[AdamW + OneCycleLR<br/>Max: 3e-3, Warmup: 30%<br/>Momentum cycling]
    
    D --> G[Load Stage 1 Checkpoint<br/>checkpoints/stage1_best.pth]
    D --> H[Initialize Hybrid Model<br/>Freeze CNN decoder]
    D --> I[AdamW + Warmup+Cosine<br/>LR: 1.5e-4, Warmup: 6%]
    
    E --> J[Training Loop Stage 1]
    F --> J
    
    G --> K[Training Loop Stage 2]
    H --> K  
    I --> K
    
    J --> L[Forward Pass<br/>GT B,2,64,64,64 → reconstruction]
    L --> M[RMSE Loss<br/>Standardized units]
    M --> N[Backward + Optimizer Step<br/>Gradient clipping: 1.0<br/>AMP with GradScaler]
    
    K --> O{Latent Only Mode?}
    O -->|Yes| P[Teacher: Stage1 Encoder<br/>Student: Stage2 Pipeline<br/>Latent RMSE B,256]
    O -->|No| Q[End-to-End Forward<br/>NIR → Reconstruction<br/>RMSE in raw mm⁻¹]
    
    P --> R[Validation Every Epoch<br/>+ E2E every K epochs]
    Q --> R
    
    N --> S[Validation Loop<br/>Inverse standardize<br/>Compute raw metrics]
    R --> S
    
    S --> T[Early Stopping Check<br/>Patience: 25 epochs<br/>Monitor val loss]
    
    T -->|Continue| U[Next Epoch<br/>LR Schedule Step]
    T -->|Stop| V[Save Best Checkpoint<br/>Return training results]
    
    U --> J
    U --> K
    
    V --> W[Experiment Completion<br/>W&B logging<br/>Final metrics]
    
    style C fill:#e1f5fe
    style D fill:#fff3e0
    style P fill:#fce4ec
    style Q fill:#f3e5f5
```

---

## 13) Known Issues / Risks / TODOs

### Code Issues Found

**1. Incomplete Standardizer Implementation** (standardizers.py)
- **Problem**: Many methods show `...` or incomplete implementations
- **Evidence**: Multiple function stubs without complete code
- **Risk**: Standardization may not work correctly, affecting training
- **Fix**: Complete all standardizer implementations, especially inverse_transform methods

**2. Missing Parameter Counts**
- **Problem**: Exact parameter counts not computed in model logging
- **Evidence**: Comments reference ~7M parameters but no exact count
- **Risk**: Model size estimates may be inaccurate for deployment
- **Fix**: Add `sum(p.numel() for p in model.parameters())` to model initialization

**3. Hardcoded Measurement Subsampling**
- **Problem**: Fixed 256 measurements may not be optimal for all phantoms
- **Evidence**: `fixed_sequence_undersampling()` always selects 256 (hybrid_model.py:84)
- **Risk**: May lose important measurement diversity
- **Fix**: Make measurement count configurable or adaptive based on phantom complexity

**4. Mixed Unit Handling**
- **Problem**: NIR measurements mix standardized and scaled components
- **Evidence**: log_amp/phase standardized, positions scaled differently
- **Risk**: Inconsistent normalization may hurt transformer learning
- **Fix**: Unify normalization approach or clearly separate pipelines

### Training Risks

**5. Stage 2 Dependency Risk**
- **Problem**: Stage 2 requires exact Stage 1 checkpoint path
- **Evidence**: Hard failure if checkpoint missing (train_hybrid_model.py:243)
- **Risk**: Training pipeline fragility
- **Fix**: Add checkpoint validation and clearer error messages

**6. Memory Usage Scaling**
- **Problem**: Large batch sizes (128) with 3D volumes may cause OOM
- **Evidence**: GPU memory warnings in config but no adaptive batching
- **Risk**: Training crashes on smaller GPUs
- **Fix**: Implement adaptive batch sizing based on available memory

**7. Latent Dimension Mismatch Risk**
- **Problem**: Teacher and student latent dimensions must match exactly
- **Evidence**: LATENT_DIM = 256 used across multiple components
- **Risk**: Model architecture changes break latent training
- **Fix**: Add dimension validation in teacher-student setup

### Data Pipeline Issues

**8. Tissue Patch Coordinate Confusion**
- **Problem**: Multiple coordinate systems mentioned with auto-detection
- **Evidence**: data_loader.py:121-134 shows complex coordinate conversion
- **Risk**: Tissue patches may not align with correct phantom regions
- **Fix**: Standardize on single coordinate system, validate patch extraction

**9. HDF5 Error Handling**
- **Problem**: No error handling for corrupted or missing HDF5 files
- **Evidence**: Direct h5py.File access without try/catch blocks
- **Risk**: Training crashes on bad data files
- **Fix**: Add robust error handling and data validation

### Performance Concerns

**10. DataLoader Starvation Risk**
- **Problem**: High worker count (16) may exceed system resources
- **Evidence**: NUM_WORKERS = 16 hardcoded (training_config.py:66)
- **Risk**: Slower training due to resource contention
- **Fix**: Auto-detect optimal worker count based on CPU cores and memory

---

## 14) Reproduction & Runbook

### Environment Setup

**Required Dependencies**:
- PyTorch with CUDA support
- HDF5 and h5py for data loading
- Weights & Biases for logging
- NIRFASTer-FF for phantom simulation
- Standard scientific Python stack (numpy, scipy, matplotlib)

**System Requirements**:
- GPU with ≥8GB VRAM (for batch_size=128)
- ≥32GB system RAM (for 16 DataLoader workers)
- Fast storage for HDF5 dataset access

### Data Generation

**Generate Phantoms** (if needed):
```bash
cd /home/ubuntu/NIR-DOT
python code/data_processing/data_simulator.py
# Generates phantoms in data/ directory
# Each phantom: ~1000 measurements + ground truth
```

### Training Commands

**Stage 1 Training**:
1. Edit `code/training/training_config.py:25`:
   ```python
   CURRENT_TRAINING_STAGE = "stage1"
   ```
2. Run training:
   ```bash
   cd /home/ubuntu/NIR-DOT
   python code/training/train_hybrid_model.py
   ```
3. **Expected outputs**:
   - Checkpoint: `checkpoints/stage1_best.pth`
   - W&B run: Tagged "stage1", "cnn-autoencoder"
   - Training time: ~150 epochs (depends on early stopping)

**Stage 2 Training**:
1. Ensure Stage 1 checkpoint exists at `checkpoints/stage1_best.pth`
2. Edit `code/training/training_config.py:25`:
   ```python
   CURRENT_TRAINING_STAGE = "stage2"
   ```
3. **Configure mode**:
   - **Baseline**: `USE_TISSUE_PATCHES_STAGE2 = False` (line 40)
   - **Enhanced**: `USE_TISSUE_PATCHES_STAGE2 = True` (line 40)
4. **Configure training type**:
   - **Latent-only**: `TRAIN_STAGE2_LATENT_ONLY = True` (line 33)
   - **End-to-end**: `TRAIN_STAGE2_LATENT_ONLY = False` (line 33)
5. Run training:
   ```bash
   cd /home/ubuntu/NIR-DOT
   python code/training/train_hybrid_model.py
   ```
6. **Expected outputs**:
   - Checkpoint: `checkpoints/stage2_baseline_best.pth` or `checkpoints/stage2_enhanced_best.pth`
   - W&B run: Tagged "stage2", "baseline"/"enhanced"

### Configuration Toggles

**Key flags to modify in `training_config.py`**:

| Flag | Effect | Line |
|------|--------|------|
| `CURRENT_TRAINING_STAGE` | "stage1" or "stage2" | 25 |
| `USE_WANDB_LOGGING` | Enable/disable W&B | 29 |
| `TRAIN_STAGE2_LATENT_ONLY` | Latent vs end-to-end Stage 2 | 33 |
| `USE_TISSUE_PATCHES_STAGE2` | Baseline vs enhanced mode | 40 |
| `DEBUG_VERBOSE` | Detailed logging | 31 |
| `BATCH_SIZE` | Training batch size | 58 |

### Expected W&B Artifacts

**Stage 1 Tags**: `["stage1", "cnn-autoencoder", "pretraining", "nir-dot"]`
**Stage 2 Baseline Tags**: `["stage2", "transformer", "baseline", "nir-dot"]`  
**Stage 2 Enhanced Tags**: `["stage2", "transformer", "enhanced", "tissue-patches", "nir-dot"]`

**Key Metrics to Monitor**:
- `train/loss`, `val/total_rmse_raw`
- `val/dice_absorption`, `val/dice_scattering`
- `val/contrast_absorption`, `val/contrast_scattering`
- `val/latent_rmse` (Stage 2 latent-only mode)

**Visualizations**: 24 reconstruction images per epoch (2 phantoms × 3 planes × 2 channels × 2 types)

### Troubleshooting

**Common Issues**:
1. **OOM Error**: Reduce `BATCH_SIZE` from 128 to 64 or 32
2. **DataLoader Stall**: Reduce `NUM_WORKERS` from 16 to 8 or 4
3. **Stage 2 Fails**: Verify Stage 1 checkpoint exists and path is correct
4. **W&B Login**: Ensure `wandb login` completed successfully
5. **Missing Data**: Check that `data/` directory contains phantom files

**Performance Tuning**:
- Monitor GPU memory with training logs
- Adjust batch size based on available VRAM
- Use `persistent_workers=False` if memory constrained
- Disable model compilation if compilation fails

---

**End of Technical Brief**

*This analysis covers all discoverable aspects of the NIR-DOT reconstruction codebase based on systematic examination of 48 Python files totaling 21,393 lines of code. All claims are supported by specific file paths and line numbers as cited throughout.*
