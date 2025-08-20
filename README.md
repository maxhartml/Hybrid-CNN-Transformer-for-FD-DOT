# 🧬 NIR-DOT Reconstruction: Two-Stage CNN-Transformer with Latent Alignment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *Towards Generalisable Inverse Modelling in Near-Infrared Diffuse Optical Tomography*
> 
> **MSc AI/ML Dissertation Project** • University of Birmingham • 2025

## 🎯 Project Overview

This project tackles the **severely ill-posed inverse problem** in Near-Infrared Diffuse Optical Tomography (NIR-DOT): reconstructing 3D tissue optical properties from sparse boundary measurements. Our solution combines the spatial expertise of CNNs with the sequence modeling power of transformers through a novel **teacher-student latent alignment** approach.

### 🚀 Key Innovation
- **Two-stage hybrid architecture**: CNN spatial priors + Transformer sequence modeling
- **Latent alignment strategy**: Student transformer learns to match teacher CNN representations
- **Preserved decoder quality**: Frozen CNN decoder maintains spatial reconstruction fidelity
- **Clinical realism**: Physics-based phantom generation with realistic probe geometries

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 1: CNN AUTOENCODER                   │
│                                                                 │
│  Ground Truth → 3D CNN Encoder → [256D Latent] → 3D CNN        │
│  Volumes                             │           Decoder       │
│                                      ▼              │          │
│                                Teacher Latent       ▼          │
│                                                Reconstructed    │
│                                                Ground Truth     │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ (Knowledge Transfer)
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 2: TRANSFORMER ENHANCEMENT                │
│                                                                 │
│  NIR Measurements → Spatially-Aware → Transformer → Global     │
│                     Embedding         Encoder      Pooling     │
│                                                        │        │
│                                                        ▼        │
│                                                [256D Student    │
│                                                 Latent] ────────┤
│                                                        │        │
│                         LATENT ALIGNMENT LOSS ←───────┘        │
│                         (Teacher ↔ Student)                    │
│                                                                 │
│                                                        ▼        │
│                                              Frozen CNN Decoder │
│                                                        │        │
│                                                        ▼        │
│                                              Enhanced Reconstruction
└─────────────────────────────────────────────────────────────────┘

Key: → Forward pass  ─── Latent alignment  ▼ Flow direction
```

### 🔧 Component Breakdown

| Component | Purpose | Architecture |
|-----------|---------|--------------|
| **CNN Autoencoder** | Spatial feature learning | 16→32→64→128→256 channels, ResNet blocks |
| **Spatially-Aware Embedding** | Measurement tokenization | Separate MLPs for measurements + positions |
| **Transformer Encoder** | Sequence modeling | 8 layers × 8 heads, d_model=256 |
| **Global Pooling** | Sequence aggregation | Multi-query attention (4 queries) |
| **Latent Alignment** | Knowledge transfer | RMSE loss between teacher-student |

---

## 📊 Data Pipeline

### Phantom Generation
- **🎲 Synthetic phantoms**: 64³ voxel volumes (1mm resolution)
- **🔬 Realistic tissue properties**: μₐ ∈ [0.003, 0.007] mm⁻¹, μ'ₛ ∈ [0.78, 1.18] mm⁻¹
- **🩻 Tumor inclusions**: 0-5 tumors per phantom, 5-15mm radius
- **🌀 Random orientations**: Eliminate directional bias through 3D rotations

### Forward Modeling
- **⚡ NIRFASTer-FF**: Frequency-domain FEM solver (140MHz)
- **📡 Realistic probe geometry**: 50 sources × 20 detectors = 1000 measurements
- **🎯 Clinical SDS range**: 10-40mm source-detector separations
- **🔊 Realistic noise**: 0.5% amplitude + ±0.5° phase noise

### Data Augmentation
- **🎰 Measurement subsampling**: 256 from 1000 measurements per training iteration
- **📈 Dataset efficiency**: 3.9× more training combinations per phantom
- **🔀 Randomized selection**: Different measurement subsets each epoch

---

## 🚀 Training Strategy

### Stage 1: CNN Foundation (200 epochs)
```python
# Spatial feature learning with aggressive optimization
optimizer = AdamW(lr=1e-4, betas=[0.9, 0.98], weight_decay=1e-3)
scheduler = OneCycleLR(max_lr=2e-3, pct_start=0.4, div_factor=20)
loss = RMSE(predictions, standardized_ground_truth)
```

**🎯 Goal**: Learn robust spatial representations and basic reconstruction capabilities

### Stage 2: Transformer Enhancement (400 epochs)
```python
# Teacher-student latent alignment
teacher_latent = frozen_cnn_encoder(ground_truth)  # 256D
student_latent = transformer_pipeline(nir_measurements)  # 256D
loss = RMSE(student_latent, teacher_latent)  # Latent-only training
```

**🎯 Goal**: Enhance CNN features with transformer sequence modeling while preserving decoder quality

---

## 📁 Project Structure

```
mah422/
├── 📂 code/                    # Core implementation
│   ├── 🔬 data_processing/     # Phantom generation & data loading
│   ├── 🤖 models/              # CNN, Transformer, Hybrid architectures
│   ├── 🏋️ training/            # Stage 1 & 2 trainers + configs
│   └── 🛠️ utils/               # Metrics, visualization, logging
├── 📊 data/                   # Generated phantom datasets (HDF5)
├── 💾 checkpoints/            # Saved model states
├── 📈 logs/                   # Training logs & metrics
├── 🎨 analysis_results/       # Visualization outputs
└── 🔧 nirfaster-FF/           # Forward modeling library
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Create Python virtual environment
python3 -m venv nir-dot-env
source nir-dot-env/bin/activate  # On Windows: nir-dot-env\Scripts\activate

# Install dependencies from requirements
pip install -r requirements.txt
```

### 2. Generate Data
```bash
# Create synthetic phantom dataset
python -m code.data_processing.data_simulator
# Generates 5000 phantoms with realistic tissue properties
```

### 3. Train Models
```bash
# Stage 1: CNN Autoencoder pre-training
python -m code.training.train_hybrid_model  # CURRENT_TRAINING_STAGE = "stage1"

# Stage 2: Transformer enhancement
python -m code.training.train_hybrid_model  # CURRENT_TRAINING_STAGE = "stage2"
```

### 4. Analyze Results
```bash
# Dataset analysis
python -m code.data_processing.phantom_dataset_analysis

# Training visualization (Weights & Biases integration)
wandb login  # View training metrics at wandb.ai
```

---

## 🎛️ Key Hyperparameters

| Stage | Optimizer | LR Schedule | Batch Size | Loss | Special Features |
|-------|-----------|-------------|------------|------|------------------|
| **1** | AdamW (1e-4) | OneCycleLR (max 2e-3) | 128 | RMSE | Momentum cycling, AMP |
| **2** | AdamW (3e-4) | Linear + Cosine | 128 | Latent RMSE | EMA, Frozen decoder |

---

## 📊 Results Highlights

### Reconstruction Quality
- **✅ Stage 1**: Robust spatial features, sharp μₐ recovery
- **🚀 Stage 2**: Enhanced global coherence, reduced artifacts
- **⚡ Performance**: 2× speedup with mixed precision training
- **🎯 Stability**: EMA provides improved generalization

### Technical Achievements
- **🧠 Parameter efficiency**: 7M CNN parameters (optimized from 26.9M)
- **💾 Memory optimization**: 97% reduction in Stage 1 data loading
- **⚙️ Infrastructure**: PyTorch 2.0 compilation, A100 optimization
- **🔬 Physics compliance**: Realistic forward modeling with NIRFASTer-FF

---

## 🛠️ Technical Features

### 🎯 Architecture Innovations
- **Spatially-aware embedding**: Separate processing of measurements and coordinates
- **Multi-query attention pooling**: Efficient sequence aggregation
- **Frozen decoder preservation**: Maintains CNN spatial priors
- **Progressive EMA decay**: Stabilizes late-stage training

### ⚡ Engineering Excellence
- **Mixed precision training**: A100-optimized with conservative scaling
- **Deterministic evaluation**: Fixed seeds for reproducible results
- **Comprehensive logging**: W&B integration with detailed metrics
- **Memory-efficient data loading**: Stage-specific optimizations

### 🔬 Physics-Based Validation
- **Realistic tissue properties**: Clinically relevant optical coefficients  
- **Proper noise modeling**: SNR ~46dB matching clinical systems
- **SDS constraints**: 10-40mm range ensures diffusive regime
- **Surface-aware probe placement**: Morphologically extracted tissue boundaries

---

## 🎓 Research Context

This work addresses fundamental challenges in medical imaging inverse problems:

1. **🎯 Ill-posed inverse problem**: Sparse measurements → dense 3D reconstructions
2. **🏥 Clinical translation gap**: Synthetic training → real measurement generalization  
3. **⚖️ Physics vs. learning trade-off**: Preserve domain knowledge while enabling ML enhancement
4. **🔧 Computational efficiency**: Real-time reconstruction for clinical workflows

Our **two-stage latent alignment** approach provides a principled framework for combining physics-based priors with data-driven enhancement.

---

## 🚧 Current Limitations & Future Work

### Limitations
- **📊 Synthetic data only**: Requires domain adaptation for real clinical measurements
- **🔍 μ'ₛ recovery challenges**: Physics-limited scattering coefficient reconstruction
- **💾 Memory requirements**: 64³ volumes demand significant GPU memory
- **⏱️ Attention scaling**: Quadratic complexity limits sequence length

### Future Directions
- **🏥 Real data validation**: Clinical measurement integration with domain adaptation
- **🌈 Multi-wavelength extension**: Spectroscopic DOT capabilities
- **🔍 Dynamic sequence undersampling**: Adaptive measurement selection
- **🧬 Tissue-patch context**: Enhanced anatomical awareness (framework implemented)

---

## 📚 Key References & Dependencies

### Core Libraries
- **🔥 PyTorch 2.0+**: Deep learning framework with compilation
- **🔬 NIRFASTer-FF**: Frequency-domain FEM forward modeling
- **📊 Weights & Biases**: Experiment tracking and visualization
- **💾 HDF5**: Efficient dataset storage and loading

### Research Foundation
- Teacher-student knowledge distillation
- Transformer architectures for medical imaging
- Physics-informed neural networks
- Diffuse optical tomography reconstruction methods

---

## 📄 License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use this code in your research, please cite:
```bibtex
@mastersthesis{hart2025nirdot,
    title={Towards Generalisable Inverse Modelling in NIR-DOT: Two-Stage CNN-Transformer with Latent Alignment},
    author={Hart, Max},
    year={2025},
    school={University of Birmingham},
    type={MSc Dissertation}
}
```

---

<div align="center">

**🧠 Built with curiosity • 🔬 Powered by physics • 🚀 Enhanced by AI**

*University of Birmingham • MSc Artificial Intelligence & Machine Learning • 2025*

</div>
