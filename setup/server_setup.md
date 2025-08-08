# 🚀 NIR-DOT Server Setup Guide

> **Fast LambdaLabs Setup - From Zero to Training in 3 Minutes!**

## 📦 Required Files to Transfer

Transfer these folders to `/home/ubuntu/NIR-DOT/`:

```
📁 code/              # Your source code
📁 nirfaster-FF/      # NIRFASTer finite element solver  
📁 setup/             # This setup folder
```

---

## ⚡ Quick Setup (New Instance)

### 1. 🏗️ Fast Bootstrap
```bash
chmod +x setup/bootstrap_lambdalabs.sh
./setup/bootstrap_lambdalabs.sh
```
⏱️ **Takes 2-3 minutes vs 20+ minutes manual setup!**

### 2. 📚 Install Dependencies
```bash
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate
pip install -r setup/requirements.txt
```

### 3. 🎯 Ready to Train!
```bash
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate
```

---

## 📺 Tmux Setup (Persistent Sessions)

### Why Use Tmux?
- ✅ **Survives SSH disconnections** - your job keeps running
- ✅ **Reconnect anytime** to check progress  
- ✅ **No interruptions** from network issues

### Basic Tmux Commands
```bash
# Create new persistent session
tmux new -s phantom_generation

# Inside tmux - run your long job
python -m code.data_processing.data_simulator --count 5000

# DETACH (leave it running): Press Ctrl+B, then D
# You can now safely disconnect SSH

# Later - reconnect to check progress
ssh ubuntu@your-server
tmux attach -t phantom_generation  # Back to your running job!

# List all sessions
tmux list-sessions

# Kill session when done
tmux kill-session -t phantom_generation
```

### Quick Reference
| Action | Command |
|--------|---------|
| **Create session** | `tmux new -s name` |
| **Detach (keep running)** | `Ctrl+B, then D` |
| **Reconnect** | `tmux attach -t name` |
| **List sessions** | `tmux list-sessions` |
| **Kill session** | `tmux kill-session -t name` |

---

## 🧬 Phantom Generation

### Generate Training Phantoms (Long-Running Job)
```bash
# For long phantom generation runs, use tmux for persistence
tmux new -s phantom_run

# Activate environment inside tmux
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate

# Generate phantoms for training (adjust count as needed)
python -m code.data_processing.data_simulator --count 5000 --output_dir data/phantoms/

# DETACH to leave running: Ctrl+B, then D
# Your phantoms will generate even if SSH disconnects!
```

### Quick Phantom Generation (No Tmux Needed)
```bash
# For short runs (< 1 hour)
python -m code.data_processing.data_simulator --count 100 --output_dir data/phantoms/
```

### Check Running Jobs
```bash
# Reconnect to phantom generation
tmux attach -t phantom_run

# Or check if running without attaching
ps aux | grep data_simulator
```

### NIRFASTer Backend Options
```bash
# Check available computational backends
python -c "from code.data_processing.data_simulator import check_nirfaster_backends; check_nirfaster_backends()"
```

---

## 🏋️ Training Options

### Stage 1 Training (Baseline CNN-Autoencoder)

#### 🔵 Baseline Mode (No Tissue Patches)
```bash
# Edit training_config.py first:
# USE_TISSUE_PATCHES_STAGE1 = False

python -m code.training.stage1_trainer
```

#### 🟢 Enhanced Mode (With Tissue Patches)
```bash
# Edit training_config.py first:
# USE_TISSUE_PATCHES_STAGE1 = True

python -m code.training.stage1_trainer
```

### Stage 2 Training (CNN-Transformer Hybrid)

#### 🔵 Baseline Hybrid
```bash
# Edit training_config.py first:
# USE_TISSUE_PATCHES_STAGE2 = False

python -m code.training.stage2_trainer
```

#### 🟢 Enhanced Hybrid (With Tissue Patches)
```bash
# Edit training_config.py first:
# USE_TISSUE_PATCHES_STAGE2 = True

python -m code.training.stage2_trainer
```

### Full Pipeline Training
```bash
# Train both stages sequentially
python -m code.training.train_hybrid_model
```

---

## ⚙️ Configuration Controls

### 📝 Edit Training Settings
```bash
# Open the config file
nano code/training/training_config.py
```

### 🎛️ Key Configuration Options

```python
# Tissue Patches Enhancement
USE_TISSUE_PATCHES_STAGE1 = True   # Enable for Stage 1
USE_TISSUE_PATCHES_STAGE2 = True   # Enable for Stage 2

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS_STAGE1 = 100
NUM_EPOCHS_STAGE2 = 50

# Data Settings
PHANTOM_COUNT = 1000
MEASUREMENT_SUBSAMPLE = 256
```

### 🔄 Quick Config Switching

**Switch to Baseline (No Enhancements):**
```bash
sed -i 's/USE_TISSUE_PATCHES_STAGE1 = True/USE_TISSUE_PATCHES_STAGE1 = False/' code/training/training_config.py
sed -i 's/USE_TISSUE_PATCHES_STAGE2 = True/USE_TISSUE_PATCHES_STAGE2 = False/' code/training/training_config.py
```

**Switch to Enhanced (With Tissue Patches):**
```bash
sed -i 's/USE_TISSUE_PATCHES_STAGE1 = False/USE_TISSUE_PATCHES_STAGE1 = True/' code/training/training_config.py
sed -i 's/USE_TISSUE_PATCHES_STAGE2 = False/USE_TISSUE_PATCHES_STAGE2 = True/' code/training/training_config.py
```

---

## 📊 Monitoring & Analysis

### 🔍 Check Training Progress
```bash
# Monitor with Weights & Biases
wandb login
# Training logs automatically sync to W&B dashboard
```

### 📈 View Results
```bash
# Generate analysis plots
python -m code.data_processing.data_analysis

# Check model performance
ls analysis_results/
```

---

## 🔄 Daily Usage (Persistent Environment)

After initial setup, just activate and train:

```bash
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate

# For long training runs, use tmux:
tmux new -s training_run

# Choose your training mode:
python -m code.training.stage1_trainer    # Stage 1 only
python -m code.training.stage2_trainer    # Stage 2 only  
python -m code.training.train_hybrid_model # Full pipeline

# Detach: Ctrl+B, then D
# Reconnect later: tmux attach -t training_run
```

---

## 🛠️ Troubleshooting

### Environment Issues
```bash
# Recreate virtual environment if needed
rm -rf /home/ubuntu/NIR-DOT/venv_diss
./setup/bootstrap_lambdalabs.sh
```

### NIRFASTer Problems
```bash
# Check NIRFASTer installation
python -c "import nirfasterff; print('✅ NIRFASTer OK')"

# Check hardware detection
python -c "from code.data_processing.data_simulator import check_hardware; check_hardware()"
```

### Memory Issues
```bash
# Monitor GPU/CPU usage
nvidia-smi
htop
```

### Tmux Issues
```bash
# List all tmux sessions
tmux list-sessions

# Force kill stuck session
tmux kill-session -t session_name

# Kill all tmux sessions
tmux kill-server
```

---

## 💡 Pro Tips

- 🔒 **Virtual environment survives instance restarts** (stored on persistent disk)
- 🚀 **Always activate environment first** before running any commands
- 📊 **Use W&B dashboard** to compare baseline vs enhanced models
- 🎯 **Start with small phantom counts** (100-200) for testing
- 🔄 **Toggle tissue patches** via config file - no code changes needed!
- 📺 **Use tmux for ALL long-running jobs** (>30 minutes)
- 🔌 **Tmux sessions survive SSH disconnections** - perfect for overnight runs
- 💰 **Remember to terminate Lambda Labs instance** when completely done!

---

## 📚 Quick Reference

| Mode | Stage 1 | Stage 2 | Description |
|------|---------|---------|-------------|
| **Baseline** | `False` | `False` | Standard CNN-Autoencoder + Transformer |
| **Enhanced** | `True` | `True` | With tissue patches enhancement |
| **Hybrid** | `False` | `True` | Baseline Stage 1, Enhanced Stage 2 |

**Happy Training! 🎉**
