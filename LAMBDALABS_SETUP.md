# 🚀 LambdaLabs A100 Setup Guide

Simple step-by-step guide to get your NIR-DOT training running on LambdaLabs A100 GPU.

## 📋 What You'll Need

- LambdaLabs A100 instance (40GB)
- FileZilla (for file transfer)
- Your SSH key for the instance

## 🎯 Simple 4-Step Process

### Step 1: Launch & Bootstrap Server
1. **Launch A100 instance** on LambdaLabs with persistent storage (500GB+)
2. **SSH into server:**
   ```bash
   ssh -i ~/.ssh/your-key.pem ubuntu@your-instance-ip
   ```
3. **Run bootstrap script:**
   ```bash
   curl -o bootstrap_lambdalabs.sh https://raw.githubusercontent.com/maxhartml/mah422/main/bootstrap_lambdalabs.sh
   chmod +x bootstrap_lambdalabs.sh
   ./bootstrap_lambdalabs.sh
   ```

### Step 2: Transfer Files with FileZilla
1. **Open FileZilla** and connect:
   - Host: `your-instance-ip`
   - Username: `ubuntu`
   - Port: `22`
   - Use your SSH key for authentication

2. **Transfer these folders/files:**
   ```
   Local Side (drag these)          →    Server Side (drop here)
   ─────────────────────────────────     ──────────────────────
   📁 mah422/                       →    /home/ubuntu/mah422/
   📁 nirfaster-FF/ (if you have it) →    /home/ubuntu/nirfaster-FF/
   📄 requirements.txt              →    /home/ubuntu/requirements.txt
   ```

   **DON'T transfer:**
   - ❌ `env_diss/` (virtual environment)
   - ❌ `__pycache__/` folders
   - ❌ `.git/` folder (optional - it's large)
   - ❌ Large data files (generate fresh on server)

### Step 3: Complete Setup
1. **SSH back into server:**
   ```bash
   ssh -i ~/.ssh/your-key.pem ubuntu@your-instance-ip
   ```

2. **Activate environment & install dependencies:**
   ```bash
   source ~/env_diss/bin/activate
   cd mah422
   pip install -r requirements.txt
   python setup_lambdalabs.py
   ```

### Step 4: Start Training!
```bash
# Generate phantoms (parallel - 8x faster!)
python -m code.data_processing.parallel_data_simulator

# Train Stage 1 (CNN)
python -m code.training.train_hybrid_model

# Train Stage 2 (Transformer)
# Edit training_config.py: CURRENT_TRAINING_STAGE = "stage2"
python -m code.training.train_hybrid_model
```

## 🔧 Useful Commands

### Monitor GPU Usage:
```bash
watch -n 1 nvidia-smi
```

### Keep Training Running (even if SSH disconnects):
```bash
# Start tmux session
tmux new-session -d -s training

# Run your training inside tmux
python -m code.training.train_hybrid_model

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### Transfer Results Back:
Use FileZilla to drag these folders back to your local machine:
- `checkpoints/` (trained models)
- `results/` (training outputs)
- `logs/` (training logs)

## 💰 Expected Costs

- **Phantom generation (5000):** ~3 hours × £1 = £3
- **Stage 1 training:** ~4 hours × £1 = £4
- **Stage 2 training:** ~8 hours × £1 = £8
- **Total:** ~£15 for complete training pipeline

## 🆘 Troubleshooting

### If bootstrap fails:
```bash
# Check Python version
python3.12 --version

# Manual virtual environment creation
python3.12 -m venv ~/env_diss
source ~/env_diss/bin/activate
```

### If requirements.txt fails:
```bash
# Install critical packages manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy scipy matplotlib psutil wandb
```

### If GPU not detected:
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 🎯 Performance Gains

| Component | MacBook CPU | A100 GPU | Speedup |
|-----------|-------------|----------|---------|
| Phantom Generation | 10s each | ~1.25s each | 8x |
| Stage 1 Training | Very slow | Fast | 20-50x |
| Stage 2 Training | Very slow | Fast | 30-80x |
| **Total Pipeline** | **Weeks** | **1-2 days** | **~10x** |

---

**🎉 That's it! You're ready for blazing fast NIR-DOT training on the A100!**
