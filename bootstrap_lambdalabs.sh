#!/bin/bash
# 🚀 LambdaLabs Bootstrap Script - Run FIRST on fresh Ubuntu server
# This prepares the server for your project files

echo "🚀 LambdaLabs Bootstrap Script Starting..."
echo "================================================"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 specifically (NIRFASTer requirement)
echo "🐍 Installing Python 3.12..."
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils

# Install essential tools
echo "🔧 Installing essential tools..."
sudo apt install -y \
    git \
    htop \
    tmux \
    nano \
    curl \
    wget \
    unzip \
    build-essential \
    pip

# Create virtual environment with Python 3.12
echo "🐍 Creating Python 3.12 virtual environment..."
python3.12 -m venv ~/env_diss
source ~/env_diss/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA first (most critical)
echo "🔥 Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "✅ Bootstrap complete!"
echo "================================================"
echo "💡 Next steps:"
echo "   1. Transfer your project files using FileZilla"
echo "   2. SSH back in and run: source ~/env_diss/bin/activate"
echo "   3. Run: cd mah422 && pip install -r requirements.txt"
echo "   4. Start training!"
echo ""
echo "🔧 To run this script:"
echo "   chmod +x bootstrap_lambdalabs.sh"
echo "   ./bootstrap_lambdalabs.sh"
