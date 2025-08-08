#!/bin/bash
# 🚀 Fast LambdaLabs Bootstrap Script
# Minimal setup - skip slow system updates!

echo "🚀 Fast LambdaLabs Bootstrap Starting..."
echo "============================================"

# Skip system updates to save time (CUDA/drivers already installed)
echo "⏭️  Skipping system updates (saves 10+ minutes)"

# Check what Python versions are available
echo "🐍 Checking available Python versions..."
python3 --version
python3.10 --version 2>/dev/null || echo "Python 3.10 not found"
python3.11 --version 2>/dev/null || echo "Python 3.11 not found"

# Use system Python 3.10 (already installed on LambdaLabs)
PYTHON_CMD="python3"
echo "✅ Using system Python: $(python3 --version)"

# Install only essential missing tools (most already installed)
echo "🔧 Installing minimal essential tools..."
sudo apt update -qq  # Quiet update, just packages list
sudo apt install -y python3-venv python3-pip tmux

# Check tmux installation
echo "📺 Checking tmux installation..."
tmux -V && echo "✅ tmux ready for persistent sessions" || echo "⚠️  tmux installation issue"

# Create virtual environment on PERSISTENT storage
echo "🐍 Creating virtual environment on persistent storage..."
$PYTHON_CMD -m venv /home/ubuntu/NIR-DOT/venv_diss
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

echo ""
echo "✅ Fast Bootstrap complete! (2-3 minutes vs 20+ minutes)"
echo "========================================================="
echo "💡 Next steps:"
echo "   1. Transfer your project files using FileZilla"
echo "   2. SSH back in and run: source /home/ubuntu/NIR-DOT/venv_diss/bin/activate"
echo "   3. Run: cd mah422 && pip install -r setup/requirements.txt"
echo "   4. For long runs: tmux new -s phantom_run (persistent sessions)"
echo "   5. Start training!"
echo ""
echo "📺 Tmux Quick Commands:"
echo "   tmux new -s session_name     # Create new session"
echo "   Ctrl+B, then D               # Detach (leave running)"
echo "   tmux attach -t session_name  # Reconnect later"
echo ""
echo "🔧 To run this script:"
echo "   chmod +x setup/bootstrap_lambdalabs.sh"
echo "   ./setup/bootstrap_lambdalabs.sh"
