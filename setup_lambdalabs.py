#!/usr/bin/env python3
"""
🚀 LambdaLabs A100 Setup Script 🚀

Quick setup script for LambdaLabs A100 instances.
Installs dependencies and tests GPU functionality.

Usage:
    python setup_lambdalabs.py

Author: Max Hart
Date: August 2025
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and print status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_pytorch():
    """Install PyTorch with CUDA support."""
    print("🔥 Installing PyTorch with CUDA 11.8 support...")
    return run_command(
        f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "PyTorch CUDA installation"
    )

def install_from_requirements():
    """Install packages from requirements.txt."""
    req_file = Path("requirements.txt")
    
    if req_file.exists():
        print(f"📦 Installing from requirements.txt...")
        return run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing from requirements.txt"
        )
    else:
        print(f"❌ requirements.txt not found!")
        print(f"💡 Make sure you transferred your project files first")
        return False

def install_dependencies():
    """Install dependencies from requirements.txt (simplified)."""
    return install_from_requirements()

def test_gpu():
    """Test GPU availability and memory."""
    print("🧪 Testing GPU functionality...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_count = torch.cuda.device_count()
            
            print(f"🚀 GPU detected: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            print(f"🔢 GPU Count: {gpu_count}")
            
            # Test tensor operations
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print(f"✅ GPU tensor operations working!")
            
            # Test memory allocation
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 Current GPU memory usage: {allocated:.2f}GB")
            
            return True
        else:
            print("❌ No GPU detected - check CUDA installation")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed properly")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def check_cpu_info():
    """Check CPU information."""
    print("💻 Checking CPU information...")
    
    try:
        import psutil
        
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        
        print(f"🖥️  Physical CPU cores: {cpu_count}")
        print(f"🧮 Logical CPU cores: {cpu_count_logical}")
        print(f"💾 Total RAM: {memory.total / 1024**3:.1f}GB")
        print(f"💾 Available RAM: {memory.available / 1024**3:.1f}GB")
        
        if cpu_count >= 8:
            print(f"🚀 CPU cores sufficient for parallel phantom generation!")
        else:
            print(f"⚠️ Limited CPU cores - parallel benefits may be reduced")
            
        return True
        
    except ImportError:
        print("❌ psutil not available")
        return False

def print_optimization_summary():
    """Print optimization summary."""
    print("\n" + "="*60)
    print("🎯 A100 OPTIMIZATION SUMMARY")
    print("="*60)
    
    try:
        import torch
        import psutil
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory > 30:  # A100 or similar
                batch_stage1 = 32
                batch_stage2 = 16
            elif gpu_memory > 15:
                batch_stage1 = 16
                batch_stage2 = 8
            else:
                batch_stage1 = 8
                batch_stage2 = 4
            
            print(f"📊 Optimal batch sizes:")
            print(f"   • Stage 1 (CNN): {batch_stage1}")
            print(f"   • Stage 2 (Transformer): {batch_stage2}")
        
        cpu_cores = psutil.cpu_count(logical=False)
        workers = max(1, cpu_cores - 2)
        
        print(f"🔄 Parallel processing:")
        print(f"   • DataLoader workers: {workers}")
        print(f"   • Phantom generation workers: {workers}")
        
        print(f"\n💡 Usage examples:")
        print(f"   • Parallel phantom generation:")
        print(f"     python -m code.data_processing.parallel_data_simulator")
        print(f"   • Training:")
        print(f"     python -m code.training.train_hybrid_model")
        
    except:
        print("❌ Could not generate optimization summary")

def main():
    """Main setup function."""
    print("🚀 LambdaLabs A100 Setup Script")
    print("="*50)
    
    success = True
    
    # Skip PyTorch - already installed in bootstrap
    print("⏭️  Skipping PyTorch installation (done in bootstrap)")
    
    # Install from requirements.txt
    if not install_dependencies():
        success = False
    
    # Test GPU
    if not test_gpu():
        success = False
    
    # Check CPU info
    if not check_cpu_info():
        success = False
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print_optimization_summary()
    else:
        print("\n❌ Setup completed with errors - check output above")
        sys.exit(1)

if __name__ == "__main__":
    main()
