#!/usr/bin/env python3
"""
🚀 Simple GPU Test Script

Just tests if PyTorch and GPU are working properly.

Usage:
    python setup_lambdalabs.py

Author: Max Hart
Date: August 2025
"""

def test_gpu():
    """Test if GPU is working."""
    print("🧪 Testing GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f}GB")
            
            # Quick test
            x = torch.randn(100, 100).cuda()
            y = x @ x
            print(f"✅ GPU operations working!")
            
            return True
        else:
            print("❌ No GPU detected")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if test_gpu():
        print("\n� Everything looks good! Ready for training.")
    else:
        print("\n❌ Something's wrong - check your setup.")
