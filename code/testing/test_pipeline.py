#!/usr/bin/env python3
"""
Simple test script for Robin Dale's pipeline - just the essentials!
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.data_processing.data_loader import create_nir_dataloaders, NIRPhantomDataset
from code.models.hybrid_model import HybridCNNTransformer
from code.training.training_utils import RMSELoss, compute_rmse, compute_mae
from code.training.stage1_trainer import Stage1Trainer
from code.training.stage2_trainer import Stage2Trainer

def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Testing Robin Dale's Pipeline - Simple Version")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Data loading
    print("\n1. Testing data loading...")
    total_tests += 1
    try:
        data_loaders = create_nir_dataloaders(
            data_dir="data", 
            batch_size=2, 
            use_tissue_patches=False
        )
        print("✅ Data loading works")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
    
    # Test 2: Model creation
    print("\n2. Testing model creation...")
    total_tests += 1
    try:
        model = HybridCNNTransformer(use_tissue_patches=False)
        print("✅ Model creation works")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
    
    # Test 3: Loss function
    print("\n3. Testing loss function...")
    total_tests += 1
    try:
        loss_fn = RMSELoss()
        pred = torch.randn(1, 1, 64, 64, 64)
        target = torch.randn(1, 1, 64, 64, 64)
        loss = loss_fn(pred, target)
        print(f"✅ RMSE loss works: {loss.item():.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Loss function failed: {e}")
    
    # Test 4: Toggle functionality
    print("\n4. Testing toggle functionality...")
    total_tests += 1
    try:
        # Test baseline (no tissue patches) - proper 5D input shape
        model_baseline = HybridCNNTransformer(use_tissue_patches=False)
        measurements = torch.randn(1, 1, 64, 64, 64)  # 5D: [batch, channel, H, W, D]
        output_baseline = model_baseline(measurements, tissue_patches=None)
        
        # Test enhanced (with tissue patches)
        model_enhanced = HybridCNNTransformer(use_tissue_patches=True)
        tissue_patches = torch.randn(1, 2, 343)  # [batch, num_patches, patch_size^3] where 7^3=343
        output_enhanced = model_enhanced(measurements, tissue_patches=tissue_patches)
        
        print("✅ Toggle functionality works")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Toggle functionality failed: {e}")
    
    # Test 5: Stage1 Trainer initialization
    print("\n5. Testing Stage1 Trainer...")
    total_tests += 1
    try:
        trainer = Stage1Trainer(learning_rate=1e-4, device="cpu")
        print("✅ Stage1 Trainer works")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Stage1 Trainer failed: {e}")
    
    # Summary
    print(f"\n🎯 Tests Summary: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Robin Dale's pipeline is ready!")
        return True
    else:
        print("❌ Some tests failed - check errors above")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)