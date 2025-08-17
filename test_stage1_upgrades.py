#!/usr/bin/env python3
"""
Smoke test for Stage 1 training upgrades.
"""
import sys
import torch
sys.path.append('/Users/maxhart/Documents/MSc_AI_ML/Dissertation/mah422')

def test_imports():
    """Test that all new imports work."""
    try:
        from code.utils.tv3d import tv3d_l1
        from code.utils.ema import EMA
        print("✅ New utility imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_tv3d():
    """Test TV3D function."""
    try:
        from code.utils.tv3d import tv3d_l1
        # Test with [B,C,D,H,W] tensor
        x = torch.randn(2, 2, 8, 8, 8)
        tv_loss = tv3d_l1(x)
        assert tv_loss.shape == (), "TV loss should be scalar"
        assert tv_loss.item() >= 0, "TV loss should be non-negative"
        print("✅ TV3D function test passed")
        return True
    except Exception as e:
        print(f"❌ TV3D test error: {e}")
        return False

def test_ema():
    """Test EMA class."""
    try:
        from code.utils.ema import EMA
        import torch.nn as nn
        
        # Create simple model
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        
        # Test update
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        ema.update(model)
        
        # Test swap in/restore
        original_params = [p.clone() for p in model.parameters()]
        ema.swap_in(model)
        ema_params = [p.clone() for p in model.parameters()]
        ema.restore(model)
        restored_params = [p.clone() for p in model.parameters()]
        
        # Check that original == restored, and ema != original (after update)
        assert torch.allclose(original_params[0], restored_params[0]), "Parameters not properly restored"
        print("✅ EMA class test passed")
        return True
    except Exception as e:
        print(f"❌ EMA test error: {e}")
        return False

def test_config():
    """Test config imports."""
    try:
        from code.training.training_config import (
            STAGE1_USE_EMA, STAGE1_EMA_DECAY, STAGE1_TV_WEIGHT,
            STAGE1_LOSS_L1_W, STAGE1_LOSS_L2_W, STAGE1_PCT_START, STAGE1_FINAL_DIV_FACTOR
        )
        print("✅ Config imports successful")
        print(f"  EMA enabled: {STAGE1_USE_EMA}")
        print(f"  EMA decay: {STAGE1_EMA_DECAY}")
        print(f"  TV weight: {STAGE1_TV_WEIGHT}")
        print(f"  L1 weight: {STAGE1_LOSS_L1_W}")
        print(f"  L2 weight: {STAGE1_LOSS_L2_W}")
        print(f"  PCT_START: {STAGE1_PCT_START}")
        print(f"  FINAL_DIV_FACTOR: {STAGE1_FINAL_DIV_FACTOR}")
        return True
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def test_stage1_trainer_import():
    """Test that Stage1Trainer imports without error."""
    try:
        from code.training.stage1_trainer import Stage1Trainer
        print("✅ Stage1Trainer import successful")
        return True
    except Exception as e:
        print(f"❌ Stage1Trainer import error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Stage 1 upgrade smoke tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_tv3d,
        test_ema,
        test_config,
        test_stage1_trainer_import
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Stage 1 upgrades ready for training.")
    else:
        print("❌ Some tests failed. Check errors above.")
        sys.exit(1)
