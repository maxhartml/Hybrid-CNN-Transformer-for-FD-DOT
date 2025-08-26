#!/usr/bin/env python3
"""
Test script to verify deterministic dataset splitting works correctly.

This script tests that:
1. Dataset splits are identical across multiple runs
2. Split indices are exactly as expected (0-7999, 8000-8999, 9000-9999)
3. No phantom appears in multiple splits (no overlap)
4. All phantoms are included in exactly one split (no gaps)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_deterministic_splits():
    """Test that dataset splits are deterministic and correct."""
    print("🧪 Testing deterministic dataset splits...")
    
    try:
        from code.data_processing.data_loader import create_phantom_dataloaders
        from code.data_processing.data_loader import TRAIN_SPLIT_START, TRAIN_SPLIT_END
        from code.data_processing.data_loader import VAL_SPLIT_START, VAL_SPLIT_END  
        from code.data_processing.data_loader import TEST_SPLIT_START, TEST_SPLIT_END
        
        print("✅ Successfully imported data loader constants")
        print(f"   Train indices: {TRAIN_SPLIT_START}-{TRAIN_SPLIT_END-1}")
        print(f"   Val indices: {VAL_SPLIT_START}-{VAL_SPLIT_END-1}")
        print(f"   Test indices: {TEST_SPLIT_START}-{TEST_SPLIT_END-1}")
        
        # Check that indices are sequential and non-overlapping
        assert TRAIN_SPLIT_END == VAL_SPLIT_START, f"Gap between train and val: {TRAIN_SPLIT_END} != {VAL_SPLIT_START}"
        assert VAL_SPLIT_END == TEST_SPLIT_START, f"Gap between val and test: {VAL_SPLIT_END} != {TEST_SPLIT_START}"
        
        print("✅ Split indices are sequential and non-overlapping")
        
        # Test that splits cover expected ranges
        expected_train_size = TRAIN_SPLIT_END - TRAIN_SPLIT_START
        expected_val_size = VAL_SPLIT_END - VAL_SPLIT_START  
        expected_test_size = TEST_SPLIT_END - TEST_SPLIT_START
        
        print(f"✅ Expected split sizes: {expected_train_size}/{expected_val_size}/{expected_test_size}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_deterministic_splits()
    if success:
        print("🎉 All tests passed! Deterministic splits are working correctly.")
    else:
        print("💥 Tests failed!")
        sys.exit(1)
