#!/usr/bin/env python3
"""
Quick validation script to ensure refactoring worked correctly.
This script checks that all the global settings are properly configured.
"""

def validate_refactoring():
    """Validate that all refactoring changes are working correctly."""
    print("🔍 Validating refactoring changes...")
    
    # Test 1: Import global constants
    try:
        from code.training.training_config import GLOBAL_SEED, GLOBAL_POOLING_QUERIES
        print(f"✅ Global constants imported successfully:")
        print(f"   GLOBAL_SEED = {GLOBAL_SEED}")
        print(f"   GLOBAL_POOLING_QUERIES = {GLOBAL_POOLING_QUERIES}")
    except Exception as e:
        print(f"❌ Failed to import global constants: {e}")
        return False
    
    # Test 2: Check data loader uses global seed
    try:
        from code.data_processing.data_loader import DEFAULT_RANDOM_SEED
        print(f"✅ Data loader default seed: {DEFAULT_RANDOM_SEED}")
        assert DEFAULT_RANDOM_SEED == GLOBAL_SEED, f"Expected {GLOBAL_SEED}, got {DEFAULT_RANDOM_SEED}"
    except Exception as e:
        print(f"❌ Data loader seed check failed: {e}")
        return False
    
    # Test 3: Check global pooling encoder config
    try:
        from code.models.global_pooling_encoder import NUM_POOL_QUERIES
        print(f"✅ Global pooling queries: {NUM_POOL_QUERIES}")
        assert NUM_POOL_QUERIES == GLOBAL_POOLING_QUERIES, f"Expected {GLOBAL_POOLING_QUERIES}, got {NUM_POOL_QUERIES}"
    except Exception as e:
        print(f"❌ Global pooling config check failed: {e}")
        return False
    
    print("🎉 All refactoring validations passed! Ready for training.")
    return True

if __name__ == "__main__":
    validate_refactoring()
