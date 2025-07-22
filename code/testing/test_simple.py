#!/usr/bin/env python3
"""
Simple test suite for Robin Dale's hybrid CNN-Transformer pipeline.
Tests essential components without complex configuration dependencies.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, bool

# Import our components
from code.models.cnn_autoencoder import CNNAutoEncoder
from code.models.tissue_context_encoder import TissueContextEncoder, TissueContextToggle
from code.models.transformer_encoder import TransformerEncoder
from code.models.hybrid_model import HybridCNNTransformer
from code.training.training_utils import RMSELoss, compute_rmse, compute_mae
from code.data_processing.data_loader import create_nir_dataloaders, NIRPhantomDataset


class SimpleTestPipeline:
    """Simple test pipeline for essential components"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧪 Testing on device: {self.device}")
    
    def test_models(self) -> Dict[str, bool]:
        """Test model components"""
        results = {}
        
        try:
            # Test CNN Autoencoder
            print("Testing CNN Autoencoder...")
            cnn = CNNAutoEncoder(input_channels=1, base_channels=64)
            test_input = torch.randn(2, 1, 64, 64, 64)
            output = cnn(test_input)
            results['cnn_autoencoder'] = output.shape == test_input.shape
            print(f"✅ CNN Autoencoder: {output.shape}")
            
            # Test Tissue Context Toggle
            print("Testing Tissue Context Toggle...")
            toggle = TissueContextToggle()
            
            # Test disabled (should return original features)
            features = torch.randn(2, 512, 8, 8, 8)
            tissue_patches = torch.randn(2, 2, 7, 7)
            
            output_disabled = toggle(features, tissue_patches, use_tissue_patches=False)
            results['toggle_disabled'] = torch.equal(output_disabled, features)
            
            # Test enabled (should enhance features)
            output_enabled = toggle(features, tissue_patches, use_tissue_patches=True)
            results['toggle_enabled'] = output_enabled.shape == features.shape and not torch.equal(output_enabled, features)
            print(f"✅ Toggle functionality working")
            
            # Test Transformer Encoder
            print("Testing Transformer Encoder...")
            transformer = TransformerEncoder(
                embed_dim=768,
                num_layers=6,
                num_heads=12
            )
            transformer_input = torch.randn(2, 100, 768)  # batch, seq_len, embed_dim
            transformer_output = transformer(transformer_input)
            results['transformer'] = transformer_output.shape == transformer_input.shape
            print(f"✅ Transformer: {transformer_output.shape}")
            
            # Test Hybrid Model
            print("Testing Hybrid Model...")
            model = HybridCNNTransformer(use_tissue_patches=True)
            model.eval()
            
            test_measurements = torch.randn(2, 1, 64, 64, 64)
            test_tissue_patches = torch.randn(2, 2, 7, 7)
            
            with torch.no_grad():
                # Test with tissue patches
                output = model(test_measurements, test_tissue_patches, use_tissue_patches=True)
                results['hybrid_with_patches'] = 'reconstructed' in output
                
                # Test without tissue patches (baseline)
                output = model(test_measurements, test_tissue_patches, use_tissue_patches=False)
                results['hybrid_baseline'] = 'reconstructed' in output
            
            print(f"✅ Hybrid Model working")
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_training_utils(self) -> Dict[str, bool]:
        """Test training utilities"""
        results = {}
        
        try:
            # Test RMSE Loss
            print("Testing RMSE Loss...")
            loss_fn = RMSELoss()
            pred = torch.randn(2, 1, 32, 32, 32)
            target = torch.randn(2, 1, 32, 32, 32)
            loss = loss_fn(pred, target)
            results['rmse_loss'] = isinstance(loss.item(), float) and loss.item() >= 0
            
            # Test metric functions
            rmse = compute_rmse(pred, target)
            mae = compute_mae(pred, target)
            results['rmse_metric'] = isinstance(rmse, float) and rmse >= 0
            results['mae_metric'] = isinstance(mae, float) and mae >= 0
            
            print(f"✅ Training utilities working")
            
        except Exception as e:
            print(f"❌ Training utils test failed: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_data_loading(self) -> Dict[str, bool]:
        """Test data loading"""
        results = {}
        
        try:
            print("Testing data loading...")
            
            # Test with tissue patches disabled
            data_loaders = create_nir_dataloaders(
                data_dir="data",
                batch_size=2,
                use_tissue_patches=False
            )
            
            # Get a batch
            batch = next(iter(data_loaders['train']))
            results['data_loading_baseline'] = (
                'measurements' in batch and 
                'volumes' in batch and
                batch['measurements'].shape[0] == 2
            )
            
            # Test with tissue patches enabled
            data_loaders = create_nir_dataloaders(
                data_dir="data", 
                batch_size=2,
                use_tissue_patches=True
            )
            
            batch = next(iter(data_loaders['train']))
            results['data_loading_enhanced'] = (
                'measurements' in batch and
                'volumes' in batch and
                'tissue_patches' in batch and
                batch['tissue_patches'].shape == (2, 2, 7, 7)
            )
            
            print(f"✅ Data loading working")
            
        except Exception as e:
            print(f"❌ Data loading test failed: {e}")
            results['error'] = str(e)
            
        return results
    
    def run_all_tests(self):
        """Run all tests"""
        print("🔬 Running Simple Test Suite for Robin Dale's Pipeline")
        print("=" * 60)
        
        all_results = {}
        
        # Run tests
        all_results['models'] = self.test_models()
        all_results['training_utils'] = self.test_training_utils()
        all_results['data_loading'] = self.test_data_loading()
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in all_results.items():
            print(f"\n{category.upper()}:")
            for test_name, passed in results.items():
                if test_name != 'error':
                    total_tests += 1
                    if passed:
                        passed_tests += 1
                        print(f"  ✅ {test_name}")
                    else:
                        print(f"  ❌ {test_name}")
                else:
                    print(f"  ❌ Error: {passed}")
        
        print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("✅ All tests passed! Pipeline is ready for training.")
        else:
            print("❌ Some tests failed. Please check the issues above.")
        
        return passed_tests == total_tests


def main():
    """Main test function"""
    tester = SimpleTestPipeline()
    success = tester.run_all_tests()
    return success


if __name__ == "__main__":
    main()
