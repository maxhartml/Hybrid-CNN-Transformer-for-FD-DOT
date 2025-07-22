#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for NIR-DOT Reconstruction Pipeline.

This comprehensive test suite validates every component of the NIR-DOT pipeline
to ensure seamless integration and robust functionality before training begins.
The tests cover all modules, their interactions, and end-to-end workflows.

Test Categories:
1. Infrastructure Tests: Logging, utilities, and core systems
2. Data Processing Tests: Loading, validation, and transformations
3. Model Architecture Tests: All model components and variants
4. Training Component Tests: Trainers, loss functions, and optimization
5. Integration Tests: End-to-end pipeline validation
6. Performance Tests: Memory usage, speed, and scalability
7. Error Handling Tests: Robustness and edge cases

Features:
- Comprehensive component validation
- Integration testing between modules
- Performance and memory monitoring
- Detailed error reporting and diagnostics
- Automated test discovery and execution
- Full pipeline validation before training
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import traceback
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import h5py

# Import all components to test
from code.utils.logging_config import NIRDOTLogger, get_testing_logger
from code.data_processing.data_loader import create_nir_dataloaders, NIRPhantomDataset
from code.data_processing.data_analysis import NIRDatasetAnalyzer
from code.models.cnn_autoencoder import CNNAutoEncoder
from code.models.tissue_context_encoder import TissueContextEncoder, TissueContextToggle
from code.models.transformer_encoder import TransformerEncoder
from code.models.hybrid_model import HybridCNNTransformer
from code.training.stage1_trainer import Stage1Trainer, RMSELoss as Stage1RMSELoss
from code.training.stage2_trainer import Stage2Trainer, RMSELoss as Stage2RMSELoss

# Add nirfaster path for simulation testing
import sys
sys.path.append("nirfaster-FF")
try:
    from code.data_processing.data_simulator import build_phantom_with_tissue_and_tumours
    NIRFASTER_AVAILABLE = True
except ImportError as e:
    NIRFASTER_AVAILABLE = False
    print(f"Warning: nirfaster not available - {e}")

# Initialize comprehensive logging
logger = get_testing_logger(__name__)


class ComprehensiveTestSuite:
    """
    Comprehensive test suite for the entire NIR-DOT reconstruction pipeline.
    
    This class orchestrates extensive testing of all components, their integration,
    and end-to-end workflows to ensure the pipeline is ready for training.
    
    Attributes:
        device (torch.device): Test execution device
        test_results (Dict): Detailed results for all test categories
        performance_metrics (Dict): Performance monitoring data
        error_log (List): Comprehensive error tracking
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the comprehensive test suite.
        
        Args:
            device (str): Device for testing ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.test_results = {
            'infrastructure': {},
            'data_processing': {},
            'models': {},
            'training': {},
            'integration': {},
            'performance': {},
            'error_handling': {}
        }
        self.performance_metrics = {}
        self.error_log = []
        
        logger.info("🧪 Comprehensive Test Suite initialized")
        logger.info(f"🖥️  Test device: {self.device}")
        logger.info(f"💾 Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Execute the complete test suite with comprehensive validation.
        
        Returns:
            Dict: Complete test results with performance metrics and diagnostics
        """
        logger.info("🚀 Starting Comprehensive Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all test categories
        test_methods = [
            ("Infrastructure Tests", self._test_infrastructure),
            ("Data Processing Tests", self._test_data_processing),
            ("Model Architecture Tests", self._test_model_architectures),
            ("Training Component Tests", self._test_training_components),
            ("Integration Tests", self._test_integration),
            ("Performance Tests", self._test_performance),
            ("Error Handling Tests", self._test_error_handling)
        ]
        
        for category_name, test_method in test_methods:
            logger.info(f"\n🔬 {category_name}")
            logger.info("-" * 60)
            
            try:
                test_method()
                logger.info(f"✅ {category_name} completed successfully")
            except Exception as e:
                self.error_log.append(f"{category_name}: {str(e)}")
                logger.error(f"❌ {category_name} failed: {str(e)}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        total_time = time.time() - start_time
        self.performance_metrics['total_test_time'] = total_time
        
        # Generate comprehensive report
        self._generate_test_report()
        
        return {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'error_log': self.error_log
        }
    
    def _test_infrastructure(self):
        """Test all infrastructure components: logging, utilities, configuration."""
        logger.info("Testing logging system...")
        
        # Test logging configuration
        test_logger = get_testing_logger("test_component")
        test_logger.info("Test log message")
        self.test_results['infrastructure']['logging'] = True
        
        # Test experiment logging
        NIRDOTLogger.log_experiment_start("test_experiment", {
            "test_param": "test_value",
            "batch_size": 32
        })
        NIRDOTLogger.log_experiment_end("test_experiment", {
            "test_result": 0.95,
            "final_loss": 0.001
        })
        self.test_results['infrastructure']['experiment_logging'] = True
        
        # Test module-specific loggers
        data_logger = NIRDOTLogger.get_logger("test_data", "data_processing")
        model_logger = NIRDOTLogger.get_logger("test_model", "models")
        training_logger = NIRDOTLogger.get_logger("test_training", "training")
        
        data_logger.info("Data processing test log")
        model_logger.info("Model test log")
        training_logger.info("Training test log")
        
        self.test_results['infrastructure']['module_loggers'] = True
        logger.info("✅ Infrastructure tests passed")
    
    def _test_data_processing(self):
        """Test all data processing components comprehensively."""
        logger.info("Testing data loading and processing...")
        
        # Test data loader creation (baseline mode)
        logger.info("Testing baseline data loaders...")
        baseline_loaders = create_nir_dataloaders(
            data_dir="data",
            batch_size=2,
            use_tissue_patches=False
        )
        
        # Validate loader structure
        assert 'train' in baseline_loaders, "Missing train loader"
        assert 'val' in baseline_loaders, "Missing validation loader"
        assert 'test' in baseline_loaders, "Missing test loader"
        
        # Test batch loading
        train_batch = next(iter(baseline_loaders['train']))
        assert 'measurements' in train_batch, "Missing measurements in batch"
        assert 'ground_truth' in train_batch, "Missing ground_truth in batch"
        
        measurements_shape = train_batch['measurements'].shape
        ground_truth_shape = train_batch['ground_truth'].shape
        
        logger.info(f"Baseline batch shapes - Measurements: {measurements_shape}, Ground Truth: {ground_truth_shape}")
        
        # Validate batch dimensions for volumetric data
        assert len(measurements_shape) == 5, f"Expected 5D measurements (B,C,H,W,D), got {measurements_shape}"
        assert len(ground_truth_shape) == 5, f"Expected 5D ground_truth (B,H,W,D,C), got {ground_truth_shape}"
        assert measurements_shape[0] == ground_truth_shape[0], "Batch size mismatch"
        
        self.test_results['data_processing']['baseline_loading'] = True
        
        # Test enhanced data loader (with tissue patches)
        logger.info("Testing enhanced data loaders with tissue patches...")
        enhanced_loaders = create_nir_dataloaders(
            data_dir="data",
            batch_size=2,
            use_tissue_patches=True
        )
        
        enhanced_batch = next(iter(enhanced_loaders['train']))
        assert 'tissue_patches' in enhanced_batch, "Missing tissue patches in enhanced batch"
        
        tissue_patches_shape = enhanced_batch['tissue_patches'].shape
        logger.info(f"Enhanced batch shapes - Tissue patches: {tissue_patches_shape}")
        
        # Validate tissue patch dimensions (batch_size, num_patches, patch_features)
        # num_patches=2 (source+detector), patch_features=patch_size^3*2
        assert len(tissue_patches_shape) == 3, f"Expected 3D tissue patches (B,N,F), got {tissue_patches_shape}"
        assert tissue_patches_shape[1] == 2, f"Expected 2 patches per sample, got {tissue_patches_shape[1]}"
        
        self.test_results['data_processing']['enhanced_loading'] = True
        
        # Test dataset properties
        train_dataset = baseline_loaders['train'].dataset
        logger.info(f"Dataset properties - Train samples: {len(train_dataset)}")
        
        # Test data analysis components
        logger.info("Testing data analysis functions...")
        
        # Test with a single phantom file
        phantom_files = list(Path("data").glob("**/phantom_*.h5"))
        if phantom_files:
            test_phantom = phantom_files[0]
            logger.info(f"Testing analysis with: {test_phantom}")
            
            # Test phantom analysis with NIRDatasetAnalyzer
            try:
                analyzer = NIRDatasetAnalyzer(data_directory="data")
                # Test basic analyzer functionality
                assert hasattr(analyzer, 'analyze_single_dataset'), "Analyzer missing analyze_single_dataset method"
                assert hasattr(analyzer, 'comprehensive_analysis'), "Analyzer missing comprehensive_analysis method"
                self.test_results['data_processing']['phantom_analysis'] = True
                logger.info("NIRDatasetAnalyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Phantom analysis test skipped: {e}")
                self.test_results['data_processing']['phantom_analysis'] = False
        
        # Test data simulator components
        logger.info("Testing data simulator functions...")
        if NIRFASTER_AVAILABLE:
            try:
                # Test phantom building function
                test_phantom = build_phantom_with_tissue_and_tumours(
                    phantom_shape=(32, 32, 32),  # Smaller for testing
                    max_tumours=1,  # Fixed parameter name
                    rng_seed=42
                )
                assert test_phantom is not None, "Phantom generation failed"
                logger.info(f"Phantom simulation successful: shape {test_phantom.shape if hasattr(test_phantom, 'shape') else 'generated'}")
                self.test_results['data_processing']['simulation'] = True
            except Exception as e:
                logger.warning(f"Simulator test failed: {e}")
                self.test_results['data_processing']['simulation'] = False
        else:
            logger.warning("nirfaster not available - skipping simulation tests")
            self.test_results['data_processing']['simulation'] = False
        
        logger.info("✅ Data processing tests completed")
    
    def _test_model_architectures(self):
        """Test all model architectures and their components."""
        logger.info("Testing model architectures...")
        
        # Test CNN Autoencoder
        logger.info("Testing CNN Autoencoder...")
        cnn_model = CNNAutoEncoder(
            input_channels=1,
            output_size=(64, 64, 64),
            base_channels=64
        ).to(self.device)
        
        # Test forward pass
        test_input = torch.randn(2, 1, 64, 64, 64).to(self.device)
        cnn_output = cnn_model(test_input)
        
        assert cnn_output.shape == test_input.shape, f"CNN output shape mismatch: {cnn_output.shape} vs {test_input.shape}"
        logger.info(f"CNN Autoencoder output shape: {cnn_output.shape}")
        
        self.test_results['models']['cnn_autoencoder'] = True
        
        # Test Tissue Context Encoder
        logger.info("Testing Tissue Context Encoder...")
        context_encoder = TissueContextEncoder(
            patch_size=(7, 7, 7),
            num_patches=2,  # Source + detector patches (as per actual usage)
            embed_dim=256
        ).to(self.device)
        
        # Test tissue patches - shape should be (batch, num_patches, patch_size^3 * 2)
        # patch_size^3 * 2 = 7*7*7*2 = 686 (flattened tissue properties)
        patch_size_flat = 7 * 7 * 7 * 2  # 686
        tissue_patches = torch.randn(2, 2, patch_size_flat).to(self.device)  # batch, num_patches, flattened_patch
        context_output = context_encoder(tissue_patches)
        
        expected_context_shape = (2, 256)  # batch, embed_dim (context is aggregated across patches)
        assert context_output.shape == expected_context_shape, f"Context output shape mismatch: {context_output.shape} vs {expected_context_shape}"
        logger.info(f"Tissue Context Encoder output shape: {context_output.shape}")
        
        self.test_results['models']['tissue_context_encoder'] = True
        
        # Test Tissue Context Toggle
        logger.info("Testing Tissue Context Toggle...")
        
        # Test tissue patch processing toggle
        batch_size, num_patches, patch_features = 2, 2, 686
        tissue_patches = torch.randn(batch_size, num_patches, patch_features).to(self.device)
        
        # Test with tissue patches enabled
        processed_enabled = TissueContextToggle.process_tissue_patches(tissue_patches, use_tissue_patches=True)
        assert processed_enabled is not None, "Should return tissue patches when enabled"
        assert torch.equal(processed_enabled, tissue_patches), "Should return original patches when enabled"
        
        # Test with tissue patches disabled
        processed_disabled = TissueContextToggle.process_tissue_patches(tissue_patches, use_tissue_patches=False)
        assert processed_disabled is None, "Should return None when disabled"
        
        # Test dummy context creation
        dummy_context = TissueContextToggle.create_dummy_context(batch_size=2, embed_dim=256, device=self.device)
        assert dummy_context.shape == (2, 256), f"Dummy context shape mismatch: {dummy_context.shape}"
        logger.info(f"Tissue Context Toggle dummy context shape: {dummy_context.shape}")
        
        logger.info("Tissue Context Toggle working correctly")
        self.test_results['models']['tissue_context_toggle'] = True
        
        # Test Transformer Encoder
        logger.info("Testing Transformer Encoder...")
        # Use correct feature dimension that matches CNN encoder output
        cnn_feature_dim = 64 * 8  # base_channels * 8 = 512
        transformer = TransformerEncoder(
            cnn_feature_dim=cnn_feature_dim,
            num_heads=8,
            num_layers=6
        ).to(self.device)
        
        # Test transformer input - should be CNN features (2D), not a sequence (3D)
        cnn_features = torch.randn(2, cnn_feature_dim).to(self.device)  # batch_size, feature_dim
        tissue_context = torch.randn(2, 256).to(self.device)  # batch_size, tissue_context_dim
        
        # Test transformer forward pass with tissue context
        enhanced_features, attention_weights = transformer(cnn_features, tissue_context, use_tissue_patches=True)
        
        assert enhanced_features.shape == cnn_features.shape, f"Transformer output shape mismatch: {enhanced_features.shape} vs {cnn_features.shape}"
        assert attention_weights is not None, "Attention weights should be returned"
        logger.info(f"Transformer Encoder output shape: {enhanced_features.shape}")
        
        # Test without tissue context
        enhanced_features_no_tissue, _ = transformer(cnn_features, use_tissue_patches=False)
        assert enhanced_features_no_tissue.shape == cnn_features.shape, f"Transformer no-tissue output shape mismatch: {enhanced_features_no_tissue.shape} vs {cnn_features.shape}"
        
        self.test_results['models']['transformer_encoder'] = True
        
        # Test Hybrid Model (Baseline mode)
        logger.info("Testing Hybrid Model - Baseline mode...")
        hybrid_baseline = HybridCNNTransformer(use_tissue_patches=False).to(self.device)
        
        # Test baseline forward pass
        test_measurements = torch.randn(2, 1, 64, 64, 64).to(self.device)  # batch, channels, height, width, depth
        baseline_output = hybrid_baseline(test_measurements, tissue_patches=None)
        
        assert 'reconstructed' in baseline_output, "Missing reconstructed volume in baseline output"
        # Stage 1 only returns 'reconstructed' and 'stage', not 'features'
        assert 'stage' in baseline_output, "Missing stage in baseline output"
        
        reconstructed_shape = baseline_output['reconstructed'].shape
        expected_volume_shape = (2, 1, 64, 64, 64)
        assert reconstructed_shape == expected_volume_shape, f"Baseline reconstruction shape mismatch: {reconstructed_shape} vs {expected_volume_shape}"
        
        logger.info(f"Hybrid Baseline output shapes - Reconstructed: {reconstructed_shape}")
        self.test_results['models']['hybrid_baseline'] = True
        
        # Test Hybrid Model (Enhanced mode)
        logger.info("Testing Hybrid Model - Enhanced mode...")
        hybrid_enhanced = HybridCNNTransformer(use_tissue_patches=True).to(self.device)
        
        # Test enhanced forward pass with tissue patches
        enhanced_tissue_patches = torch.randint(0, 4, (2, 8, 7, 7, 7)).to(self.device)
        enhanced_output = hybrid_enhanced(test_measurements, tissue_patches=enhanced_tissue_patches)
        
        assert 'reconstructed' in enhanced_output, "Missing reconstructed volume in enhanced output"
        # Stage 1 only returns 'reconstructed' and 'stage', not 'features' or 'tissue_context' 
        assert 'stage' in enhanced_output, "Missing stage in enhanced output"
        
        enhanced_reconstructed_shape = enhanced_output['reconstructed'].shape
        assert enhanced_reconstructed_shape == expected_volume_shape, f"Enhanced reconstruction shape mismatch: {enhanced_reconstructed_shape} vs {expected_volume_shape}"
        
        logger.info(f"Hybrid Enhanced output shapes - Reconstructed: {enhanced_reconstructed_shape}")
        self.test_results['models']['hybrid_enhanced'] = True
        
        # Test model parameter counts
        self._test_model_parameters()
        
        logger.info("✅ Model architecture tests completed")
    
    def _test_model_parameters(self):
        """Test model parameter counts and memory usage."""
        logger.info("Testing model parameters and memory usage...")
        
        models_to_test = [
            ("CNN Autoencoder", CNNAutoEncoder(input_channels=1, output_size=(64, 64, 64))),
            ("Tissue Context Encoder", TissueContextEncoder(patch_size=(7, 7, 7), num_patches=2)),
            ("Transformer Encoder", TransformerEncoder(cnn_feature_dim=1024, num_heads=8, num_layers=6)),
            ("Hybrid Baseline", HybridCNNTransformer(use_tissue_patches=False)),
            ("Hybrid Enhanced", HybridCNNTransformer(use_tissue_patches=True))
        ]
        
        for model_name, model in models_to_test:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"{model_name} - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            self.performance_metrics[f'{model_name.lower().replace(" ", "_")}_params'] = {
                'total': total_params,
                'trainable': trainable_params
            }
    
    def _test_training_components(self):
        """Test all training components: trainers, loss functions, optimizers."""
        logger.info("Testing training components...")
        
        # Test RMSE Loss functions
        logger.info("Testing RMSE Loss functions...")
        stage1_loss = Stage1RMSELoss()
        stage2_loss = Stage2RMSELoss()
        
        # Test loss computation
        pred = torch.randn(2, 1, 64, 64, 64)
        target = torch.randn(2, 1, 64, 64, 64)
        
        loss1 = stage1_loss(pred, target)
        loss2 = stage2_loss(pred, target)
        
        assert loss1.item() > 0, "Stage 1 RMSE loss should be positive"
        assert loss2.item() > 0, "Stage 2 RMSE loss should be positive"
        assert torch.allclose(loss1, loss2), "RMSE losses should be identical"
        
        logger.info(f"RMSE Loss test value: {loss1.item():.6f}")
        self.test_results['training']['rmse_loss'] = True
        
        # Test Stage 1 Trainer initialization
        logger.info("Testing Stage 1 Trainer...")
        stage1_trainer = Stage1Trainer(
            learning_rate=1e-4,
            device=str(self.device)
        )
        
        # Verify trainer components
        assert hasattr(stage1_trainer, 'model'), "Stage 1 trainer missing model"
        assert hasattr(stage1_trainer, 'criterion'), "Stage 1 trainer missing criterion"
        assert hasattr(stage1_trainer, 'optimizer'), "Stage 1 trainer missing optimizer"
        
        # Test model is in correct mode
        assert not stage1_trainer.model.use_tissue_patches, "Stage 1 should not use tissue patches"
        
        logger.info("Stage 1 Trainer initialized successfully")
        self.test_results['training']['stage1_trainer'] = True
        
        # Create a temporary checkpoint for Stage 2 testing
        logger.info("Creating temporary checkpoint for Stage 2 testing...")
        checkpoint_path = "temp_stage1_checkpoint.pth"
        stage1_trainer.save_checkpoint(checkpoint_path, epoch=0, val_loss=0.1)
        
        # Test Stage 2 Trainer initialization
        logger.info("Testing Stage 2 Trainer...")
        
        # Test baseline mode
        stage2_baseline = Stage2Trainer(
            stage1_checkpoint_path=checkpoint_path,
            use_tissue_patches=False,
            learning_rate=5e-5,
            device=str(self.device)
        )
        
        assert not stage2_baseline.use_tissue_patches, "Stage 2 baseline should not use tissue patches"
        logger.info("Stage 2 Baseline Trainer initialized successfully")
        
        # Test enhanced mode
        stage2_enhanced = Stage2Trainer(
            stage1_checkpoint_path=checkpoint_path,
            use_tissue_patches=True,
            learning_rate=5e-5,
            device=str(self.device)
        )
        
        assert stage2_enhanced.use_tissue_patches, "Stage 2 enhanced should use tissue patches"
        logger.info("Stage 2 Enhanced Trainer initialized successfully")
        
        self.test_results['training']['stage2_trainer'] = True
        
        # Test frozen parameters in Stage 2
        logger.info("Testing frozen parameters in Stage 2...")
        trainable_params = sum(p.numel() for p in stage2_enhanced.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in stage2_enhanced.model.parameters())
        frozen_ratio = (total_params - trainable_params) / total_params
        
        logger.info(f"Stage 2 parameter freeze - Trainable: {trainable_params:,}/{total_params:,} ({100*(1-frozen_ratio):.1f}%)")
        assert frozen_ratio > 0.5, "Should have significant frozen parameters in Stage 2"
        
        self.test_results['training']['parameter_freezing'] = True
        
        # Clean up temporary checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        logger.info("✅ Training component tests completed")
    
    def _test_integration(self):
        """Test end-to-end integration between all components."""
        logger.info("Testing end-to-end integration...")
        
        # Test complete pipeline flow
        logger.info("Testing complete pipeline integration...")
        
        # 1. Load data
        data_loaders = create_nir_dataloaders(
            data_dir="data",
            batch_size=2,
            use_tissue_patches=True
        )
        
        # 2. Get a batch
        batch = next(iter(data_loaders['train']))
        measurements = batch['measurements'].to(self.device)
        volumes = batch['volumes'].to(self.device)  # Use correct key name from data loader  
        tissue_patches = batch['tissue_patches'].to(self.device)
        
        # 3. Test baseline model forward pass - configure for correct output size
        baseline_model = HybridCNNTransformer(
            output_size=(60, 60, 60),  # Match data dimensions
            use_tissue_patches=False
        ).to(self.device)
        baseline_output = baseline_model(measurements, tissue_patches=None)
        
        # 4. Test enhanced model forward pass - configure for correct output size  
        enhanced_model = HybridCNNTransformer(
            output_size=(60, 60, 60),  # Match data dimensions
            use_tissue_patches=True
        ).to(self.device)
        enhanced_output = enhanced_model(measurements, tissue_patches=tissue_patches)
        
        # 5. Test loss computation - use only one channel for comparison
        loss_fn = Stage1RMSELoss()
        # Extract first channel from ground truth to match model output (1 channel)
        volumes_single = volumes[..., 0:1].permute(0, 4, 1, 2, 3)  # (B, H, W, D, 1) -> (B, 1, H, W, D)
        baseline_loss = loss_fn(baseline_output['reconstructed'], volumes_single)
        enhanced_loss = loss_fn(enhanced_output['reconstructed'], volumes_single)
        
        logger.info(f"Integration test losses - Baseline: {baseline_loss.item():.6f}, Enhanced: {enhanced_loss.item():.6f}")
        
        # 6. Test gradient computation
        baseline_loss.backward()
        enhanced_loss.backward()
        
        # Verify gradients exist
        baseline_has_grads = any(p.grad is not None for p in baseline_model.parameters() if p.requires_grad)
        enhanced_has_grads = any(p.grad is not None for p in enhanced_model.parameters() if p.requires_grad)
        
        assert baseline_has_grads, "Baseline model should have gradients after backward pass"
        assert enhanced_has_grads, "Enhanced model should have gradients after backward pass"
        
        self.test_results['integration']['pipeline_flow'] = True
        
        # Test Stage 1 to Stage 2 transition
        logger.info("Testing Stage 1 to Stage 2 transition...")
        
        # Create Stage 1 trainer and save checkpoint
        stage1_trainer = Stage1Trainer(learning_rate=1e-4, device=str(self.device))
        checkpoint_path = "temp_integration_checkpoint.pth"
        stage1_trainer.save_checkpoint(checkpoint_path, epoch=0, val_loss=0.05)
        
        # Initialize Stage 2 trainer with Stage 1 checkpoint
        stage2_trainer = Stage2Trainer(
            stage1_checkpoint_path=checkpoint_path,
            use_tissue_patches=True,
            learning_rate=5e-5,
            device=str(self.device)
        )
        
        # Verify Stage 1 weights were loaded
        stage1_state = stage1_trainer.model.state_dict()
        stage2_state = stage2_trainer.model.state_dict()
        
        # Check that CNN components match
        cnn_keys = [k for k in stage1_state.keys() if 'cnn_autoencoder' in k]
        for key in cnn_keys:
            if key in stage2_state:
                assert torch.allclose(stage1_state[key], stage2_state[key]), f"Mismatch in transferred weights: {key}"
        
        logger.info("Stage 1 to Stage 2 transition successful")
        self.test_results['integration']['stage_transition'] = True
        
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        logger.info("✅ Integration tests completed")
    
    def _test_performance(self):
        """Test performance characteristics: speed, memory usage, scalability."""
        logger.info("Testing performance characteristics...")
        
        # Test inference speed
        logger.info("Testing inference speed...")
        model = HybridCNNTransformer(use_tissue_patches=True).to(self.device)
        model.eval()
        
        batch_sizes = [1, 2]  # Reduced for faster testing
        inference_times = {}
        
        with torch.no_grad():
            for batch_size in batch_sizes:
                measurements = torch.randn(batch_size, 1, 64, 64, 64).to(self.device)  # volumetric input
                tissue_patches = torch.randn(batch_size, 2, 686).to(self.device)  # flattened patches
                
                # Warm up (reduced iterations)
                for _ in range(2):
                    _ = model(measurements, tissue_patches)
                
                # Time inference (reduced iterations for speed)
                start_time = time.time()
                for _ in range(3):  # Reduced from 10 to 3
                    output = model(measurements, tissue_patches)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 3
                inference_times[batch_size] = avg_time
                
                logger.info(f"Batch size {batch_size}: {avg_time:.4f}s per inference")
        
        self.performance_metrics['inference_times'] = inference_times
        
        # Test memory usage
        logger.info("Testing memory usage...")
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            # Create models and run forward pass
            large_batch = torch.randn(8, 1024).to(self.device)
            tissue_patches = torch.randint(0, 4, (8, 8, 7, 7, 7)).to(self.device)
            output = model(large_batch, tissue_patches)
            
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / (1024**2)  # MB
            
            logger.info(f"Memory usage for batch size 8: {memory_used:.2f} MB")
            self.performance_metrics['memory_usage_mb'] = memory_used
        else:
            logger.info("CUDA not available, skipping GPU memory test")
        
        # Test data loading speed
        logger.info("Testing data loading speed...")
        data_loaders = create_nir_dataloaders(
            data_dir="data",
            batch_size=4,
            use_tissue_patches=True
        )
        
        start_time = time.time()
        batch_count = 0
        for batch in data_loaders['train']:
            batch_count += 1
            if batch_count >= 10:  # Test first 10 batches
                break
        end_time = time.time()
        
        avg_batch_time = (end_time - start_time) / batch_count
        logger.info(f"Average batch loading time: {avg_batch_time:.4f}s")
        self.performance_metrics['batch_loading_time'] = avg_batch_time
        
        self.test_results['performance']['speed_tests'] = True
        logger.info("✅ Performance tests completed")
    
    def _test_error_handling(self):
        """Test error handling and edge cases."""
        logger.info("Testing error handling and edge cases...")
        
        # Test invalid input shapes
        logger.info("Testing invalid input handling...")
        model = HybridCNNTransformer(use_tissue_patches=False).to(self.device)
        
        try:
            # Wrong measurement dimension
            invalid_measurements = torch.randn(2, 512).to(self.device)  # Should be 1024
            output = model(invalid_measurements, tissue_patches=None)
            self.test_results['error_handling']['invalid_shapes'] = False
            logger.warning("Model should have failed with invalid input shape")
        except Exception as e:
            logger.info(f"✅ Correctly caught invalid input: {type(e).__name__}")
            self.test_results['error_handling']['invalid_shapes'] = True
        
        # Test missing tissue patches when required
        try:
            enhanced_model = HybridCNNTransformer(use_tissue_patches=True).to(self.device)
            measurements = torch.randn(2, 1024).to(self.device)
            output = enhanced_model(measurements, tissue_patches=None)
            logger.warning("Enhanced model should require tissue patches")
            self.test_results['error_handling']['missing_tissue_patches'] = False
        except Exception as e:
            logger.info(f"✅ Correctly caught missing tissue patches: {type(e).__name__}")
            self.test_results['error_handling']['missing_tissue_patches'] = True
        
        # Test checkpoint loading with wrong path
        try:
            stage2_trainer = Stage2Trainer(
                stage1_checkpoint_path="nonexistent_checkpoint.pth",
                use_tissue_patches=False,
                device=str(self.device)
            )
            self.test_results['error_handling']['invalid_checkpoint'] = False
            logger.warning("Should have failed with invalid checkpoint path")
        except Exception as e:
            logger.info(f"✅ Correctly caught invalid checkpoint: {type(e).__name__}")
            self.test_results['error_handling']['invalid_checkpoint'] = True
        
        # Test data loader with invalid directory
        try:
            invalid_loaders = create_nir_dataloaders(
                data_dir="nonexistent_directory",
                batch_size=2
            )
            self.test_results['error_handling']['invalid_data_dir'] = False
            logger.warning("Should have failed with invalid data directory")
        except Exception as e:
            logger.info(f"✅ Correctly caught invalid data directory: {type(e).__name__}")
            self.test_results['error_handling']['invalid_data_dir'] = True
        
        logger.info("✅ Error handling tests completed")
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 COMPREHENSIVE TEST SUITE REPORT")
        logger.info("=" * 80)
        
        # Count passed/failed tests
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            category_total = len(tests)
            category_passed = sum(1 for result in tests.values() if result)
            
            total_tests += category_total
            passed_tests += category_passed
            
            logger.info(f"\n📊 {category.upper().replace('_', ' ')}:")
            for test_name, result in tests.items():
                status = "✅" if result else "❌"
                logger.info(f"  {status} {test_name}")
            
            logger.info(f"  📈 Category Score: {category_passed}/{category_total} ({100*category_passed/category_total:.1f}%)" if category_total > 0 else f"  📈 Category Score: 0/0 (N/A)")
        
        # Overall results
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        logger.info(f"\n🎯 OVERALL RESULTS:")
        logger.info(f"  📊 Tests Passed: {passed_tests}/{total_tests} ({100*success_rate:.1f}%)")
        logger.info(f"  ⏱️  Total Time: {self.performance_metrics.get('total_test_time', 0):.2f}s")
        
        if self.error_log:
            logger.info(f"\n❌ ERRORS ENCOUNTERED:")
            for error in self.error_log:
                logger.error(f"  • {error}")
        
        # Performance summary
        if self.performance_metrics:
            logger.info(f"\n⚡ PERFORMANCE METRICS:")
            for metric, value in self.performance_metrics.items():
                if isinstance(value, dict):
                    logger.info(f"  📈 {metric}:")
                    for k, v in value.items():
                        logger.info(f"    - {k}: {v}")
                else:
                    logger.info(f"  📈 {metric}: {value}")
        
        # Final verdict
        if success_rate >= 0.95:
            logger.info(f"\n🚀 VERDICT: EXCELLENT! Pipeline ready for training!")
        elif success_rate >= 0.85:
            logger.info(f"\n✅ VERDICT: GOOD! Minor issues may need attention.")
        elif success_rate >= 0.70:
            logger.info(f"\n⚠️  VERDICT: FAIR! Several issues need fixing before training.")
        else:
            logger.info(f"\n❌ VERDICT: POOR! Significant issues must be resolved.")
        
        logger.info("=" * 80)


def main():
    """Run the comprehensive test suite."""
    # Setup logging
    NIRDOTLogger.setup_logging(log_level="INFO")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize and run test suite
    test_suite = ComprehensiveTestSuite(device=device)
    results = test_suite.run_all_tests()
    
    return results


if __name__ == "__main__":
    main()
