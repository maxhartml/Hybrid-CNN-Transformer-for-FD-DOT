"""
Comprehensive Testing Suite for NIR-DOT Reconstruction Pipeline.

This package provides extensive testing capabilities for validating every
component of the NIR-DOT reconstruction pipeline before training begins.
The testing suite ensures seamless integration, robust functionality,
and optimal performance across all system components.

Test Categories:
- Infrastructure Tests: Logging, utilities, and core systems validation
- Data Processing Tests: Data loading, analysis, and transformation validation
- Model Architecture Tests: All neural network components and variants
- Training Component Tests: Trainers, loss functions, and optimization
- Integration Tests: End-to-end pipeline workflows and component interaction
- Performance Tests: Speed, memory usage, and scalability assessment
- Error Handling Tests: Robustness validation and edge case handling

Components:
- ComprehensiveTestSuite: Main testing orchestrator with full pipeline validation
- Automated test discovery and execution
- Detailed performance monitoring and diagnostics
- Comprehensive error reporting and analysis

Features:
- Complete component validation before training
- Integration testing between all modules
- Performance benchmarking and optimization insights
- Memory usage monitoring and leak detection
- Automated error detection and robustness testing
- Detailed reporting with actionable insights

Usage:
    Run the comprehensive test suite to validate your entire pipeline:
    
    ```bash
    PYTHONPATH="$PWD" env_diss/bin/python code/testing/test_comprehensive.py
    ```
    
    This will execute all test categories and provide a detailed report
    on system readiness for training operations.
"""

from .test_comprehensive import ComprehensiveTestSuite

__all__ = ['ComprehensiveTestSuite']
