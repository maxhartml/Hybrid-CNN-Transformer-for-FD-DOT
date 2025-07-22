"""
Shared utilities for NIR-DOT pipeline infrastructure.

This package provides common utilities and infrastructure components
used across the entire NIR-DOT reconstruction pipeline, including
logging, configuration management, and shared helper functions.

Components:
- logging_config: Centralized logging system with module-specific loggers
- Future utilities for configuration, metrics, and other shared functionality
"""

from .logging_config import (
    NIRDOTLogger,
    get_data_logger,
    get_model_logger, 
    get_training_logger,
    get_testing_logger
)

__all__ = [
    'NIRDOTLogger',
    'get_data_logger',
    'get_model_logger',
    'get_training_logger', 
    'get_testing_logger'
]
