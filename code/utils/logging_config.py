"""
Centralized logging configuration for the NIR-DOT reconstruction pipeline.

This module provides a comprehensive logging system with module-specific loggers,
organized log file structure, and consistent formatting across all pipeline components.
Designed to facilitate debugging, monitoring, and experiment tracking.

Key Features:
- Module-specific log files and directories
- Rotating file handlers to manage disk space
- Consistent formatting with timestamps and context
- Experiment tracking capabilities
- Performance monitoring support
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class NIRDOTLogger:
    """
    Centralized logging system for the NIR-DOT reconstruction pipeline.
    
    Provides a unified logging interface with automatic log file organization,
    rotation, and module-specific separation. Supports both development
    debugging and production monitoring workflows.
    
    Features:
    - Automatic log directory structure creation
    - Module-specific log files (data_processing, models, training, testing)
    - Rotating file handlers with configurable size limits
    - Console output for immediate feedback
    - Experiment tracking and result logging
    
    Class Attributes:
        _loggers (dict): Cache of created logger instances
        _initialized (bool): Initialization status flag
    """
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logging(cls, 
                      log_dir: str = "logs",
                      log_level: str = "INFO",
                      max_file_size: int = 10 * 1024 * 1024,  # 10MB
                      backup_count: int = 5):
        """
        Initialize the centralized logging system.
        
        Sets up the complete logging infrastructure including directory structure,
        file handlers, formatters, and console output. Should be called once at
        application startup.
        
        Args:
            log_dir (str, optional): Base directory for all log files. Defaults to "logs".
            log_level (str, optional): Minimum logging level (DEBUG, INFO, WARNING, ERROR). 
                Defaults to "INFO".
            max_file_size (int, optional): Maximum size per log file before rotation in bytes. 
                Defaults to 10MB.
            backup_count (int, optional): Number of backup files to retain during rotation. 
                Defaults to 5.
        """
        if cls._initialized:
            return
        
        # Create hierarchical log directory structure
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create module-specific subdirectories for organized logging
        modules = ['data_processing', 'models', 'training', 'testing']
        for module in modules:
            (log_path / module).mkdir(exist_ok=True)
        
        # Configure root logger with appropriate level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers to avoid duplication
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create consistent formatter for all handlers
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler for immediate development feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Main log file handler with rotation (comprehensive logging)
        main_log_file = log_path / "main.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file, 
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        main_handler.setLevel(getattr(logging, log_level.upper()))
        main_handler.setFormatter(formatter)
        root_logger.addHandler(main_handler)
        
        cls._initialized = True
        
        # Log successful initialization with system info
        logger = cls.get_logger("logging_config")
        logger.info("🚀 NIR-DOT Logging System Initialized")
        logger.info(f"📂 Log directory: {log_path.absolute()}")
        logger.info(f"📊 Log level: {log_level}")
        logger.info(f"🔄 File rotation: {max_file_size // (1024*1024)}MB, {backup_count} backups")
    
    @classmethod
    def get_logger(cls, 
                   name: str, 
                   module: Optional[str] = None,
                   log_dir: str = "logs") -> logging.Logger:
        """
        Get a configured logger instance for a specific module or component.
        
        This method provides the primary interface for obtaining loggers throughout
        the NIR-DOT pipeline. It ensures proper initialization and creates module-
        specific log files when requested, enabling organized log management.
        
        Args:
            name (str): Logger name identifier, typically the module name or component.
                       This appears in log messages and helps identify the source.
            module (str, optional): Module type for organized logging. When specified,
                                  creates a dedicated subdirectory and log file.
                                  Common values: 'data_processing', 'models', 
                                  'training', 'testing'.
            log_dir (str): Base log directory path. Defaults to "logs".
        
        Returns:
            logging.Logger: A fully configured logger instance with appropriate
                           handlers, formatters, and rotation settings. Includes
                           both console output and file logging capabilities.
        
        Example:
            >>> # Basic logger for general use
            >>> logger = NIRDOTLogger.get_logger("main")
            >>> 
            >>> # Module-specific logger with dedicated file
            >>> data_logger = NIRDOTLogger.get_logger("phantom_loader", "data_processing")
            >>> data_logger.info("Loading phantom scan...")
            >>> # Creates logs/data_processing/data_processing.log
        """
        # Ensure logging is initialized
        if not cls._initialized:
            cls.setup_logging(log_dir)
        
        # Return existing logger if already created
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        
        # Add module-specific file handler if module is specified
        if module:
            log_path = Path(log_dir) / module
            log_path.mkdir(exist_ok=True)
            
            # Create rotating file handler for this module
            module_log_file = log_path / f"{module}.log"
            module_handler = logging.handlers.RotatingFileHandler(
                module_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            module_handler.setLevel(logging.DEBUG)
            
            # Module-specific formatter
            module_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(funcName)-15s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            module_handler.setFormatter(module_formatter)
            
            logger.addHandler(module_handler)
        
        # Cache the logger
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def log_experiment_start(cls, experiment_name: str, config: dict):
        """
        Log the start of a training experiment with comprehensive configuration details.
        
        This method creates a formatted log entry marking the beginning of an experiment,
        including all configuration parameters for reproducibility and debugging.
        The logs are automatically directed to the training module's log files.
        
        Args:
            experiment_name (str): Human-readable name for the experiment.
                                 Used for identification in logs and results tracking.
            config (dict): Complete configuration dictionary containing all
                          experiment parameters, hyperparameters, and settings.
        
        Example:
            >>> config = {
            ...     "model_type": "hybrid_transformer",
            ...     "learning_rate": 0.001,
            ...     "batch_size": 32,
            ...     "epochs": 100
            ... }
            >>> NIRDOTLogger.log_experiment_start("phantom_reconstruction_v2", config)
        """
        logger = cls.get_logger("experiment", "training")
        logger.info("=" * 60)
        logger.info(f"🧪 EXPERIMENT STARTED: {experiment_name}")
        logger.info("=" * 60)
        logger.info("📋 Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
    
    @classmethod
    def log_experiment_end(cls, experiment_name: str, results: dict):
        """
        Log the completion of a training experiment with final results.
        
        This method creates a formatted log entry marking the end of an experiment,
        including all final metrics and results for analysis and comparison.
        The logs complement the experiment start logs for complete tracking.
        
        Args:
            experiment_name (str): Name of the completed experiment, should match
                                 the name used in log_experiment_start().
            results (dict): Final results dictionary containing metrics, scores,
                           and any other relevant experiment outcomes.
        
        Example:
            >>> results = {
            ...     "final_loss": 0.0234,
            ...     "validation_accuracy": 0.956,
            ...     "training_time": "2h 34m",
            ...     "best_epoch": 87
            ... }
            >>> NIRDOTLogger.log_experiment_end("phantom_reconstruction_v2", results)
        """
        logger = cls.get_logger("experiment", "training")
        logger.info("=" * 60)
        logger.info(f"🏁 EXPERIMENT COMPLETED: {experiment_name}")
        logger.info("📊 Final Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)


# Convenience functions for each module
def get_data_logger(name: str) -> logging.Logger:
    """Get logger for data processing components"""
    return NIRDOTLogger.get_logger(name, "data_processing")

def get_model_logger(name: str) -> logging.Logger:
    """Get logger for model components"""
    return NIRDOTLogger.get_logger(name, "models")

# Convenience functions for module-specific logging
def get_data_logger(name: str) -> logging.Logger:
    """
    Get a logger specifically configured for data processing components.
    
    Args:
        name (str): Component name, typically the module or function name.
    
    Returns:
        logging.Logger: Logger instance configured for data processing module
                       with output directed to logs/data_processing/.
    """
    return NIRDOTLogger.get_logger(name, "data_processing")

def get_model_logger(name: str) -> logging.Logger:
    """
    Get a logger specifically configured for model components.
    
    Args:
        name (str): Component name, typically the model class or module name.
    
    Returns:
        logging.Logger: Logger instance configured for models module
                       with output directed to logs/models/.
    """
    return NIRDOTLogger.get_logger(name, "models")

def get_training_logger(name: str) -> logging.Logger:
    """
    Get a logger specifically configured for training components.
    
    Args:
        name (str): Component name, typically the trainer class or module name.
    
    Returns:
        logging.Logger: Logger instance configured for training module
                       with output directed to logs/training/.
    """
    return NIRDOTLogger.get_logger(name, "training")

def get_testing_logger(name: str) -> logging.Logger:
    """
    Get a logger specifically configured for testing components.
    
    Args:
        name (str): Component name, typically the test class or module name.
    
    Returns:
        logging.Logger: Logger instance configured for testing module
                       with output directed to logs/testing/.
    """
    return NIRDOTLogger.get_logger(name, "testing")
