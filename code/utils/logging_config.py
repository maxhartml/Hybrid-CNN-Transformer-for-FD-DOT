"""
Centralized logging configuration for NIR-DOT reconstruction pipeline.

Simple logging system with:
- Module-specific log files (data_processing, models, training, testing)
- Console + file output with rotation
- Easy DEBUG/INFO/WARNING/ERROR level control
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class NIRDOTLogger:
    """
    Centralized logging for the NIR-DOT pipeline.
    
    Features:
    - Automatic log directory creation
    - Module-specific log files  
    - Console + file output with rotation
    - Experiment tracking
    """
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logging(cls, 
                      log_dir: str = "logs",
                      log_level: str = "DEBUG",
                      max_file_size: int = 10 * 1024 * 1024,  # 10MB
                      backup_count: int = 5):
        """
        Initialize logging system.
        
        Args:
            log_dir: Directory for log files
            log_level: DEBUG, INFO, WARNING, or ERROR
            max_file_size: Max size before rotation (bytes)
            backup_count: Number of backup files to keep
        """
        if cls._initialized:
            return
        
        # Create log directories
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create module subdirectories
        modules = ['data_processing', 'models', 'training', 'testing']
        for module in modules:
            (log_path / module).mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Main file handler with rotation
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
        
        # Log initialization
        logger = cls.get_logger("logging_config")
        logger.info("🚀 NIR-DOT Logging Initialized")
        logger.info(f"📂 Log directory: {log_path.absolute()}")
        logger.info(f"📊 Log level: {log_level}")
        logger.info(f"🔄 File rotation: {max_file_size // (1024*1024)}MB, {backup_count} backups")
    
    @classmethod
    def get_logger(cls, 
                   name: str, 
                   module: Optional[str] = None,
                   log_dir: str = "logs") -> logging.Logger:
        """
        Get a logger for a specific component.
        
        Args:
            name: Logger name (usually module name)
            module: Module type for organized logging ('data_processing', 'models', etc.)
            log_dir: Base log directory
        
        Returns:
            Configured logger instance
        """
        # Ensure logging is initialized
        if not cls._initialized:
            cls.setup_logging(log_dir)
        
        # Return cached logger if exists
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        
        # Add module-specific file handler
        if module:
            log_path = Path(log_dir) / module
            log_path.mkdir(exist_ok=True)
            
            # Create rotating file handler
            module_log_file = log_path / f"{module}.log"
            module_handler = logging.handlers.RotatingFileHandler(
                module_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            module_handler.setLevel(logging.DEBUG)
            
            # Module formatter
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
        """Log the start of a training experiment."""
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
        """Log the completion of a training experiment."""
        logger = cls.get_logger("experiment", "training")
        logger.info("=" * 60)
        logger.info(f"🏁 EXPERIMENT COMPLETED: {experiment_name}")
        logger.info("📊 Final Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)


# Convenience functions for module-specific logging
def get_data_logger(name: str) -> logging.Logger:
    """Get logger for data processing components."""
    return NIRDOTLogger.get_logger(name, "data_processing")

def get_model_logger(name: str) -> logging.Logger:
    """Get logger for model components."""
    return NIRDOTLogger.get_logger(name, "models")

def get_training_logger(name: str) -> logging.Logger:
    """Get logger for training components."""
    return NIRDOTLogger.get_logger(name, "training")

def get_testing_logger(name: str) -> logging.Logger:
    """Get logger for testing components."""
    return NIRDOTLogger.get_logger(name, "testing")
