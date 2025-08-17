#!/usr/bin/env python3
"""
Common Training Utilities for NIR-DOT Pipeline

This module contains shared utility functions used across Stage 1 and Stage 2
training to reduce code duplication and ensure consistency.

Functions:
    save_checkpoint: Common checkpoint saving logic with timestamped filenames
    find_best_checkpoint: Automatic best checkpoint selection based on validation loss
    parse_checkpoint_filename: Parse timestamp and loss from checkpoint filenames
    setup_checkpoint_directory: Directory setup for checkpoints
"""

import os
import re
import glob
import torch
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from pathlib import Path
from code.utils.logging_config import get_training_logger

# Initialize module logger
logger = get_training_logger(__name__)


def generate_checkpoint_filename(stage_name: str, val_loss: float, base_dir: str = "checkpoints") -> str:
    """
    Generate a timestamped checkpoint filename with validation loss.
    
    Format: checkpoint_stage{N}_{timestamp}_loss{val_loss:.4f}.pt
    Example: checkpoint_stage1_20250817-201530_loss0.3917.pt
    
    Args:
        stage_name: Stage identifier ("stage1", "stage2_baseline", "stage2_enhanced")
        val_loss: Validation loss value for filename
        base_dir: Base checkpoint directory
        
    Returns:
        Full path to the checkpoint file
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Format loss to 4 decimal places
    loss_str = f"{val_loss:.4f}"
    
    # Create filename
    filename = f"checkpoint_{stage_name}_{timestamp}_loss{loss_str}.pt"
    
    # Return full path
    return os.path.join(base_dir, filename)


def parse_checkpoint_filename(filepath: str) -> Optional[Tuple[str, str, float]]:
    """
    Parse a checkpoint filename to extract stage, timestamp, and validation loss.
    
    Expected format: checkpoint_{stage}_{timestamp}_loss{val_loss}.pt
    
    Args:
        filepath: Path to the checkpoint file
        
    Returns:
        Tuple of (stage, timestamp, val_loss) if parsing succeeds, None otherwise
    """
    filename = os.path.basename(filepath)
    
    # Pattern to match: checkpoint_stage1_20250817-201530_loss0.3917.pt
    pattern = r'checkpoint_(stage\d+(?:_\w+)?|stage\d+)_(\d{8}-\d{6})_loss(\d+\.\d+)\.pt'
    
    match = re.match(pattern, filename)
    if match:
        stage = match.group(1)
        timestamp = match.group(2)
        val_loss = float(match.group(3))
        return stage, timestamp, val_loss
    
    return None


def find_best_checkpoint(checkpoint_dir: str, stage_pattern: str) -> Optional[str]:
    """
    Find the checkpoint with the lowest validation loss for a given stage.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        stage_pattern: Stage pattern to match (e.g., "stage1", "stage2_baseline")
        
    Returns:
        Path to the best checkpoint file, or None if no valid checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"📁 Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Find all checkpoint files matching the pattern
    pattern = f"checkpoint_{stage_pattern}_*_loss*.pt"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        logger.warning(f"🔍 No checkpoint files found matching pattern: {pattern}")
        return None
    
    logger.info(f"🔍 Found {len(checkpoint_files)} checkpoint files for {stage_pattern}")
    
    best_checkpoint = None
    best_loss = float('inf')
    valid_checkpoints = []
    
    for checkpoint_file in checkpoint_files:
        parsed = parse_checkpoint_filename(checkpoint_file)
        if parsed:
            stage, timestamp, val_loss = parsed
            valid_checkpoints.append((checkpoint_file, stage, timestamp, val_loss))
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = checkpoint_file
        else:
            logger.debug(f"⚠️ Could not parse checkpoint filename: {checkpoint_file}")
    
    if best_checkpoint:
        parsed = parse_checkpoint_filename(best_checkpoint)
        stage, timestamp, val_loss = parsed
        logger.info(f"🏆 BEST CHECKPOINT SELECTED:")
        logger.info(f"   📄 File: {os.path.basename(best_checkpoint)}")
        logger.info(f"   📅 Timestamp: {timestamp}")
        logger.info(f"   📊 Validation Loss: {val_loss:.6f}")
        logger.info(f"   🔢 Selected from {len(valid_checkpoints)} valid checkpoints")
        
        # Log all available checkpoints for reference
        logger.debug(f"📋 All valid checkpoints found:")
        for filepath, stage, timestamp, val_loss in sorted(valid_checkpoints, key=lambda x: x[3]):
            logger.debug(f"   • {os.path.basename(filepath)} (loss: {val_loss:.6f}, time: {timestamp})")
    
    return best_checkpoint


def save_checkpoint(path: str, 
                   epoch: int,
                   model_state: Dict[str, Any],
                   optimizer_state: Dict[str, Any], 
                   val_loss: float,
                   stage_name: str,
                   extra_data: Optional[Dict[str, Any]] = None,
                   use_timestamped_name: bool = True) -> str:
    """
    Save training checkpoint with common structure across stages.
    
    Args:
        path: File path for saving the checkpoint (used as base if timestamped)
        epoch: Current training epoch number  
        model_state: Model state dictionary
        optimizer_state: Optimizer state dictionary
        val_loss: Current validation loss value
        stage_name: Stage identifier for logging ("Stage 1" or "Stage 2")
        extra_data: Additional stage-specific data to include
        use_timestamped_name: Whether to generate timestamped filename (default: True)
        
    Returns:
        Actual path where checkpoint was saved
    """
    # Generate timestamped filename if requested
    if use_timestamped_name:
        # Extract stage identifier from stage_name for filename
        stage_name_lower = stage_name.lower()
        if "stage 1" in stage_name_lower:
            stage_id = "stage1"
        elif "stage 2" in stage_name_lower:
            if "baseline" in stage_name_lower:
                stage_id = "stage2_baseline"
            elif "enhanced" in stage_name_lower:
                stage_id = "stage2_enhanced"
            else:
                stage_id = "stage2"
        else:
            # Fallback - try to extract from the passed stage_name parameter
            stage_id = stage_name_lower.replace(" ", "").replace("(", "").replace(")", "")
        
        # Get base directory from original path
        base_dir = os.path.dirname(path) if path else "checkpoints"
        actual_path = generate_checkpoint_filename(stage_id, val_loss, base_dir)
    else:
        actual_path = path
    
    # Setup checkpoint directory
    dir_path = os.path.dirname(actual_path)
    if dir_path:  # Only create directory if it's not empty (i.e., not current directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"📁 Created checkpoint directory: {dir_path}")
    
    # Build checkpoint data with common structure
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'val_loss': val_loss,
    }
    
    # Add any extra stage-specific data
    if extra_data:
        checkpoint_data.update(extra_data)
    
    # Save checkpoint
    torch.save(checkpoint_data, actual_path)
    
    # Enhanced logging with full filename details
    filename = os.path.basename(actual_path)
    if use_timestamped_name:
        parsed = parse_checkpoint_filename(actual_path)
        if parsed:
            stage, timestamp, file_loss = parsed
            logger.info(f"💾 ✅ TIMESTAMPED CHECKPOINT SAVED | {stage_name}")
            logger.info(f"   📄 Filename: {filename}")
            logger.info(f"   📅 Timestamp: {timestamp}")
            logger.info(f"   📊 Epoch: {epoch+1} | Val Loss: {val_loss:.6f}")
        else:
            logger.info(f"💾 ✅ CHECKPOINT SAVED | {stage_name} | File: {filename} | Epoch: {epoch+1} | Val Loss: {val_loss:.6f}")
    else:
        logger.info(f"💾 ✅ CHECKPOINT SAVED | {stage_name} | File: {filename} | Epoch: {epoch+1} | Val Loss: {val_loss:.6f}")
    
    logger.debug(f"📊 Checkpoint data keys: {list(checkpoint_data.keys())}")
    
    return actual_path
