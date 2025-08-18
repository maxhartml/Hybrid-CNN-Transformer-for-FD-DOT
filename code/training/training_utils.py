"""
Minimal per-run overwrite checkpoint system.
One checkpoint file per training run, overwritten only on improvement.
"""

import os
import torch
from datetime import datetime
from code.utils.logging_config import get_training_logger

logger = get_training_logger(__name__)

# Module-level globals for run ID caching and best metric tracking
_run_ids = {}  # {stage_id: run_id}
_best_metrics = {}  # {(stage_id, run_id): best_val_std_rmse}

def get_or_create_run_id(stage_id: str) -> str:
    """Get cached run ID or create new timestamp-based run ID for stage."""
    if stage_id not in _run_ids:
        _run_ids[stage_id] = datetime.now().strftime('%Y%m%d-%H%M%S')
        logger.info(f"🆔 Created run_id for {stage_id}: {_run_ids[stage_id]}")
    return _run_ids[stage_id]

def get_checkpoint_path(stage_id: str, run_id: str) -> str:
    """Generate fixed checkpoint filename for this stage and run."""
    from .training_config import CHECKPOINT_BASE_DIR
    os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_BASE_DIR, f"checkpoint_{stage_id}_{run_id}.pt")

def save_checkpoint(checkpoint_path: str, checkpoint_data: dict, val_std_rmse: float) -> bool:
    """Save checkpoint only if val_std_rmse improved. Returns True if saved."""
    
    # Extract stage_id and run_id from filename
    filename = os.path.basename(checkpoint_path)
    parts = filename.replace('.pt', '').split('_')  # ['checkpoint', stage_id, run_id]
    stage_id, run_id = parts[1], parts[2]
    
    cache_key = (stage_id, run_id)
    current_best = _best_metrics.get(cache_key, float('inf'))
    
    if val_std_rmse < current_best:
        # Improvement detected - save checkpoint
        _best_metrics[cache_key] = val_std_rmse
        
        # Add stage and run metadata
        checkpoint_data['stage_id'] = stage_id
        checkpoint_data['run_id'] = run_id
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path} (val_std_rmse={val_std_rmse:.4f})")
        return True
    else:
        logger.debug(f"📊 No improvement: current={val_std_rmse:.4f} vs best={current_best:.4f}")
        return False

def find_best_checkpoint(stage_id: str, checkpoint_dir: str = None) -> str:
    """Find checkpoint with best val_std_rmse for given stage by reading metrics from files."""
    if checkpoint_dir is None:
        from .training_config import CHECKPOINT_BASE_DIR
        checkpoint_dir = CHECKPOINT_BASE_DIR
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    pattern = f"checkpoint_{stage_id}_"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(pattern) and f.endswith('.pt')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found for stage '{stage_id}' in {checkpoint_dir}")
    
    best_path = None
    best_metric = float('inf')
    
    for filename in checkpoint_files:
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            val_std_rmse = checkpoint['metrics']['val_std_rmse']
            
            if val_std_rmse < best_metric:
                best_metric = val_std_rmse
                best_path = filepath
                
        except Exception as e:
            logger.warning(f"⚠️  Skipping corrupted checkpoint {filename}: {e}")
            continue
    
    if best_path is None:
        raise RuntimeError(f"No valid checkpoints found for stage '{stage_id}'")
    
    logger.info(f"🏆 Best checkpoint for {stage_id}: {os.path.basename(best_path)} (val_std_rmse={best_metric:.4f})")
    return best_path
