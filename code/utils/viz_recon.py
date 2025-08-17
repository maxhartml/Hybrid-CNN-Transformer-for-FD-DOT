# code/utils/viz_recon.py
"""
Strict visualization system for NIR-DOT reconstruction logging to W&B.

This module provides a hardened, physics-aware visualization pipeline that:
- Ensures inputs are in raw physical units (mm^-1)
- Applies proper physics-based normalization
- Logs exactly 2 phantoms × 3 planes × 2 channels × (pred+target) = 24 images
- Removes all viz_stats noise and autocontrast complexity

Key Design Principles:
1. Raw physical units only - no standardized/normalized inputs
2. Automatic spatial axis detection to handle different tensor layouts
3. Physics-based color mapping with known tissue property ranges
4. Strict validation with guardrails against degenerate cases
"""
import torch, numpy as np, wandb
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)
# Flag to log warnings once per run to avoid spam
_shape_warning_logged = False

# Physical property ranges for tissue at NIR wavelengths (mm^-1)
PHYS_MAX = {
    "mu_a": 0.0245,  # Absorption coefficient maximum
    "mu_s": 2.95     # Reduced scattering coefficient maximum
}

def assert_raw_units(name: str, mu_a: torch.Tensor, mu_s: torch.Tensor) -> None:
    """
    Assert that both channels are in raw physical units (mm^-1).
    
    Args:
        name: Name for error messages
        mu_a: Absorption coefficient tensor
        mu_s: Reduced scattering coefficient tensor
        
    Raises:
        AssertionError: If values are not in expected physical ranges
    """
    eps = 1e-6
    
    # Check μₐ is in valid range [0, PHYS_MAX['mu_a']*1.1] with small tolerance
    mu_a_min, mu_a_max = mu_a.min().item(), mu_a.max().item()
    assert mu_a_min >= -eps, f"{name}: μₐ minimum {mu_a_min:.6f} < 0 - not raw mm^-1 units"
    assert mu_a_max <= PHYS_MAX["mu_a"] * 1.1, f"{name}: μₐ maximum {mu_a_max:.6f} > {PHYS_MAX['mu_a']*1.1:.4f} - not raw mm^-1 units"
    
    # Check μ′ₛ is in valid range [0, PHYS_MAX['mu_s']*1.1] with small tolerance  
    mu_s_min, mu_s_max = mu_s.min().item(), mu_s.max().item()
    assert mu_s_min >= -eps, f"{name}: μ′ₛ minimum {mu_s_min:.6f} < 0 - not raw mm^-1 units"
    assert mu_s_max <= PHYS_MAX["mu_s"] * 1.1, f"{name}: μ′ₛ maximum {mu_s_max:.6f} > {PHYS_MAX['mu_s']*1.1:.4f} - not raw mm^-1 units"

def _ensure_correct_shape(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure tensor is in [B,2,D,H,W] format, with warnings for wrong formats.
    
    Args:
        tensor: Input tensor that may be in wrong format
        name: Name for logging
        
    Returns:
        tensor: Tensor in [B,2,D,H,W] format
    """
    global _shape_warning_logged
    
    if tensor.ndim == 5 and tensor.shape[1] == 2:
        # Already in correct format [B,2,D,H,W]
        return tensor
    elif tensor.ndim == 5 and tensor.shape[-1] == 2:
        # Channels-last [B,D,H,W,2] - permute to channels-first
        if not _shape_warning_logged:
            logger.warning(f"⚠️ Detected channels-last format {name}: {tuple(tensor.shape)} - permuting to [B,2,D,H,W]")
            _shape_warning_logged = True
        return tensor.permute(0, 4, 1, 2, 3).contiguous()
    elif tensor.ndim == 4 and tensor.shape[0] == 2:
        # Missing batch dimension [2,D,H,W] - add batch dim
        if not _shape_warning_logged:
            logger.warning(f"⚠️ Detected missing batch dim {name}: {tuple(tensor.shape)} - adding batch dimension")
            _shape_warning_logged = True
        return tensor.unsqueeze(0)  # [2,D,H,W] -> [1,2,D,H,W]
    else:
        raise ValueError(f"Cannot convert {name} from {tuple(tensor.shape)} to [B,2,D,H,W] format")

def _check_tensor_similarity(pred: torch.Tensor, tgt: torch.Tensor) -> None:
    """
    Check if prediction and target tensors are suspiciously similar (same data).
    
    Args:
        pred: Prediction tensor
        tgt: Target tensor
        
    Raises:
        Warning if tensors appear to be identical
    """
    if torch.is_tensor(pred) and torch.is_tensor(tgt):
        if pred.data_ptr() == tgt.data_ptr():
            logger.warning("⚠️ Prediction and target tensors have same memory pointer - using identical data!")
        elif torch.allclose(pred, tgt, atol=1e-6):
            logger.warning("⚠️ Prediction and target tensors are nearly identical - check if using correct data!")

def _should_inverse_standardize(tensor: torch.Tensor) -> bool:
    """
    Heuristic to determine if tensor needs inverse standardization.
    
    If >20% of μₐ voxels exceed PHYS_MAX['mu_a']*1.5 OR any μ′ₛ > PHYS_MAX['mu_s']*1.5,
    treat as standardized and need inverse_transform. Otherwise assume already raw.
    
    Args:
        tensor: Input tensor [B,2,D,H,W]
        
    Returns:
        bool: True if should inverse standardize, False if already raw
    """
    mu_a = tensor[:, 0]  # Absorption channel
    mu_s = tensor[:, 1]  # Scattering channel
    
    # Check if values exceed physical bounds significantly
    mu_a_exceed_threshold = PHYS_MAX["mu_a"] * 1.5
    mu_s_exceed_threshold = PHYS_MAX["mu_s"] * 1.5
    
    # Count fraction of μₐ voxels that exceed threshold
    mu_a_exceed_fraction = (mu_a > mu_a_exceed_threshold).float().mean().item()
    
    # Check if any μ′ₛ exceeds threshold
    mu_s_max = mu_s.max().item()
    
    # If >20% of μₐ exceeds OR any μ′ₛ exceeds, assume standardized
    needs_inverse = mu_a_exceed_fraction > 0.2 or mu_s_max > mu_s_exceed_threshold
    
    if needs_inverse:
        logger.debug(f"Detected standardized data: μₐ exceed fraction={mu_a_exceed_fraction:.3f}, μ′ₛ max={mu_s_max:.3f}")
    else:
        logger.debug(f"Detected raw data: μₐ exceed fraction={mu_a_exceed_fraction:.3f}, μ′ₛ max={mu_s_max:.3f}")
    
    return needs_inverse
def _inv_std_chlast(x: torch.Tensor, stdzr) -> torch.Tensor:
    """
    Apply inverse standardization correctly - standardizer expects channel-first format.
    
    Args:
        x: Channel-first tensor [B, 2, D, H, W] (standardized)
        stdzr: Stage 1 ground_truth_standardizer trained on channel-first layout
        
    Returns:
        Tensor in raw mm^-1 units, still [B, 2, D, H, W] layout
    """
    assert x.ndim == 5 and x.shape[1] == 2, f"Expected [B,2,D,H,W], got {tuple(x.shape)}"
    dev, dtype = x.device, x.dtype
    
    # Apply inverse standardization directly (standardizer expects channel-first)
    inv = getattr(stdzr, "inverse_transform", None) or getattr(stdzr, "inverse_transform_ground_truth", None)
    assert inv is not None, "Standardizer missing inverse transform method"
    
    with torch.no_grad():
        x_raw = inv(x)                                  # Direct call, no permutation needed
    
    return x_raw.to(device=dev, dtype=dtype)


def _physics_norm(arr2d: np.ndarray, vmax: float) -> np.ndarray:
    """
    Convert a 2D array in physical units to 8-bit grayscale for visualization.
    Uses adaptive normalization based on actual data range for better contrast.
    
    Args:
        arr2d: 2D numpy array in physical units (mm^-1)
        vmax: Maximum physical value for normalization (unused, kept for compatibility)
        
    Returns:
        8-bit grayscale image array [0-255]
    """
    # Use adaptive normalization based on actual data range
    data_min = float(arr2d.min())
    data_max = float(arr2d.max())
    
    if data_max - data_min < 1e-8:
        # Handle constant arrays (avoid division by zero)
        return np.full_like(arr2d, 128, dtype=np.uint8)
    
    # Normalize to [0, 255] using actual data range
    normalized = ((arr2d - data_min) / (data_max - data_min)) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)

def _center_slices(vol_ch_first: np.ndarray):
    """
    Extract center slices from a 3D volume along all three orthogonal planes.
    
    Args:
        vol_ch_first: Volume array with shape [2, D, H, W] where:
                     - Channel 0: μₐ (absorption)
                     - Channel 1: μ′ₛ (reduced scattering) 
                     - D, H, W: Depth, Height, Width spatial dimensions
                     
    Returns:
        Dict with 6 2D slices: {xy,xz,yz} × {mu_a,mu_s}
        Each slice is taken from the center of the perpendicular axis
    """
    _, D, H, W = vol_ch_first.shape
    z, y, x = D // 2, H // 2, W // 2  # Center indices for each axis
    
    return {
        # XY plane (coronal) - slice through center depth
        "xy_mu_a": vol_ch_first[0, z, :, :],
        "xy_mu_s": vol_ch_first[1, z, :, :],
        # XZ plane (sagittal) - slice through center height  
        "xz_mu_a": vol_ch_first[0, :, y, :],
        "xz_mu_s": vol_ch_first[1, :, y, :],
        # YZ plane (axial) - slice through center width
        "yz_mu_a": vol_ch_first[0, :, :, x],
        "yz_mu_s": vol_ch_first[1, :, :, x],
    }

def prepare_raw_DHW(pred, tgt, standardizer=None):
    """
    Prepare prediction and target tensors for visualization by ensuring they are
    in raw physical units (mm^-1) with strict [B,2,D,H,W] layout.
    
    This function handles:
    1. Strict tensor layout enforcement with automatic format detection
    2. Intelligent inverse standardization based on value ranges
    3. Validation that data is in reasonable physical ranges
    4. Guardrails against degenerate cases (all zeros, non-finite values)
    
    Args:
        pred: Prediction tensor (may be standardized, various layouts possible)
        tgt: Target tensor (may be standardized, various layouts possible) 
        standardizer: Optional standardizer object with inverse_transform method
        
    Returns:
        Tuple of (pred_raw, tgt_raw) both as CPU float32 tensors with shape
        [B, 2, D, H, W] in raw physical units, clamped to tissue property ranges
        
    Raises:
        AssertionError: If inputs have wrong shapes, contain non-finite values,
                       or appear to be all zeros (indicating unit conversion issues)
    """
    # Step 1: Ensure correct tensor layout [B,2,D,H,W]
    pred = _ensure_correct_shape(pred, "pred")
    tgt = _ensure_correct_shape(tgt, "tgt")
    
    # Step 2: Check for suspicious similarity
    _check_tensor_similarity(pred, tgt)
    
    # Step 3: Intelligent inverse standardization (only if needed)
    if standardizer is not None:
        # Check if data appears to be standardized
        if _should_inverse_standardize(pred):
            logger.debug("Applying inverse standardization to pred")
            pred = _inv_std_chlast(pred, standardizer)
        if _should_inverse_standardize(tgt):
            logger.debug("Applying inverse standardization to tgt")
            tgt = _inv_std_chlast(tgt, standardizer)

    # Step 4: Move to CPU and ensure float32 for consistent processing
    pred = pred.float().cpu()
    tgt = tgt.float().cpu()

    # Step 5: Quick sanity checks (fail fast if values are degenerate)
    assert torch.isfinite(pred).all() and torch.isfinite(tgt).all(), \
        "Found NaN/Inf in volumes after inverse standardization"
    
    # Check that inverse standardization produced reasonable magnitudes
    assert pred[:,0].max() > 1e-4, \
        f"μₐ too small after processing: max={float(pred[:,0].max()):.2e} - check standardizer"
    assert pred[:,1].max() > 1e-2, \
        f"μ′ₛ too small after processing: max={float(pred[:,1].max()):.2e} - check standardizer"

    # Step 6: Validate raw physical units
    assert_raw_units("pred", pred[:,0], pred[:,1])
    assert_raw_units("tgt", tgt[:,0], tgt[:,1])

    # Step 7: Clamp to physically realistic tissue property ranges
    pred[:,0].clamp_(0.0, PHYS_MAX["mu_a"])  # μₐ: [0, 0.0245] mm^-1
    pred[:,1].clamp_(0.0, PHYS_MAX["mu_s"])  # μ′ₛ: [0, 2.95] mm^-1
    tgt[:,0].clamp_(0.0, PHYS_MAX["mu_a"])   
    tgt[:,1].clamp_(0.0, PHYS_MAX["mu_s"])   
    
    return pred, tgt

def log_recon_slices_raw(pred_raw: torch.Tensor,
                         tgt_raw: torch.Tensor,
                         epoch: int,
                         phantom_ids=None,
                         prefix: str = "Reconstructions",
                         teacher_raw: torch.Tensor = None):
    """
    Log reconstruction slices to W&B with strict format control.
    
    Creates exactly 24 images per call for pred+target, or 36 images if teacher included:
    - 2 phantoms × 3 orthogonal planes × 2 tissue properties × (prediction + target [+ teacher])
    - Physics-based color mapping using known tissue property ranges
    - Organized hierarchical naming for easy W&B navigation
    
    Args:
        pred_raw: Prediction tensor [B, 2, D, H, W] in raw physical units (mm^-1)
        tgt_raw: Target tensor [B, 2, D, H, W] in raw physical units (mm^-1)  
        epoch: Current training epoch (for logging context)
        phantom_ids: Optional list of phantom IDs for naming (uses indices if None)
        prefix: W&B log prefix for organization (default: "Reconstructions")
        teacher_raw: Optional teacher tensor [B, 2, D, H, W] in raw physical units (mm^-1)
        
    Note:
        Input tensors MUST already be in raw mm^-1 units and [B,2,D,H,W] layout.
        Use prepare_raw_DHW() first if needed for preprocessing.
    """
    # Validate input tensor shapes and consistency
    assert pred_raw.ndim == 5 and tgt_raw.ndim == 5 and pred_raw.shape == tgt_raw.shape, \
        f"Tensor shape mismatch: pred={pred_raw.shape}, tgt={tgt_raw.shape}"
    
    if teacher_raw is not None:
        assert teacher_raw.ndim == 5 and teacher_raw.shape == pred_raw.shape, \
            f"Teacher shape mismatch: teacher={teacher_raw.shape}, expected={pred_raw.shape}"
    
    B = pred_raw.shape[0]
    n = min(2, B)  # Log exactly 2 phantoms (or fewer if batch smaller)

    # Debug print to check value ranges (keep minimal output)
    print(f"viz ranges: μₐ {float(pred_raw[:,0].min()):.4f}-{float(pred_raw[:,0].max()):.4f}, "
          f"μ′ₛ {float(pred_raw[:,1].min()):.4f}-{float(pred_raw[:,1].max()):.4f}")
    
    # Abort if any channel appears to be all zeros (indicates preprocessing bug)
    if pred_raw[:,0].max() < 1e-6 or pred_raw[:,1].max() < 1e-4:
        raise ValueError(f"Prediction values too small - preprocessing failed. "
                        f"μₐ max: {float(pred_raw[:,0].max()):.2e}, "
                        f"μ′ₛ max: {float(pred_raw[:,1].max()):.2e}")

    logs = {}  # Collect all images for single W&B commit
    
    for i in range(n):
        # Generate phantom identifier for logging
        pid = int(phantom_ids[i]) if (phantom_ids is not None and i < len(phantom_ids)) else i
        tag = f"{prefix}/Phantom_{pid:03d}"

        # Convert to numpy for slice extraction [2, D, H, W]
        p = pred_raw[i].numpy()  
        t = tgt_raw[i].numpy()
        teach = teacher_raw[i].numpy() if teacher_raw is not None else None

        # Extract center slices from volumes
        ps = _center_slices(p)  # Prediction slices
        ts = _center_slices(t)  # Target slices
        teach_s = _center_slices(teach) if teach is not None else None  # Teacher slices

        # Log images with target-prediction-teacher alignment
        # Layout: target on top, prediction in middle, teacher on bottom (if present)
        for plane in ("xy", "xz", "yz"):
            for channel, ch_name in [("mu_a", "μₐ"), ("mu_s", "μ′ₛ")]:
                # Target on top
                logs[f"{tag}/tgt_{plane}_{channel}"] = wandb.Image(
                    _physics_norm(ts[f"{plane}_{channel}"], PHYS_MAX[channel])
                )
                # Student prediction in middle
                logs[f"{tag}/student_{plane}_{channel}"] = wandb.Image(
                    _physics_norm(ps[f"{plane}_{channel}"], PHYS_MAX[channel])
                )
                # Teacher reconstruction on bottom (if available)
                if teach_s is not None:
                    logs[f"{tag}/teacher_{plane}_{channel}"] = wandb.Image(
                        _physics_norm(teach_s[f"{plane}_{channel}"], PHYS_MAX[channel])
                    )

    # Single atomic commit to W&B (prevents partial uploads)
    wandb.log(logs)
