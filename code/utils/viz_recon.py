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

# Physical property ranges for tissue at NIR wavelengths (mm^-1)
PHYS_MAX = {
    "mu_a": 0.0245,  # Absorption coefficient maximum
    "mu_s": 2.95     # Reduced scattering coefficient maximum
}

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
    1. Strict tensor layout enforcement (no permutation guessing)
    2. Correct inverse standardization with channel axis handling
    3. Validation that data is in reasonable physical ranges
    4. Guardrails against degenerate cases (all zeros, non-finite values)
    
    Args:
        pred: Prediction tensor [B, 2, D, H, W] (may be standardized)
        tgt: Target tensor [B, 2, D, H, W] (may be standardized) 
        standardizer: Optional standardizer object with inverse_transform method
        
    Returns:
        Tuple of (pred_raw, tgt_raw) both as CPU float32 tensors with shape
        [B, 2, D, H, W] in raw physical units, clamped to tissue property ranges
        
    Raises:
        AssertionError: If inputs have wrong shapes, contain non-finite values,
                       or appear to be all zeros (indicating unit conversion issues)
    """
    # Require strict channel-first input layout
    assert pred.ndim == 5 and tgt.ndim == 5 and pred.shape[1] == 2 and tgt.shape[1] == 2, \
        f"Expected [B,2,D,H,W]; got pred={tuple(pred.shape)}, tgt={tuple(tgt.shape)}"

    # Apply inverse standardization with correct channel axis handling
    if standardizer is not None:
        pred = _inv_std_chlast(pred, standardizer)     # Correct axis handling
        tgt  = _inv_std_chlast(tgt, standardizer)

    # Move to CPU and ensure float32 for consistent processing
    pred = pred.float().cpu()
    tgt  = tgt.float().cpu()

    # Quick sanity checks (fail fast if values are degenerate)
    assert torch.isfinite(pred).all() and torch.isfinite(tgt).all(), \
        "Found NaN/Inf in volumes after inverse standardization"
    
    # Check that inverse standardization produced reasonable magnitudes
    assert pred[:,0].max() > 1e-4, \
        f"μₐ too small after inverse standardization: max={float(pred[:,0].max()):.2e} - check standardizer"
    assert pred[:,1].max() > 1e-2, \
        f"μ′ₛ too small after inverse standardization: max={float(pred[:,1].max()):.2e} - check standardizer"

    # Clamp to physically realistic tissue property ranges
    pred[:,0].clamp_(0.0, PHYS_MAX["mu_a"])  # μₐ: [0, 0.0245] mm^-1
    pred[:,1].clamp_(0.0, PHYS_MAX["mu_s"])  # μ′ₛ: [0, 2.95] mm^-1
    tgt[:,0].clamp_(0.0, PHYS_MAX["mu_a"])   
    tgt[:,1].clamp_(0.0, PHYS_MAX["mu_s"])   
    
    return pred, tgt

def log_recon_slices_raw(pred_raw: torch.Tensor,
                         tgt_raw: torch.Tensor,
                         epoch: int,
                         phantom_ids=None,
                         prefix: str = "Reconstructions"):
    """
    Log reconstruction slices to W&B with strict format control.
    
    Creates exactly 24 images per call:
    - 2 phantoms × 3 orthogonal planes × 2 tissue properties × (prediction + target)
    - Physics-based color mapping using known tissue property ranges
    - Organized hierarchical naming for easy W&B navigation
    
    Args:
        pred_raw: Prediction tensor [B, 2, D, H, W] in raw physical units (mm^-1)
        tgt_raw: Target tensor [B, 2, D, H, W] in raw physical units (mm^-1)  
        epoch: Current training epoch (for logging context)
        phantom_ids: Optional list of phantom IDs for naming (uses indices if None)
        prefix: W&B log prefix for organization (default: "Reconstructions")
        
    Note:
        Input tensors MUST already be in raw mm^-1 units and [B,2,D,H,W] layout.
        Use prepare_raw_DHW() first if needed for preprocessing.
    """
    # Validate input tensor shapes and consistency
    assert pred_raw.ndim == 5 and tgt_raw.ndim == 5 and pred_raw.shape == tgt_raw.shape, \
        f"Tensor shape mismatch: pred={pred_raw.shape}, tgt={tgt_raw.shape}"
    
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

        # Extract center slices from both prediction and target volumes
        ps = _center_slices(p)  # Prediction slices
        ts = _center_slices(t)  # Target slices

        # Log images with target-prediction pairs vertically aligned
        # Layout: target on top, prediction directly below for easy comparison
        for plane in ("xy", "xz", "yz"):
            for channel, ch_name in [("mu_a", "μₐ"), ("mu_s", "μ′ₛ")]:
                # Target on top
                logs[f"{tag}/tgt_{plane}_{channel}"] = wandb.Image(
                    _physics_norm(ts[f"{plane}_{channel}"], PHYS_MAX[channel])
                )
                # Prediction directly below target
                logs[f"{tag}/pred_{plane}_{channel}"] = wandb.Image(
                    _physics_norm(ps[f"{plane}_{channel}"], PHYS_MAX[channel])
                )

    # Single atomic commit to W&B (prevents partial uploads)
    wandb.log(logs)
