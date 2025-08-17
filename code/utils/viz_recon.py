# code/utils/viz_recon.py
"""
RAW-only visualization system for NIR-DOT reconstruction logging to W&B.

This module provides a hardened, physics-aware visualization pipeline that:
- All functions expect RAW mm⁻¹, channel-first [B,2,D,H,W]. No standardized inputs here.
- Applies proper physics-based normalization only
- Logs exactly 2 phantoms × 3 planes × 2 channels × (pred+target) = 24 images
- Removes all standardizer dependencies and legacy conversion paths

Key Design Principles:
1. Raw physical units only - no standardized/normalized inputs accepted
2. Channel-first tensor layout [B, 2, D, H, W] assumed throughout
3. Physics-based color mapping using fixed tissue property ranges
4. Strict validation with guardrails against degenerate cases
5. VALIDATION-ONLY usage - never called during training loops
"""
import torch, numpy as np, wandb

# Physical property ranges for tissue at NIR wavelengths (mm^-1)
PHYS_MAX = {
    "mu_a": 0.0245,  # Absorption coefficient maximum
    "mu_s": 2.95     # Reduced scattering coefficient maximum
}

def _physics_norm(arr2d: np.ndarray, vmax: float, adaptive: bool = False) -> np.ndarray:
    """
    Convert a 2D array in physical units to 8-bit grayscale for visualization.
    
    Args:
        arr2d: 2D numpy array in physical units (mm⁻¹)
        vmax: Maximum physical value for normalization
        adaptive: If True, use data range; if False, use physics-based range
        
    Returns:
        8-bit grayscale image array [0-255]
    """
    if adaptive:
        a0, a1 = float(arr2d.min()), float(arr2d.max())
        if a1 - a0 < 1e-8:
            return np.full_like(arr2d, 128, dtype=np.uint8)
        scaled = (arr2d - a0) / (a1 - a0)
    else:
        clipped = np.clip(arr2d, 0.0, vmax)
        scaled = clipped / (vmax + 1e-12)
    
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

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

def prepare_raw_DHW(pred_raw: torch.Tensor, tgt_raw: torch.Tensor):
    """
    Validate that both tensors are RAW mm⁻¹ in [B,2,D,H,W], channel-first.
    Returns CPU float32 copies for imaging.
    """
    assert pred_raw.ndim == 5 and tgt_raw.ndim == 5 and pred_raw.shape[1] == 2 and tgt_raw.shape[1] == 2, \
        f"Expected [B,2,D,H,W]; got pred={tuple(pred_raw.shape)}, tgt={tuple(tgt_raw.shape)}"
    pred_raw = pred_raw.float().cpu()
    tgt_raw  = tgt_raw.float().cpu()
    assert torch.isfinite(pred_raw).all() and torch.isfinite(tgt_raw).all(), "NaN/Inf in raw volumes"
    # Loose physical sanity (allow slight slack)
    assert float(pred_raw[:,0].max()) <= PHYS_MAX["mu_a"]*1.05 + 1e-6 and float(pred_raw[:,1].max()) <= PHYS_MAX["mu_s"]*1.05 + 1e-6, "pred out of phys range"
    assert float(tgt_raw[:,0].max())  <= PHYS_MAX["mu_a"]*1.05 + 1e-6 and float(tgt_raw[:,1].max())  <= PHYS_MAX["mu_s"]*1.05 + 1e-6, "tgt out of phys range"
    return pred_raw, tgt_raw

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
                # Target on top - uses physics-based normalization (adaptive=False)
                logs[f"{tag}/tgt_{plane}_{channel}"] = wandb.Image(
                    _physics_norm(ts[f"{plane}_{channel}"], PHYS_MAX[channel], adaptive=False)
                )
                # Prediction directly below target - uses physics-based normalization (adaptive=False)
                logs[f"{tag}/pred_{plane}_{channel}"] = wandb.Image(
                    _physics_norm(ps[f"{plane}_{channel}"], PHYS_MAX[channel], adaptive=False)
                )

    # Single atomic commit to W&B (prevents partial uploads)
    wandb.log(logs)
