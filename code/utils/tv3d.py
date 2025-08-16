import torch

def tv3d_l1(x: torch.Tensor) -> torch.Tensor:
    """
    3D Total Variation (L1 norm) for volumetric data.
    
    Args:
        x: Input tensor [B, C, D, H, W] (standardized space)
        
    Returns:
        torch.Tensor: Total variation loss (scalar)
    """
    # x: [B, C, D, H, W] (standardized space)
    dx = x[..., 1:, :, :] - x[..., :-1, :, :]
    dy = x[..., :, 1:, :] - x[..., :, :-1, :]
    dz = x[..., :, :, 1:] - x[..., :, :, :-1]
    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()) / 3.0
