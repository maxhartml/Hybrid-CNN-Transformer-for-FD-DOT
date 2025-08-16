"""
Latent Operations and LatentAdapter for NIR-DOT Teacher-Student Training

This module provides utilities for computing and tracking statistics
of latent representations during Stage 2 training, including RMSE,
cosine similarity, and distribution metrics between teacher and student.

Key Components:
- LatentAdapter: Small near-identity MLP + LayerNorm for student latent alignment
- Composite latent loss functions for teacher-student matching
- Statistics tracking for monitoring training progress
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

# =============================================================================
# LATENT ADAPTER FOR TEACHER-STUDENT ALIGNMENT
# =============================================================================

class LatentAdapter(nn.Module):
    """
    Small near-identity MLP + LayerNorm for aligning student latent scale/direction to teacher.
    
    This adapter fine-tunes the student latent representations to better match the
    teacher's latent distribution without changing the core learned features.
    
    Args:
        d: Latent dimension (default 256)
        hidden: Hidden layer dimension
        init_gamma: Initial scaling factor for residual connection
    """
    
    def __init__(self, d: int = 256, hidden: int = 512, init_gamma: float = 0.1):
        super().__init__()
        
        # Layer normalization for input
        self.ln1 = nn.LayerNorm(d)
        
        # Small MLP transformation
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d)
        
        # Output layer normalization
        self.ln2 = nn.LayerNorm(d)
        
        # Learnable scaling factor for residual connection
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        
        # Initialize weights for near-identity behavior
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)  # Start with zero output for identity
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            z: Student latent representation [batch_size, latent_dim]
            
        Returns:
            z_adapted: Adapted latent representation [batch_size, latent_dim]
        """
        # Compute small perturbation
        y = self.fc2(self.act(self.fc1(self.ln1(z))))
        
        # Apply residual connection with learnable scaling
        z_adapted = self.ln2(z + self.gamma * y)
        
        return z_adapted

# =============================================================================
# LATENT STATISTICS TRACKER
# =============================================================================


class LatentStats:
    """
    Compute and track latent space statistics for teacher-student training.
    
    Tracks metrics like RMSE, cosine similarity, magnitude differences,
    and distribution statistics between teacher and student latent representations.
    """
    
    def __init__(self):
        """Initialize latent statistics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.stats = {
            'rmse_sum': 0.0,
            'cosine_sim_sum': 0.0,
            'teacher_mag_sum': 0.0,
            'student_mag_sum': 0.0,
            'mag_diff_sum': 0.0,
            'count': 0
        }
    
    def update(self, teacher_latent: torch.Tensor, student_latent: torch.Tensor) -> Dict[str, float]:
        """
        Update statistics with a new batch of teacher-student latent pairs.
        
        Args:
            teacher_latent: Teacher latent representations [batch_size, latent_dim]
            student_latent: Student latent representations [batch_size, latent_dim]
            
        Returns:
            batch_stats: Dictionary of statistics for this batch
        """
        batch_size = teacher_latent.shape[0]
        
        # Compute RMSE
        rmse = torch.sqrt(F.mse_loss(student_latent, teacher_latent))
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(teacher_latent, student_latent, dim=1).mean()
        
        # Compute magnitudes
        teacher_mag = torch.norm(teacher_latent, dim=1).mean()
        student_mag = torch.norm(student_latent, dim=1).mean()
        mag_diff = torch.abs(teacher_mag - student_mag)
        
        # Update accumulated stats
        self.stats['rmse_sum'] += rmse.item() * batch_size
        self.stats['cosine_sim_sum'] += cosine_sim.item() * batch_size
        self.stats['teacher_mag_sum'] += teacher_mag.item() * batch_size
        self.stats['student_mag_sum'] += student_mag.item() * batch_size
        self.stats['mag_diff_sum'] += mag_diff.item() * batch_size
        self.stats['count'] += batch_size
        
        # Return batch statistics
        batch_stats = {
            'latent_rmse': rmse.item(),
            'latent_cosine_sim': cosine_sim.item(),
            'teacher_magnitude': teacher_mag.item(),
            'student_magnitude': student_mag.item(),
            'magnitude_diff': mag_diff.item()
        }
        
        return batch_stats
    
    def compute_epoch_stats(self) -> Dict[str, float]:
        """
        Compute epoch-level statistics from accumulated values.
        
        Returns:
            epoch_stats: Dictionary of epoch-averaged statistics
        """
        if self.stats['count'] == 0:
            return {}
        
        count = self.stats['count']
        epoch_stats = {
            'latent_rmse': self.stats['rmse_sum'] / count,
            'latent_cosine_sim': self.stats['cosine_sim_sum'] / count,
            'teacher_magnitude': self.stats['teacher_mag_sum'] / count,
            'student_magnitude': self.stats['student_mag_sum'] / count,
            'magnitude_diff': self.stats['mag_diff_sum'] / count
        }
        
        return epoch_stats
    
    def compute_detailed_stats(self, teacher_latent: torch.Tensor, 
                             student_latent: torch.Tensor) -> Dict[str, float]:
        """
        Compute detailed statistics for a batch of latent representations.
        
        Args:
            teacher_latent: Teacher latent representations [batch_size, latent_dim]
            student_latent: Student latent representations [batch_size, latent_dim]
            
        Returns:
            detailed_stats: Extended statistics including percentiles and distributions
        """
        with torch.no_grad():
            # Basic stats
            basic_stats = self.update(teacher_latent, student_latent)
            
            # Per-sample RMSE distribution
            per_sample_rmse = torch.sqrt(F.mse_loss(student_latent, teacher_latent, reduction='none').mean(dim=1))
            
            # Per-sample cosine similarity
            per_sample_cosine = F.cosine_similarity(teacher_latent, student_latent, dim=1)
            
            # Distribution statistics - handle large tensors for quantile computation
            try:
                # Sample down if tensor is too large for quantile computation
                if per_sample_rmse.numel() > 10000:  # Limit to avoid "tensor too large" error
                    indices = torch.randperm(per_sample_rmse.numel())[:5000]
                    rmse_sample = per_sample_rmse[indices]
                else:
                    rmse_sample = per_sample_rmse
                    
                detailed_stats = basic_stats.copy()
                detailed_stats.update({
                    'latent_rmse_std': per_sample_rmse.std().item(),
                    'latent_rmse_min': per_sample_rmse.min().item(),
                    'latent_rmse_max': per_sample_rmse.max().item(),
                    'latent_rmse_p25': rmse_sample.quantile(0.25).item(),
                    'latent_rmse_p75': rmse_sample.quantile(0.75).item(),
                    'cosine_sim_std': per_sample_cosine.std().item(),
                    'cosine_sim_min': per_sample_cosine.min().item(),
                    'cosine_sim_max': per_sample_cosine.max().item(),
                })
            except RuntimeError as e:
                # Fallback if quantile still fails
                detailed_stats = basic_stats.copy()
                detailed_stats.update({
                    'latent_rmse_std': per_sample_rmse.std().item(),
                    'latent_rmse_min': per_sample_rmse.min().item(),
                    'latent_rmse_max': per_sample_rmse.max().item(),
                    'cosine_sim_std': per_sample_cosine.std().item(),
                    'cosine_sim_min': per_sample_cosine.min().item(),
                    'cosine_sim_max': per_sample_cosine.max().item(),
                })
            
            return detailed_stats


def compute_latent_rmse(teacher_latent: torch.Tensor, student_latent: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSE loss between teacher and student latent representations.
    
    This is the primary loss function for latent-only training.
    
    Args:
        teacher_latent: Teacher latent representations [batch_size, latent_dim]
        student_latent: Student latent representations [batch_size, latent_dim]
        
    Returns:
        rmse_loss: Root mean squared error between latents
    """
    return torch.sqrt(F.mse_loss(student_latent, teacher_latent))


def compute_latent_cosine_similarity(teacher_latent: torch.Tensor, 
                                   student_latent: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between teacher and student latent representations.
    
    Args:
        teacher_latent: Teacher latent representations [batch_size, latent_dim]
        student_latent: Student latent representations [batch_size, latent_dim]
        
    Returns:
        cosine_sim: Mean cosine similarity across the batch
    """
    return F.cosine_similarity(teacher_latent, student_latent, dim=1).mean()


# =============================================================================
# COMPOSITE LATENT LOSS FUNCTIONS
# =============================================================================

def latent_rmse(z_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSE between student and teacher latents.
    
    Args:
        z_hat: Student latent representations [batch_size, latent_dim]
        z_t: Teacher latent representations [batch_size, latent_dim]
        
    Returns:
        rmse: Root mean squared error
    """
    return torch.sqrt(torch.mean((z_hat - z_t)**2))


def latent_cosine_sim(z_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between student and teacher latents.
    
    Args:
        z_hat: Student latent representations [batch_size, latent_dim]
        z_t: Teacher latent representations [batch_size, latent_dim]
        
    Returns:
        cosine_sim: Mean cosine similarity across batch
    """
    return torch.nn.functional.cosine_similarity(z_hat, z_t, dim=1).mean()


def latent_mag_loss(z_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """
    Compute magnitude difference loss between student and teacher latents.
    
    Args:
        z_hat: Student latent representations [batch_size, latent_dim]
        z_t: Teacher latent representations [batch_size, latent_dim]
        
    Returns:
        mag_loss: Mean squared difference in magnitude
    """
    return ((z_hat.norm(dim=1) - z_t.norm(dim=1))**2).mean()


def composite_latent_loss(z_hat: torch.Tensor, z_t: torch.Tensor, 
                         w_cos: float = 0.5, w_mag: float = 0.05) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute composite latent loss combining RMSE, cosine similarity, and magnitude alignment.
    
    Args:
        z_hat: Student latent representations [batch_size, latent_dim]
        z_t: Teacher latent representations [batch_size, latent_dim]
        w_cos: Weight for cosine similarity loss
        w_mag: Weight for magnitude loss
        
    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary with individual loss components
    """
    # Compute individual loss components
    cos = latent_cosine_sim(z_hat, z_t)
    L_rmse = latent_rmse(z_hat, z_t)
    L_cos = 1.0 - cos  # Convert similarity to loss
    L_mag = latent_mag_loss(z_hat, z_t)
    
    # Compute total weighted loss
    total_loss = L_rmse + w_cos * L_cos + w_mag * L_mag
    
    # Return both total loss and individual components for logging
    loss_dict = {
        'latent_rmse': L_rmse,
        'latent_cosine_sim': cos,  # Keep as similarity for monitoring
        'latent_mag': L_mag
    }
    
    return total_loss, loss_dict


def analyze_latent_distribution(latent: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    """
    Analyze the distribution properties of latent representations.
    
    Args:
        latent: Latent representations [batch_size, latent_dim]
        prefix: Prefix for metric names
        
    Returns:
        distribution_stats: Dictionary of distribution statistics
    """
    with torch.no_grad():
        # Flatten for distribution analysis
        flat_latent = latent.view(-1)
        
        stats = {
            f'{prefix}mean': flat_latent.mean().item(),
            f'{prefix}std': flat_latent.std().item(),
            f'{prefix}min': flat_latent.min().item(),
            f'{prefix}max': flat_latent.max().item(),
            f'{prefix}magnitude_mean': torch.norm(latent, dim=1).mean().item(),
            f'{prefix}magnitude_std': torch.norm(latent, dim=1).std().item(),
        }
        
        # Add percentiles - handle large tensors 
        try:
            # Sample down if tensor is too large for quantile computation
            if flat_latent.numel() > 10000:  # Limit to avoid "tensor too large" error
                indices = torch.randperm(flat_latent.numel())[:5000]
                latent_sample = flat_latent[indices]
            else:
                latent_sample = flat_latent
                
            stats.update({
                f'{prefix}p25': latent_sample.quantile(0.25).item(),
                f'{prefix}p50': latent_sample.quantile(0.50).item(),
                f'{prefix}p75': latent_sample.quantile(0.75).item(),
                f'{prefix}p95': latent_sample.quantile(0.95).item(),
            })
        except RuntimeError:
            # Skip percentiles if quantile operation fails
            pass
        
        return stats


if __name__ == "__main__":
    # Test latent statistics and adapter
    print("Testing latent operations...")
    
    # Create dummy teacher and student latents
    batch_size, latent_dim = 32, 256
    teacher_latent = torch.randn(batch_size, latent_dim)
    student_latent = teacher_latent + 0.1 * torch.randn(batch_size, latent_dim)  # Similar but not identical
    
    # Test LatentAdapter
    adapter = LatentAdapter(d=latent_dim)
    adapted_student = adapter(student_latent)
    print(f"✓ LatentAdapter: input {student_latent.shape} -> output {adapted_student.shape}")
    
    # Test LatentStats class
    stats_tracker = LatentStats()
    batch_stats = stats_tracker.update(teacher_latent, student_latent)
    epoch_stats = stats_tracker.compute_epoch_stats()
    
    print("✓ Batch stats:", {k: f"{v:.4f}" for k, v in batch_stats.items()})
    print("✓ Epoch stats:", {k: f"{v:.4f}" for k, v in epoch_stats.items()})
    
    # Test detailed stats
    detailed_stats = stats_tracker.compute_detailed_stats(teacher_latent, student_latent)
    print("✓ Detailed stats keys:", list(detailed_stats.keys()))
    
    # Test individual functions
    rmse = compute_latent_rmse(teacher_latent, student_latent)
    cosine_sim = compute_latent_cosine_similarity(teacher_latent, student_latent)
    print(f"✓ RMSE: {rmse:.4f}, Cosine sim: {cosine_sim:.4f}")
    
    # Test composite loss functions
    total_loss, loss_dict = composite_latent_loss(adapted_student, teacher_latent)
    print(f"✓ Composite loss: {total_loss:.4f}")
    print("✓ Loss components:", {k: f"{v:.4f}" for k, v in loss_dict.items()})
    
    # Test distribution analysis
    teacher_dist = analyze_latent_distribution(teacher_latent, "teacher_")
    student_dist = analyze_latent_distribution(student_latent, "student_")
    print("✓ Teacher distribution keys:", list(teacher_dist.keys()))
    print("✓ Student distribution keys:", list(student_dist.keys()))
    
    print("✓ All latent operations tests passed!")
