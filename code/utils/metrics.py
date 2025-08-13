#!/usr/bin/env python3
"""
Enhanced Metrics for NIR-DOT Reconstruction Evaluation.

This module implements comprehensive metrics for evaluating NIR-DOT reconstruction
quality, including standard reconstruction metrics (SSIM, PSNR, RMSE) and 
advanced feature analysis metrics for transformer enhancement evaluation.

Metrics Categories:
1. Reconstruction Quality: SSIM, PSNR, channel-specific RMSE
2. Feature Analysis: Enhancement ratio, attention entropy (Stage 2 only)
3. W&B Integration: Automatic logging with proper formatting

Author: Max Hart
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import math
from typing import Dict, Optional, Tuple, Any, List

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np

# Project imports
from code.utils.logging_config import get_model_logger

# =============================================================================
# CONSTANTS
# =============================================================================

# SSIM Configuration
SSIM_WINDOW_SIZE = 11                   # Window size for SSIM calculation
SSIM_SIGMA = 1.5                        # Gaussian window standard deviation
SSIM_K1 = 0.01                          # SSIM stability constant 1
SSIM_K2 = 0.03                          # SSIM stability constant 2
SSIM_DATA_RANGE = 1.0                   # Assumed data range for SSIM

# PSNR Configuration
PSNR_DATA_RANGE = 1.0                   # Assumed data range for PSNR
PSNR_EPS = 1e-8                         # Small epsilon to prevent log(0)

# Feature Analysis Configuration
ENTROPY_EPS = 1e-12                     # Small epsilon for entropy calculation
ENHANCEMENT_RATIO_EPS = 1e-8            # Small epsilon for ratio calculation

# Channel Names for Logging
ABSORPTION_CHANNEL = "absorption"        # μₐ channel name
SCATTERING_CHANNEL = "scattering"        # μₛ channel name
CHANNEL_NAMES = [ABSORPTION_CHANNEL, SCATTERING_CHANNEL]

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class RMSELoss(nn.Module):
    """
    Root Mean Square Error loss function for volumetric reconstruction.
    
    This loss function computes the RMSE between predicted and target volumes,
    providing a measure of reconstruction accuracy that is sensitive to both
    small and large errors. RMSE is particularly suitable for volumetric
    reconstruction tasks where spatial accuracy is critical.
    
    The loss is computed as: sqrt(mean((pred - target)^2))
    
    Returns:
        torch.Tensor: Scalar RMSE loss value for optimization
    """
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE loss between input and target tensors.
        
        Args:
            input (torch.Tensor): Predicted volume reconstruction
            target (torch.Tensor): Ground truth volume data
            
        Returns:
            torch.Tensor: RMSE loss value
        """
        mse = F.mse_loss(input, target)
        return torch.sqrt(mse)


class ChannelWeightedRMSELoss(nn.Module):
    """
    Channel-weighted RMSE loss to balance absorption vs scattering learning.
    
    Since absorption typically converges faster than scattering in NIR-DOT,
    this loss applies higher weight to scattering channel to encourage
    balanced learning across both tissue properties.
    
    Weights: [1.0, 1.2] for [absorption, scattering] channels respectively
    """
    
    def __init__(self, channel_weights: List[float] = [1.0, 1.2]):
        """
        Initialize channel-weighted RMSE loss.
        
        Args:
            channel_weights: List of weights for [absorption, scattering] channels
        """
        super().__init__()
        self.channel_weights = torch.tensor(channel_weights)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute channel-weighted RMSE loss.
        
        Args:
            input (torch.Tensor): Predicted volume [batch, 2, D, H, W]
            target (torch.Tensor): Ground truth volume [batch, 2, D, H, W]
            
        Returns:
            torch.Tensor: Weighted RMSE loss value
        """
        self.channel_weights = self.channel_weights.to(input.device)
        
        # Compute per-channel MSE
        channel_mse = []
        for c in range(input.shape[1]):  # For each channel
            mse_c = F.mse_loss(input[:, c], target[:, c])
            weighted_mse_c = self.channel_weights[c] * mse_c
            channel_mse.append(weighted_mse_c)
        
        # Average across channels and take sqrt
        total_mse = torch.stack(channel_mse).mean()
        return torch.sqrt(total_mse)


# =============================================================================
# CORE RECONSTRUCTION METRICS
# =============================================================================

class SSIMMetric(nn.Module):
    """
    Structural Similarity Index Metric (SSIM) for 3D volumes.
    
    SSIM measures structural similarity between predicted and target volumes,
    considering luminance, contrast, and structure. Values range from -1 to 1,
    with 1 indicating perfect similarity.
    
    This implementation supports both 2D slices and 3D volumes with proper
    padding handling and numerical stability.
    """
    
    def __init__(self, window_size: int = SSIM_WINDOW_SIZE, 
                 sigma: float = SSIM_SIGMA, 
                 data_range: float = SSIM_DATA_RANGE,
                 k1: float = SSIM_K1, 
                 k2: float = SSIM_K2):
        super().__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        
        # Create Gaussian window
        self.register_buffer('window', self._create_gaussian_window())
        
        logger.debug(f"🔧 SSIM metric initialized: window_size={window_size}, sigma={sigma}")
    
    def _create_gaussian_window(self) -> torch.Tensor:
        """Create Gaussian window for SSIM calculation."""
        coords = torch.arange(self.window_size, dtype=torch.float32)
        coords -= self.window_size // 2
        
        # Create 1D Gaussian
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g /= g.sum()
        
        # Create 3D Gaussian window
        window_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        return window_3d.unsqueeze(0).unsqueeze(0)  # [1, 1, W, H, D]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate SSIM between predicted and target volumes.
        
        Args:
            pred (torch.Tensor): Predicted volume [B, C, D, H, W]
            target (torch.Tensor): Target volume [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Mean SSIM value across batch and channels
        """
        logger.debug(f"🏃 SSIM calculation: pred {pred.shape}, target {target.shape}")
        
        # Calculate SSIM per channel and average
        batch_size, num_channels = pred.shape[0], pred.shape[1]
        ssim_values = []
        
        for c in range(num_channels):
            pred_c = pred[:, c:c+1, ...]  # [B, 1, D, H, W]
            target_c = target[:, c:c+1, ...]  # [B, 1, D, H, W]
            
            # Ensure same device and dtype for mixed precision compatibility
            # Convert both to the same dtype (prefer float32 for stability)
            common_dtype = torch.float32
            pred_c = pred_c.to(dtype=common_dtype)
            target_c = target_c.to(dtype=common_dtype)
            window = self.window.to(pred_c.device, dtype=common_dtype)
            
            # Constants for SSIM
            c1 = (self.k1 * self.data_range) ** 2
            c2 = (self.k2 * self.data_range) ** 2
            
            # Calculate means
            mu1 = F.conv3d(pred_c, window, padding=self.window_size // 2, groups=1)
            mu2 = F.conv3d(target_c, window, padding=self.window_size // 2, groups=1)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            # Calculate variances and covariance
            sigma1_sq = F.conv3d(pred_c ** 2, window, padding=self.window_size // 2, groups=1) - mu1_sq
            sigma2_sq = F.conv3d(target_c ** 2, window, padding=self.window_size // 2, groups=1) - mu2_sq
            sigma12 = F.conv3d(pred_c * target_c, window, padding=self.window_size // 2, groups=1) - mu1_mu2
            
            # Calculate SSIM
            numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            
            ssim_map = numerator / (denominator + PSNR_EPS)
            ssim_values.append(ssim_map.mean())
        
        # Average across all channels
        mean_ssim = torch.stack(ssim_values).mean()
        logger.debug(f"📦 SSIM value: {mean_ssim.item():.6f}")
        return mean_ssim


class PSNRMetric(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) for 3D volumes.
    
    PSNR measures reconstruction quality in decibels (dB). Higher values
    indicate better reconstruction quality. Typical ranges for good
    reconstruction are 20-40 dB.
    """
    
    def __init__(self, data_range: float = PSNR_DATA_RANGE):
        super().__init__()
        self.data_range = data_range
        logger.debug(f"🔧 PSNR metric initialized: data_range={data_range}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate PSNR between predicted and target volumes.
        
        Args:
            pred (torch.Tensor): Predicted volume [B, C, D, H, W]
            target (torch.Tensor): Target volume [B, C, D, H, W]
            
        Returns:
            torch.Tensor: PSNR value in dB
        """
        logger.debug(f"🏃 PSNR calculation: pred {pred.shape}, target {target.shape}")
        
        # Calculate MSE
        mse = F.mse_loss(pred, target)
        
        # Prevent log(0) for perfect reconstruction
        mse = torch.clamp(mse, min=PSNR_EPS)
        
        # Calculate PSNR
        psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse))
        
        logger.debug(f"📦 PSNR value: {psnr.item():.2f} dB")
        return psnr


class ChannelSpecificRMSE(nn.Module):
    """
    Channel-specific Root Mean Square Error for absorption and scattering channels.
    
    Calculates RMSE separately for each tissue property channel (μₐ, μₛ)
    to provide detailed reconstruction analysis.
    """
    
    def __init__(self):
        super().__init__()
        logger.debug("🔧 Channel-specific RMSE metric initialized")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate channel-specific RMSE.
        
        Args:
            pred (torch.Tensor): Predicted volume [B, C, D, H, W] where C=2
            target (torch.Tensor): Target volume [B, C, D, H, W] where C=2
            
        Returns:
            Dict[str, torch.Tensor]: RMSE values for each channel
        """
        logger.debug(f"🏃 Channel RMSE calculation: pred {pred.shape}, target {target.shape}")
        
        rmse_values = {}
        
        for i, channel_name in enumerate(CHANNEL_NAMES):
            if i < pred.shape[1]:  # Ensure channel exists
                pred_channel = pred[:, i, ...]
                target_channel = target[:, i, ...]
                
                mse = F.mse_loss(pred_channel, target_channel)
                rmse = torch.sqrt(mse)
                
                rmse_values[f"rmse_{channel_name}"] = rmse
                logger.debug(f"📦 {channel_name} RMSE: {rmse.item():.6f}")
            
        return rmse_values


# =============================================================================
# FEATURE ANALYSIS METRICS (STAGE 2 ONLY)
# =============================================================================

class FeatureEnhancementRatio(nn.Module):
    """
    Feature Enhancement Ratio for evaluating transformer improvements.
    
    Measures how much the transformer enhances CNN features compared to
    the baseline CNN-only features. Higher ratios indicate more significant
    feature enhancement.
    
    Ratio = ||enhanced_features - cnn_features|| / ||cnn_features||
    """
    
    def __init__(self):
        super().__init__()
        logger.debug("🔧 Feature Enhancement Ratio metric initialized")
    
    def forward(self, enhanced_features: torch.Tensor, 
                cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate feature enhancement ratio.
        
        Args:
            enhanced_features (torch.Tensor): Transformer-enhanced features [B, D]
            cnn_features (torch.Tensor): Original CNN features [B, D] or [B, N, D] (multi-token)
            
        Returns:
            torch.Tensor: Enhancement ratio value
        """
        logger.debug(f"🏃 Enhancement ratio: enhanced {enhanced_features.shape}, cnn {cnn_features.shape}")
        
        # Handle multi-token CNN features by aggregating to single vector
        if len(cnn_features.shape) == 3:  # [B, N, D] -> [B, D]
            cnn_features_agg = cnn_features.mean(dim=1)
            logger.debug(f"📦 Aggregated multi-token CNN features: {cnn_features.shape} → {cnn_features_agg.shape}")
        else:  # [B, D]
            cnn_features_agg = cnn_features
        
        # Calculate feature differences
        feature_diff = enhanced_features - cnn_features_agg
        diff_norm = torch.norm(feature_diff, dim=-1)  # [B]
        
        # Calculate CNN feature magnitude
        cnn_norm = torch.norm(cnn_features_agg, dim=-1)  # [B]
        
        # Calculate ratio with numerical stability
        ratio = diff_norm / (cnn_norm + ENHANCEMENT_RATIO_EPS)
        mean_ratio = ratio.mean()
        
        logger.debug(f"📦 Enhancement ratio: {mean_ratio.item():.6f}")
        return mean_ratio


class AttentionEntropy(nn.Module):
    """
    Attention Entropy for measuring attention distribution diversity.
    
    Calculates the entropy of attention weights to measure how focused
    or distributed the attention is. Higher entropy indicates more
    distributed attention, lower entropy indicates more focused attention.
    
    Entropy = -sum(p * log(p)) where p are attention probabilities
    """
    
    def __init__(self):
        super().__init__()
        logger.debug("🔧 Attention Entropy metric initialized")
    
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention entropy.
        
        Args:
            attention_weights (torch.Tensor): Attention weights [B, L, H, S, S]
                where L=layers, H=heads, S=sequence_length
                
        Returns:
            torch.Tensor: Mean attention entropy across all dimensions
        """
        if attention_weights is None:
            logger.debug("⚠️ No attention weights provided, returning zero entropy")
            return torch.tensor(0.0, device='cpu')
        
        logger.debug(f"🏃 Attention entropy calculation: weights {attention_weights.shape}")
        
        # Attention weights are already probabilities from softmax in transformer
        attention_probs = attention_weights
        
        # Add small epsilon for numerical stability (avoid log(0))
        attention_probs = torch.clamp(attention_probs, min=ENTROPY_EPS, max=1.0)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(attention_probs * torch.log(attention_probs), dim=-1)
        
        # Average across all dimensions (batch, layers, heads, sequence)
        mean_entropy = entropy.mean()
        
        logger.debug(f"📦 Attention entropy: {mean_entropy.item():.6f}")
        return mean_entropy


# =============================================================================
# METRICS COLLECTION AND W&B INTEGRATION
# =============================================================================

class NIRDOTMetrics:
    """
    Comprehensive metrics collection for NIR-DOT reconstruction evaluation.
    
    This class coordinates all metrics calculation and provides W&B integration
    for automatic logging. Supports both Stage 1 (reconstruction only) and
    Stage 2 (reconstruction + feature analysis) metrics.
    """
    
    def __init__(self, stage: str = "stage1"):
        """
        Initialize metrics collection.
        
        Args:
            stage (str): Training stage ("stage1" or "stage2")
        """
        self.stage = stage
        
        # Initialize reconstruction metrics (both stages)
        self.ssim_metric = SSIMMetric()
        self.psnr_metric = PSNRMetric()
        self.rmse_metric = ChannelSpecificRMSE()
        
        # Initialize feature analysis metrics (Stage 2 only)
        if stage == "stage2":
            self.enhancement_ratio_metric = FeatureEnhancementRatio()
            self.attention_entropy_metric = AttentionEntropy()
        
        logger.info(f"📊 NIRDOTMetrics initialized for {stage}")
    
    def calculate_reconstruction_metrics(self, pred: torch.Tensor, 
                                       target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all reconstruction quality metrics.
        
        Args:
            pred (torch.Tensor): Predicted volume [B, C, D, H, W]
            target (torch.Tensor): Target volume [B, C, D, H, W]
            
        Returns:
            Dict[str, float]: Reconstruction metrics
        """
        metrics = {}
        
        # Calculate SSIM
        ssim_value = self.ssim_metric(pred, target)
        metrics['ssim'] = ssim_value.item()
        
        # Calculate PSNR
        psnr_value = self.psnr_metric(pred, target)
        metrics['psnr'] = psnr_value.item()
        
        # Calculate channel-specific RMSE
        rmse_values = self.rmse_metric(pred, target)
        for key, value in rmse_values.items():
            metrics[key] = value.item()
        
        # Calculate overall RMSE for comparison
        overall_rmse = torch.sqrt(F.mse_loss(pred, target))
        metrics['rmse_overall'] = overall_rmse.item()
        
        logger.debug(f"📊 Reconstruction metrics: {metrics}")
        return metrics
    
    def calculate_feature_metrics(self, enhanced_features: torch.Tensor,
                                cnn_features: torch.Tensor,
                                attention_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate feature analysis metrics (Stage 2 only).
        
        Args:
            enhanced_features (torch.Tensor): Transformer-enhanced features
            cnn_features (torch.Tensor): Original CNN features
            attention_weights (torch.Tensor, optional): Attention weights
            
        Returns:
            Dict[str, float]: Feature analysis metrics
        """
        if self.stage != "stage2":
            logger.warning("⚠️ Feature metrics only available for Stage 2")
            return {}
        
        metrics = {}
        
        # Calculate enhancement ratio
        enhancement_ratio = self.enhancement_ratio_metric(enhanced_features, cnn_features)
        metrics['feature_enhancement_ratio'] = enhancement_ratio.item()
        
        # Calculate attention entropy if available
        if attention_weights is not None:
            logger.debug(f"🎯 Calculating attention entropy with weights shape: {attention_weights.shape}")
            attention_entropy = self.attention_entropy_metric(attention_weights)
            metrics['attention_entropy'] = attention_entropy.item()
        else:
            logger.debug("⚠️ No attention weights available - setting entropy to 0.0")
            metrics['attention_entropy'] = 0.0
        
        logger.debug(f"📊 Feature metrics: {metrics}")
        return metrics
    
    def calculate_all_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                            enhanced_features: Optional[torch.Tensor] = None,
                            cnn_features: Optional[torch.Tensor] = None,
                            attention_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate all available metrics for the current stage.
        
        Args:
            pred (torch.Tensor): Predicted volume
            target (torch.Tensor): Target volume
            enhanced_features (torch.Tensor, optional): Enhanced features (Stage 2)
            cnn_features (torch.Tensor, optional): CNN features (Stage 2)
            attention_weights (torch.Tensor, optional): Attention weights (Stage 2)
            
        Returns:
            Dict[str, float]: Complete metrics dictionary
        """
        all_metrics = {}
        
        # Always calculate reconstruction metrics
        recon_metrics = self.calculate_reconstruction_metrics(pred, target)
        all_metrics.update(recon_metrics)
        
        # Calculate feature metrics for Stage 2
        if (self.stage == "stage2" and enhanced_features is not None 
            and cnn_features is not None):
            feature_metrics = self.calculate_feature_metrics(
                enhanced_features, cnn_features, attention_weights
            )
            all_metrics.update(feature_metrics)
        
        return all_metrics
    
    @staticmethod
    def log_to_wandb(metrics: Dict[str, float], epoch: int, 
                    mode: str = "train", use_wandb: bool = True):
        """
        Log metrics to Weights & Biases with proper formatting.
        
        Args:
            metrics (Dict[str, float]): Metrics to log
            epoch (int): Current epoch
            mode (str): Training mode ("train" or "val")
            use_wandb (bool): Whether to actually log to W&B
        """
        if not use_wandb:
            return
        
        try:
            import wandb
            
            # Format metrics with proper prefixes
            formatted_metrics = {}
            for key, value in metrics.items():
                formatted_key = f"{mode}/{key}"
                formatted_metrics[formatted_key] = value
            
            # Add epoch
            formatted_metrics["epoch"] = epoch
            
            # Log to W&B with step parameter to maintain consistency
            wandb.log(formatted_metrics, step=epoch)
            logger.debug(f"📤 Logged metrics to W&B: {mode} epoch {epoch}")
            
        except ImportError:
            logger.warning("⚠️ wandb not available, skipping W&B logging")
        except Exception as e:
            logger.error(f"❌ Error logging to W&B: {e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_metrics_for_stage(stage: str) -> NIRDOTMetrics:
    """
    Factory function to create appropriate metrics for training stage.
    
    Args:
        stage (str): Training stage ("stage1" or "stage2")
        
    Returns:
        NIRDOTMetrics: Configured metrics instance
    """
    return NIRDOTMetrics(stage=stage)


def calculate_batch_metrics(metrics: NIRDOTMetrics, outputs: Dict[str, torch.Tensor],
                          targets: torch.Tensor, stage: str) -> Dict[str, float]:
    """
    Calculate metrics for a single batch based on model outputs.
    
    Args:
        metrics (NIRDOTMetrics): Metrics calculator
        outputs (Dict[str, torch.Tensor]): Model outputs
        targets (torch.Tensor): Ground truth targets
        stage (str): Training stage
        
    Returns:
        Dict[str, float]: Batch metrics
    """
    pred = outputs['reconstructed']
    
    if stage == "stage1":
        return metrics.calculate_reconstruction_metrics(pred, targets)
    else:
        # Stage 2: include feature analysis if available
        enhanced_features = outputs.get('enhanced_features')
        cnn_features = outputs.get('cnn_features')
        attention_weights = outputs.get('attention_weights')
        
        return metrics.calculate_all_metrics(
            pred, targets, enhanced_features, cnn_features, attention_weights
        )


if __name__ == "__main__":
    # Test metrics with dummy data
    logger.info("🧪 Testing NIR-DOT metrics...")
    
    # Create test data
    batch_size, channels, depth, height, width = 2, 2, 32, 32, 32
    pred = torch.randn(batch_size, channels, depth, height, width)
    target = torch.randn(batch_size, channels, depth, height, width)
    
    # Test Stage 1 metrics
    stage1_metrics = NIRDOTMetrics("stage1")
    recon_metrics = stage1_metrics.calculate_reconstruction_metrics(pred, target)
    logger.info(f"📊 Stage 1 metrics: {recon_metrics}")
    
    # Test Stage 2 metrics
    stage2_metrics = NIRDOTMetrics("stage2")
    enhanced_features = torch.randn(batch_size, 256)
    cnn_features = torch.randn(batch_size, 256)
    attention_weights = torch.randn(batch_size, 4, 8, 2, 2)  # [B, L, H, S, S]
    
    all_metrics = stage2_metrics.calculate_all_metrics(
        pred, target, enhanced_features, cnn_features, attention_weights
    )
    logger.info(f"📊 Stage 2 metrics: {all_metrics}")
    
    logger.info("✅ Metrics testing completed successfully")
