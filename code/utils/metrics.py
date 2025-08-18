#!/usr/bin/env python3
"""
Enhanced Metrics for NIR-DOT Reconstruction Evaluation.

This module implements comprehensive metrics for evaluating NIR-DOT reconstruction
quality, including standard reconstruction metrics (Dice Coefficient, Contrast Ratio, RMSE) and 
advanced feature analysis metrics for transformer enhancement evaluation.

Metrics Categories:
1. Reconstruction Quality: Dice Coefficient, Contrast Ratio, channel-specific RMSE
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

# Dice Coefficient Configuration
DICE_SMOOTH = 1e-6                      # Smoothing factor for numerical stability
DICE_THRESHOLD = 0.5                    # Threshold for binary mask creation

# Contrast Ratio Configuration  
CONTRAST_EPS = 1e-8                     # Small epsilon to prevent division by zero

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


# =============================================================================
# CORE RECONSTRUCTION METRICS
# =============================================================================

class DiceCoefficient(nn.Module):
    """
    Sørensen–Dice Coefficient (SDC) for 3D volumes.
    
    Dice coefficient measures spatial similarity between predicted and target volumes
    by comparing overlap between binary masks. Values range from 0 to 1,
    with 1 indicating perfect spatial overlap.
    
    Formula: SDC = 2 * |M̂ ∩ M| / (|M̂| + |M|)
    where M̂ is the predicted anomaly mask and M is the ground-truth anomaly mask.
    """
    
    def __init__(self, threshold: float = DICE_THRESHOLD, smooth: float = DICE_SMOOTH):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth
        logger.debug(f"🔧 Dice coefficient initialized: threshold={threshold}, smooth={smooth}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice coefficient between predicted and target volumes.
        
        Args:
            pred (torch.Tensor): Predicted volume [B, C, D, H, W]
            target (torch.Tensor): Target volume [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Mean Dice coefficient across batch and channels
        """
        logger.debug(f"🏃 Dice calculation: pred {pred.shape}, target {target.shape}")
        
        batch_size, num_channels = pred.shape[0], pred.shape[1]
        dice_values = []
        
        for c in range(num_channels):
            pred_c = pred[:, c, ...]  # [B, D, H, W]
            target_c = target[:, c, ...]  # [B, D, H, W]
            
            # Normalize to [0, 1] range using min-max normalization
            pred_norm = (pred_c - pred_c.min()) / (pred_c.max() - pred_c.min() + CONTRAST_EPS)
            target_norm = (target_c - target_c.min()) / (target_c.max() - target_c.min() + CONTRAST_EPS)
            
            # Create binary masks using threshold
            pred_mask = (pred_norm > self.threshold).float()
            target_mask = (target_norm > self.threshold).float()
            
            # Calculate intersection and union
            intersection = (pred_mask * target_mask).sum(dim=[1, 2, 3])  # [B]
            pred_sum = pred_mask.sum(dim=[1, 2, 3])  # [B]
            target_sum = target_mask.sum(dim=[1, 2, 3])  # [B]
            
            # Calculate Dice coefficient
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_values.append(dice.mean())  # Average across batch
        
        # Average across all channels
        mean_dice = torch.stack(dice_values).mean()
        logger.debug(f"📦 Dice coefficient: {mean_dice.item():.6f}")
        return mean_dice


class ContrastRatio(nn.Module):
    """
    Contrast Ratio (CR) for 3D volumes.
    
    Measures the ratio between reconstructed anomaly/background contrast 
    and the ground-truth contrast. Values closer to 1 indicate better
    contrast preservation.
    
    Formula: CR = (⟨ŷ_M⟩/⟨ŷ_¬M⟩) / (⟨y_M⟩/⟨y_¬M⟩)
    where M is the anomaly mask, ⟨⟩ denotes spatial average.
    """
    
    def __init__(self, threshold: float = DICE_THRESHOLD):
        super().__init__()
        self.threshold = threshold
        logger.debug(f"🔧 Contrast ratio initialized: threshold={threshold}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrast ratio between predicted and target volumes.
        
        Args:
            pred (torch.Tensor): Predicted volume [B, C, D, H, W]
            target (torch.Tensor): Target volume [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Mean contrast ratio across batch and channels
        """
        logger.debug(f"🏃 Contrast ratio calculation: pred {pred.shape}, target {target.shape}")
        
        batch_size, num_channels = pred.shape[0], pred.shape[1]
        contrast_ratios = []
        
        for c in range(num_channels):
            pred_c = pred[:, c, ...]  # [B, D, H, W]
            target_c = target[:, c, ...]  # [B, D, H, W]
            
            # Normalize target to [0, 1] to create anomaly mask
            target_norm = (target_c - target_c.min()) / (target_c.max() - target_c.min() + CONTRAST_EPS)
            anomaly_mask = (target_norm > self.threshold).float()  # [B, D, H, W]
            background_mask = 1.0 - anomaly_mask
            
            batch_ratios = []
            for b in range(batch_size):
                pred_b = pred_c[b]  # [D, H, W]
                target_b = target_c[b]  # [D, H, W]
                mask_anomaly = anomaly_mask[b]  # [D, H, W]
                mask_background = background_mask[b]  # [D, H, W]
                
                # Check if there are any anomaly and background voxels
                if mask_anomaly.sum() > 0 and mask_background.sum() > 0:
                    # Calculate mean values in anomaly and background regions
                    pred_anomaly_mean = (pred_b * mask_anomaly).sum() / (mask_anomaly.sum() + CONTRAST_EPS)
                    pred_background_mean = (pred_b * mask_background).sum() / (mask_background.sum() + CONTRAST_EPS)
                    
                    target_anomaly_mean = (target_b * mask_anomaly).sum() / (mask_anomaly.sum() + CONTRAST_EPS)
                    target_background_mean = (target_b * mask_background).sum() / (mask_background.sum() + CONTRAST_EPS)
                    
                    # Calculate contrast ratios
                    pred_contrast = pred_anomaly_mean / (pred_background_mean + CONTRAST_EPS)
                    target_contrast = target_anomaly_mean / (target_background_mean + CONTRAST_EPS)
                    
                    # Calculate contrast ratio
                    contrast_ratio = pred_contrast / (target_contrast + CONTRAST_EPS)
                    batch_ratios.append(contrast_ratio)
                else:
                    # If no anomalies detected, set contrast ratio to 1.0
                    batch_ratios.append(torch.tensor(1.0, device=pred.device))
            
            if batch_ratios:
                contrast_ratios.append(torch.stack(batch_ratios).mean())
            else:
                contrast_ratios.append(torch.tensor(1.0, device=pred.device))
        
        # Average across all channels
        mean_contrast_ratio = torch.stack(contrast_ratios).mean()
        logger.debug(f"📦 Contrast ratio: {mean_contrast_ratio.item():.6f}")
        return mean_contrast_ratio


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
        self.dice_metric = DiceCoefficient()
        self.contrast_ratio_metric = ContrastRatio()
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
        
        # Calculate Dice Coefficient
        dice_value = self.dice_metric(pred, target)
        metrics['dice'] = dice_value.item()
        
        # Calculate Contrast Ratio
        contrast_ratio_value = self.contrast_ratio_metric(pred, target)
        metrics['contrast_ratio'] = contrast_ratio_value.item()
        
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
# PER-CHANNEL VALIDATION METRICS
# =============================================================================

def _tumor_mask_from_gt(gt_ch: torch.Tensor) -> torch.Tensor:
    """
    Create a tumor mask from GT for a single channel tensor shaped [B, D, H, W].
    Uses a robust percentile threshold (85th) per-sample to separate tumor vs background.
    
    Args:
        gt_ch: Ground truth channel tensor [B, D, H, W]
        
    Returns:
        torch.Tensor: Binary tumor mask [B, D, H, W]
    """
    B = gt_ch.shape[0]
    flat = gt_ch.reshape(B, -1)
    thr = torch.quantile(flat, 0.85, dim=1, keepdim=True)  # [B,1]
    mask = (flat > thr).float().reshape_as(gt_ch)          # [B,D,H,W]
    return mask


def dice_per_channel(pred_raw: torch.Tensor, gt_raw: torch.Tensor, channel: int) -> torch.Tensor:
    """
    Compute Dice coefficient for a single channel in raw physical units.
    
    Args:
        pred_raw: Predicted volumes [B, 2, D, H, W] in physical units
        gt_raw: Ground truth volumes [B, 2, D, H, W] in physical units  
        channel: Channel index (0 for μₐ, 1 for μ′ₛ)
        
    Returns:
        torch.Tensor: Mean Dice coefficient over batch for the selected channel
    """
    gt_ch   = gt_raw[:, channel]   # [B,D,H,W]
    pred_ch = pred_raw[:, channel] # [B,D,H,W]
    gt_mask = _tumor_mask_from_gt(gt_ch)
    
    # Threshold pred using the SAME GT-derived thresholding scheme for fairness:
    B = gt_ch.shape[0]
    gt_flat   = gt_ch.reshape(B, -1)
    thr       = torch.quantile(gt_flat, 0.85, dim=1, keepdim=True)     # [B,1]
    pred_mask = (pred_ch.reshape(B, -1) > thr).float().reshape_as(gt_ch)

    inter = (pred_mask * gt_mask).sum(dim=[1,2,3])
    denom = pred_mask.sum(dim=[1,2,3]) + gt_mask.sum(dim=[1,2,3]) + 1e-6
    dice  = (2.0 * inter / denom).mean()
    return dice


# Contrast ratio per channel, computed identically to overall contrast_ratio for consistency.
# Returns per-channel values on the same scale as global contrast.
def contrast_ratio_per_channel(pred_raw: torch.Tensor, gt_raw: torch.Tensor, channel: int) -> torch.Tensor:
    """
    Compute contrast ratio for a single channel in raw physical units.
    Uses the same formula as ContrastRatio class: CR = (pred_contrast) / (gt_contrast)
    where contrast = anomaly_mean / background_mean.
    
    Args:
        pred_raw: Predicted volumes [B, 2, D, H, W] in physical units
        gt_raw: Ground truth volumes [B, 2, D, H, W] in physical units
        channel: Channel index (0 for μₐ, 1 for μ′ₛ)
        
    Returns:
        torch.Tensor: Mean contrast ratio over batch for the selected channel
    """
    pred_ch = pred_raw[:, channel]  # [B,D,H,W]
    gt_ch = gt_raw[:, channel]  # [B,D,H,W]
    
    batch_size = pred_ch.shape[0]
    batch_ratios = []
    
    for b in range(batch_size):
        pred_b = pred_ch[b]  # [D,H,W]
        gt_b = gt_ch[b]  # [D,H,W]
        
        # Create anomaly mask from normalized ground truth (same as ContrastRatio class)
        gt_norm = (gt_b - gt_b.min()) / (gt_b.max() - gt_b.min() + CONTRAST_EPS)
        anomaly_mask = (gt_norm > DICE_THRESHOLD).float()
        background_mask = 1.0 - anomaly_mask
        
        # Check if there are any anomaly and background voxels
        if anomaly_mask.sum() > 0 and background_mask.sum() > 0:
            # Calculate mean values in anomaly and background regions for both pred and gt
            pred_anomaly_mean = (pred_b * anomaly_mask).sum() / (anomaly_mask.sum() + CONTRAST_EPS)
            pred_background_mean = (pred_b * background_mask).sum() / (background_mask.sum() + CONTRAST_EPS)
            
            gt_anomaly_mean = (gt_b * anomaly_mask).sum() / (anomaly_mask.sum() + CONTRAST_EPS)
            gt_background_mean = (gt_b * background_mask).sum() / (background_mask.sum() + CONTRAST_EPS)
            
            # Calculate contrast ratios (same formula as ContrastRatio class)
            pred_contrast = pred_anomaly_mean / (pred_background_mean + CONTRAST_EPS)
            gt_contrast = gt_anomaly_mean / (gt_background_mean + CONTRAST_EPS)
            
            # Calculate contrast ratio: pred_contrast / gt_contrast
            contrast_ratio = pred_contrast / (gt_contrast + CONTRAST_EPS)
            batch_ratios.append(contrast_ratio)
        else:
            # If no anomalies detected, set contrast ratio to 1.0
            batch_ratios.append(torch.tensor(1.0, device=pred_raw.device))
    
    if batch_ratios:
        return torch.stack(batch_ratios).mean()
    else:
        return torch.tensor(1.0, device=pred_raw.device)


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
        
        logger.debug(f"🔍 Metrics input shapes - enhanced: {enhanced_features.shape if enhanced_features is not None else 'None'}, "
                    f"cnn: {cnn_features.shape if cnn_features is not None else 'None'}, "
                    f"attention: {attention_weights.shape if attention_weights is not None else 'None'}")
        
        return metrics.calculate_all_metrics(
            pred, targets, enhanced_features, cnn_features, attention_weights
        )
