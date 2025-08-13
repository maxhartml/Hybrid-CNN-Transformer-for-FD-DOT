#!/usr/bin/env python3
"""
NIR-DOT Reconstruction Evaluation Metrics - Anomaly Detection Focus.

This module implements comprehensive metrics for evaluating NIR-DOT reconstruction
quality with a focus on anomaly detection and spatial accuracy measurement.

Core Metrics:
1. Root-Mean-Squared-Error (RMSE): Overall accuracy of optical property values
2. Sørensen-Dice Coefficient (SDC): Spatial similarity of reconstructed anomalies  
3. Contrast Ratio (CR): Ratio between reconstructed and ground-truth anomaly contrast
4. Channel-specific RMSE: Separate evaluation for μₐ and μₛ channels

Additional Stage 2 Metrics:
- Feature Enhancement Ratio: Transformer enhancement quantification
- Attention Entropy: Attention mechanism focus measurement

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

# Anomaly Detection Configuration
ANOMALY_THRESHOLD = 0.5                 # Threshold for anomaly detection
DICE_SMOOTH = 1e-6                      # Smoothing factor for Dice coefficient
CONTRAST_EPS = 1e-8                     # Small epsilon for contrast ratio calculation

# Feature Analysis Configuration (Stage 2 only)
ENTROPY_EPS = 1e-12                     # Small epsilon for entropy calculation
ENHANCEMENT_RATIO_EPS = 1e-8            # Small epsilon for ratio calculation

# Channel Names for Logging
ABSORPTION_CHANNEL = "mu_a"             # μₐ channel name
SCATTERING_CHANNEL = "mu_s"             # μₛ channel name
CHANNEL_NAMES = [ABSORPTION_CHANNEL, SCATTERING_CHANNEL]

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# CORE RECONSTRUCTION METRICS
# =============================================================================

class RMSELoss(nn.Module):
    """
    Standard Root Mean Square Error loss for reconstruction accuracy.
    
    This implements the primary reconstruction metric used for:
    - Volume reconstruction accuracy (Stage 1)
    - Latent vector prediction accuracy (Stage 2)
    - Overall accuracy measurement of any tensor comparison
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE loss between input and target tensors.
        
        Args:
            input (torch.Tensor): Predicted values (volumes, latents, etc.)
            target (torch.Tensor): Ground truth values (volumes, latents, etc.)
            
        Returns:
            torch.Tensor: RMSE value
        """
        mse = F.mse_loss(input, target)
        rmse = torch.sqrt(mse + 1e-8)  # Small epsilon for numerical stability
        return rmse


class SorensenDiceCoefficient(nn.Module):
    """
    Sørensen-Dice coefficient for spatial similarity of anomalies.
    
    Measures the overlap between predicted and ground truth anomalies,
    critical for evaluating spatial accuracy in tumor detection tasks.
    """
    
    def __init__(self, threshold: float = ANOMALY_THRESHOLD, smooth: float = DICE_SMOOTH):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Sørensen-Dice coefficient for anomaly detection.
        
        Args:
            pred (torch.Tensor): Predicted volumes [batch, channels, D, H, W]
            target (torch.Tensor): Ground truth volumes [batch, channels, D, H, W]
            
        Returns:
            torch.Tensor: Dice coefficient (0 = no overlap, 1 = perfect overlap)
        """
        # Flatten spatial dimensions for calculation
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # Create binary masks for anomaly detection
        pred_binary = (pred_flat > self.threshold).float()
        target_binary = (target_flat > self.threshold).float()
        
        # Calculate intersection and union
        intersection = (pred_binary * target_binary).sum(dim=-1)
        total = pred_binary.sum(dim=-1) + target_binary.sum(dim=-1)
        
        # Dice coefficient with smoothing
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        # Return mean across batch and channels
        return dice.mean()


class ContrastRatio(nn.Module):
    """
    Contrast ratio between anomaly and background regions.
    
    Measures how well the reconstructed anomaly contrast is preserved
    compared to the ground truth, critical for quantitative accuracy.
    """
    
    def __init__(self, anomaly_threshold: float = ANOMALY_THRESHOLD):
        super().__init__()
        self.anomaly_threshold = anomaly_threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrast ratio for anomaly detection.
        
        Args:
            pred (torch.Tensor): Predicted volumes [batch, channels, D, H, W]
            target (torch.Tensor): Ground truth volumes [batch, channels, D, H, W]
            
        Returns:
            torch.Tensor: Contrast ratio (1 = perfect contrast preservation)
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # Identify anomaly and background regions from ground truth
        anomaly_mask = target_flat > self.anomaly_threshold
        background_mask = ~anomaly_mask
        
        # Calculate contrast in predictions
        pred_anomaly_mean = []
        pred_background_mean = []
        target_anomaly_mean = []
        target_background_mean = []
        
        for b in range(pred.size(0)):
            for c in range(pred.size(1)):
                pred_ch = pred_flat[b, c]
                target_ch = target_flat[b, c]
                anomaly_mask_ch = anomaly_mask[b, c]
                background_mask_ch = background_mask[b, c]
                
                if anomaly_mask_ch.sum() > 0 and background_mask_ch.sum() > 0:
                    pred_anomaly_mean.append(pred_ch[anomaly_mask_ch].mean())
                    pred_background_mean.append(pred_ch[background_mask_ch].mean())
                    target_anomaly_mean.append(target_ch[anomaly_mask_ch].mean())
                    target_background_mean.append(target_ch[background_mask_ch].mean())
        
        if len(pred_anomaly_mean) == 0:
            return torch.tensor(1.0, device=pred.device)  # No anomalies detected
        
        pred_anomaly_mean = torch.stack(pred_anomaly_mean)
        pred_background_mean = torch.stack(pred_background_mean)
        target_anomaly_mean = torch.stack(target_anomaly_mean)
        target_background_mean = torch.stack(target_background_mean)
        
        # Calculate contrast ratios
        pred_contrast = pred_anomaly_mean - pred_background_mean
        target_contrast = target_anomaly_mean - target_background_mean
        
        # Contrast ratio with numerical stability
        contrast_ratio = pred_contrast / (target_contrast + CONTRAST_EPS)
        
        return contrast_ratio.mean()


class ChannelSpecificRMSE(nn.Module):
    """
    Calculate RMSE separately for each tissue property channel (μₐ, μₛ).
    
    This provides detailed analysis of reconstruction accuracy for each
    optical property, enabling targeted model improvements.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate channel-specific RMSE.
        
        Args:
            pred (torch.Tensor): Predicted volumes [batch, 2, D, H, W]
            target (torch.Tensor): Ground truth volumes [batch, 2, D, H, W]
            
        Returns:
            Dict[str, torch.Tensor]: RMSE for each channel
        """
        results = {}
        
        for i, channel_name in enumerate(CHANNEL_NAMES):
            channel_pred = pred[:, i, :, :, :]
            channel_target = target[:, i, :, :, :]
            
            mse = F.mse_loss(channel_pred, channel_target)
            rmse = torch.sqrt(mse + 1e-8)
            results[f"rmse_{channel_name}"] = rmse
        
        # Overall RMSE
        overall_mse = F.mse_loss(pred, target)
        results["rmse_overall"] = torch.sqrt(overall_mse + 1e-8)
        
        return results


# =============================================================================
# STAGE 2 FEATURE ANALYSIS METRICS
# =============================================================================

class FeatureEnhancementRatio(nn.Module):
    """
    Calculate feature enhancement ratio for transformer analysis.
    
    Measures how much the transformer enhances features compared to
    CNN-only processing, providing insight into transformer contribution.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, enhanced_features: torch.Tensor, 
                cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate feature enhancement ratio.
        
        Args:
            enhanced_features (torch.Tensor): Features after transformer enhancement
            cnn_features (torch.Tensor): Features from CNN only
            
        Returns:
            torch.Tensor: Enhancement ratio
        """
        # Calculate feature differences
        feature_diff = enhanced_features - cnn_features
        enhancement_magnitude = torch.norm(feature_diff, dim=-1)
        
        # Calculate CNN feature magnitude
        cnn_magnitude = torch.norm(cnn_features, dim=-1)
        
        # Calculate ratio with numerical stability
        ratio = enhancement_magnitude / (cnn_magnitude + ENHANCEMENT_RATIO_EPS)
        
        return ratio.mean()


class AttentionEntropy(nn.Module):
    """
    Calculate attention entropy to measure attention focus.
    
    Measures how focused the attention weights are, with low entropy
    indicating focused attention and high entropy indicating diffuse attention.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention entropy.
        
        Args:
            attention_weights (torch.Tensor): Attention weights [batch, heads, seq, seq]
            
        Returns:
            torch.Tensor: Mean attention entropy
        """
        # Add small epsilon for numerical stability
        attention_weights = attention_weights + ENTROPY_EPS
        
        # Calculate entropy: -sum(p * log(p))
        log_attention = torch.log(attention_weights)
        entropy = -(attention_weights * log_attention).sum(dim=-1)
        
        # Return mean entropy across all dimensions
        return entropy.mean()


# =============================================================================
# METRICS COLLECTION AND W&B INTEGRATION
# =============================================================================

class NIRDOTMetrics:
    """
    Comprehensive metrics collection for NIR-DOT reconstruction evaluation.
    
    This class manages all metrics calculation and provides convenient
    interfaces for training loop integration and W&B logging.
    """
    
    def __init__(self, stage: str = "stage1"):
        """
        Initialize metrics for specific training stage.
        
        Args:
            stage (str): Training stage ("stage1" or "stage2")
        """
        self.stage = stage
        
        # Core reconstruction metrics (used in both stages)
        self.rmse = RMSELoss()  # Unified RMSE for volumes and latents
        self.dice = SorensenDiceCoefficient()
        self.contrast_ratio = ContrastRatio()
        self.channel_rmse = ChannelSpecificRMSE()
        
        # Stage 2 specific metrics
        if stage == "stage2":
            self.feature_enhancement = FeatureEnhancementRatio()
            self.attention_entropy = AttentionEntropy()
        
        logger.info(f"📊 NIRDOTMetrics initialized for {stage} with unified RMSE support")
    
    def calculate_reconstruction_metrics(self, pred: torch.Tensor, 
                                       target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate core reconstruction metrics.
        
        Args:
            pred (torch.Tensor): Predicted volumes
            target (torch.Tensor): Ground truth volumes
            
        Returns:
            Dict[str, float]: Reconstruction metrics
        """
        metrics = {}
        
        # Core metrics
        metrics["dice"] = self.dice(pred, target).item()  # Use 'dice' for consistency
        metrics["contrast_ratio"] = self.contrast_ratio(pred, target).item()
        
        # Channel-specific RMSE
        channel_metrics = self.channel_rmse(pred, target)
        for key, value in channel_metrics.items():
            metrics[key] = value.item()
        
        return metrics
    
    def calculate_feature_metrics(self, enhanced_features: torch.Tensor,
                                cnn_features: torch.Tensor,
                                attention_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate Stage 2 feature analysis metrics.
        
        Args:
            enhanced_features (torch.Tensor): Enhanced features from transformer
            cnn_features (torch.Tensor): CNN-only features
            attention_weights (Optional[torch.Tensor]): Attention weights
            
        Returns:
            Dict[str, float]: Feature metrics
        """
        if self.stage != "stage2":
            return {}
        
        metrics = {}
        
        # Feature enhancement ratio
        enhancement_ratio = self.feature_enhancement(enhanced_features, cnn_features)
        metrics["enhancement_ratio"] = enhancement_ratio.item()
        
        # Attention entropy
        if attention_weights is not None:
            entropy = self.attention_entropy(attention_weights)
            metrics["attention_entropy"] = entropy.item()
        
        return metrics
    
    def calculate_all_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                            enhanced_features: Optional[torch.Tensor] = None,
                            cnn_features: Optional[torch.Tensor] = None,
                            attention_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate all relevant metrics for the current stage.
        
        Args:
            pred (torch.Tensor): Predicted volumes
            target (torch.Tensor): Ground truth volumes
            enhanced_features (Optional[torch.Tensor]): Enhanced features
            cnn_features (Optional[torch.Tensor]): CNN features
            attention_weights (Optional[torch.Tensor]): Attention weights
            
        Returns:
            Dict[str, float]: All calculated metrics
        """
        # Calculate reconstruction metrics
        metrics = self.calculate_reconstruction_metrics(pred, target)
        
        # Add feature metrics for Stage 2
        if self.stage == "stage2" and enhanced_features is not None and cnn_features is not None:
            feature_metrics = self.calculate_feature_metrics(
                enhanced_features, cnn_features, attention_weights
            )
            metrics.update(feature_metrics)
        
        return metrics


def log_to_wandb(metrics: Dict[str, float], epoch: int, 
                prefix: str = "train") -> None:
    """
    Log metrics to Weights & Biases with proper formatting.
    
    Args:
        metrics (Dict[str, float]): Metrics dictionary
        epoch (int): Current epoch
        prefix (str): Prefix for metric names ("train", "val", "test")
    """
    if not wandb.run:
        return
    
    # Format metrics for W&B logging
    wandb_metrics = {}
    for key, value in metrics.items():
        wandb_metrics[f"{prefix}/{key}"] = value
    
    wandb_metrics["epoch"] = epoch
    wandb.log(wandb_metrics)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_metrics_for_stage(stage: str) -> NIRDOTMetrics:
    """
    Create metrics instance for specific training stage.
    
    Args:
        stage (str): Training stage ("stage1" or "stage2")
        
    Returns:
        NIRDOTMetrics: Configured metrics instance
    """
    return NIRDOTMetrics(stage=stage)


def calculate_batch_metrics(metrics: NIRDOTMetrics, outputs: Dict[str, torch.Tensor],
                          targets: torch.Tensor, stage: str, 
                          target_latent: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Calculate metrics for a single batch based on model outputs.
    
    Phase 2 Simplified: For Stage 2, uses simple RMSE on latent vectors.
    
    Args:
        metrics (NIRDOTMetrics): Metrics calculator
        outputs (Dict[str, torch.Tensor]): Model outputs
        targets (torch.Tensor): Ground truth volumes
        stage (str): Training stage
        target_latent (Optional[torch.Tensor]): Ground truth latent vectors for Stage 2
        
    Returns:
        Dict[str, float]: Batch metrics
    """
    if stage == "stage1":
        # Stage 1: Standard volume reconstruction metrics
        pred = outputs['reconstructed']
        return metrics.calculate_reconstruction_metrics(pred, targets)
        
    else:  # Stage 2
        batch_metrics = {}
        
        # Phase 2: Simple latent vector prediction with RMSE
        if 'predicted_latent' in outputs and target_latent is not None:
            predicted_latent = outputs['predicted_latent']
            # Simple RMSE in latent space using unified RMSE
            latent_rmse = metrics.rmse(predicted_latent, target_latent)
            batch_metrics['latent_rmse'] = latent_rmse.item()
            
            # Volume metrics for evaluation (using decoded volumes)
            if 'reconstructed' in outputs:
                pred = outputs['reconstructed']
                volume_metrics = metrics.calculate_reconstruction_metrics(pred, targets)
                batch_metrics.update(volume_metrics)
        else:
            # Fallback to volume metrics
            pred = outputs['reconstructed']
            volume_metrics = metrics.calculate_reconstruction_metrics(pred, targets)
            batch_metrics.update(volume_metrics)
        
        return batch_metrics


# =============================================================================
# SCAN DENSITY EVALUATION (VALIDATION APPROACH)
# =============================================================================

def evaluate_scan_density_performance(model, test_data, measurement_counts: List[int], 
                                     device: torch.device) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model performance across different scan densities.
    
    This function tests model performance with varying numbers of measurements
    to understand the relationship between measurement density and reconstruction quality.
    
    Args:
        model: Trained model for evaluation
        test_data: Test dataset
        measurement_counts (List[int]): Numbers of measurements to test
        device (torch.device): Device for computation
        
    Returns:
        Dict[int, Dict[str, float]]: Performance metrics for each measurement count
    """
    model.eval()
    results = {}
    
    metrics_calculator = create_metrics_for_stage("stage2")
    
    with torch.no_grad():
        for num_measurements in measurement_counts:
            logger.info(f"📊 Evaluating with {num_measurements} measurements...")
            
            batch_metrics = []
            
            for batch in test_data:
                # Extract data
                nir_measurements = batch['measurements'].to(device)
                targets = batch['ground_truth'].to(device)
                tissue_patches = batch.get('tissue_patches')
                
                # Subsample measurements randomly
                total_measurements = nir_measurements.size(1)
                if num_measurements < total_measurements:
                    indices = torch.randperm(total_measurements)[:num_measurements]
                    indices = indices.sort()[0]  # Sort for consistency
                    nir_measurements = nir_measurements[:, indices, :]
                    
                    if tissue_patches is not None:
                        tissue_patches = tissue_patches[:, indices, :]
                
                # Forward pass
                outputs = model(nir_measurements, tissue_patches)
                
                # Calculate metrics
                metrics = calculate_batch_metrics(
                    metrics_calculator, outputs, targets, "stage2"
                )
                batch_metrics.append(metrics)
            
            # Average metrics across batches
            avg_metrics = {}
            for key in batch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in batch_metrics])
            
            results[num_measurements] = avg_metrics
            
            logger.info(f"✅ {num_measurements} measurements: "
                       f"SDC={avg_metrics['dice']:.3f}, "
                       f"CR={avg_metrics['contrast_ratio']:.3f}, "
                       f"RMSE μₐ={avg_metrics['rmse_mu_a']:.4f}")
    
    return results

