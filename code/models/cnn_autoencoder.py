#!/usr/bin/env python3
"""
CNN Autoencoder for NIR-DOT Reconstruction.

This module implements a 3D convolutional autoencoder for near-infrared diffuse optical 
tomography (NIR-DOT) volume reconstruction. The architecture uses residual blocks for 
improved gradient flow and progressive downsampling/upsampling for spatial feature learning.

The autoencoder is designed for stage 1 pre-training in a two-stage hybrid approach,
focusing on learning low-level spatial features from DOT measurements.

Architecture Optimization:
- Base channels: 16 (optimized from 32/64 to reduce parameters)
- Target parameters: ~7M total (down from 26.9M)
- Feature dimension: 256 (maintained as required by supervisor)
- Channel progression: 16→32→64→128→256 (efficient scaling)

Classes:
    ResidualBlock: 3D residual block with skip connections
    CNNEncoder: Progressive downsampling encoder for feature extraction
    CNNDecoder: Progressive upsampling decoder for volume reconstruction
    CNNAutoEncoder: Complete autoencoder combining encoder and decoder

Author: Max Hart
Date: July 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import sys
from typing import Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Project imports
from code.utils.logging_config import get_model_logger
from code.training.training_config import VOLUME_SHAPE, LATENT_DIM

# =============================================================================
# HYPERPARAMETERS AND CONSTANTS
# =============================================================================

# Model Architecture Parameters
INPUT_CHANNELS = 2                      # Absorption and scattering coefficients
OUTPUT_SIZE = VOLUME_SHAPE              # Target volume dimensions from config
BASE_CHANNELS = 16                      # Base number of CNN channels (optimized for ~7M params)
FEATURE_DIM = LATENT_DIM                # Encoder output feature dimension - matches latent dimension

# Encoder Architecture Configuration
ENCODER_KERNEL_INITIAL = 7              # Initial convolution kernel size
ENCODER_STRIDE_INITIAL = 2              # Initial convolution stride
ENCODER_PADDING_INITIAL = 3             # Initial convolution padding
ENCODER_MAXPOOL_KERNEL = 3              # Max pooling kernel size
ENCODER_MAXPOOL_STRIDE = 2              # Max pooling stride
ENCODER_MAXPOOL_PADDING = 1             # Max pooling padding

# Residual Block Configuration
RESIDUAL_CONV_KERNEL = 3                # Residual block convolution kernel size
RESIDUAL_CONV_PADDING = 1               # Residual block convolution padding
RESIDUAL_SHORTCUT_KERNEL = 1            # Shortcut connection kernel size
RESIDUAL_BLOCKS_PER_LAYER = 1           # Number of residual blocks per layer (optimized)

# Decoder Architecture Configuration
DECODER_INIT_SIZE = 2                   # Initial spatial size for decoder
DECODER_TRANSCONV_KERNEL = 4            # Transposed convolution kernel size
DECODER_TRANSCONV_STRIDE = 2            # Transposed convolution stride
DECODER_TRANSCONV_PADDING = 1           # Transposed convolution padding
DECODER_FINAL_CONV_KERNEL = 3           # Final convolution kernel size
DECODER_FINAL_CONV_PADDING = 1          # Final convolution padding

# Channel Progression (optimized for parameter efficiency)
ENCODER_CHANNEL_MULTIPLIERS = [1, 2, 4, 8, 16]    # Progressive increase: 16→32→64→128→256
DECODER_CHANNEL_DIVISORS = [16, 8, 4, 2, 1]       # Progressive decrease: 256→128→64→32→16

# Weight Initialization
WEIGHT_INIT_STD = 0.02                  # Standard deviation for weight initialization

# Safe Improvements Configuration
USE_GROUP_NORM = True                   # Use GroupNorm instead of BatchNorm for stability
GROUP_NORM_GROUPS = 8                   # Number of groups for GroupNorm
USE_WEIGHT_STANDARDIZATION = True       # Enable weight standardization for conv layers
USE_SQUEEZE_EXCITATION = False          # Enable squeeze-and-excitation blocks (default off)
SE_REDUCTION_RATIO = 16                 # Reduction ratio for SE blocks
ENCODER_DROPOUT_DEEP = 0.1              # Dropout rate for deeper encoder layers (layer3, layer4)

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# HELPER COMPONENTS FOR SAFE IMPROVEMENTS
# =============================================================================

class WeightStandardizedConv3d(nn.Conv3d):
    """
    3D Convolution with Weight Standardization for improved training stability.
    
    Weight standardization normalizes the weights of each convolutional kernel
    to have zero mean and unit variance, which helps stabilize training and
    improves convergence, especially beneficial for small batch sizes.
    """
    
    def forward(self, x):
        if self.training or USE_WEIGHT_STANDARDIZATION:
            # Standardize weights: subtract mean and divide by std
            weight = self.weight
            weight_mean = weight.mean(dim=[1, 2, 3, 4], keepdim=True)
            weight_std = weight.std(dim=[1, 2, 3, 4], keepdim=True) + 1e-5
            weight = (weight - weight_mean) / weight_std
            
            # Use standardized weights for convolution
            return F.conv3d(x, weight, self.bias, self.stride, 
                          self.padding, self.dilation, self.groups)
        else:
            return super().forward(x)


class WeightStandardizedConvTranspose3d(nn.ConvTranspose3d):
    """
    3D Transposed Convolution with Weight Standardization for improved training stability.
    """
    
    def forward(self, x, output_size=None):
        if self.training or USE_WEIGHT_STANDARDIZATION:
            # Standardize weights: subtract mean and divide by std
            weight = self.weight
            weight_mean = weight.mean(dim=[0, 2, 3, 4], keepdim=True)
            weight_std = weight.std(dim=[0, 2, 3, 4], keepdim=True) + 1e-5
            weight = (weight - weight_mean) / weight_std
            
            # Use standardized weights for transposed convolution
            return F.conv_transpose3d(x, weight, self.bias, self.stride,
                                    self.padding, self.output_padding, 
                                    self.groups, self.dilation)
        else:
            return super().forward(x)


class SqueezeExcitation3d(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.
    
    SE blocks improve feature calibration by learning channel-wise attention weights.
    This is a safe improvement that doesn't leak spatial information and helps
    the model focus on important channels without breaking the latent bottleneck.
    """
    
    def __init__(self, channels, reduction_ratio=SE_REDUCTION_RATIO):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, d, h, w = x.size()
        # Global average pooling
        se = self.global_pool(x).view(b, c)
        # Channel attention weights
        se = self.fc(se).view(b, c, 1, 1, 1)
        # Apply attention weights
        return x * se


def get_norm_layer(channels, use_group_norm=USE_GROUP_NORM, num_groups=GROUP_NORM_GROUPS):
    """
    Get normalization layer (GroupNorm or BatchNorm3d).
    
    GroupNorm is more stable for small batch sizes and doesn't depend on
    batch statistics, making it safer for varying batch sizes during training.
    """
    if use_group_norm:
        # Ensure num_groups divides channels evenly
        effective_groups = min(num_groups, channels)
        while channels % effective_groups != 0 and effective_groups > 1:
            effective_groups -= 1
        return nn.GroupNorm(effective_groups, channels)
    else:
        return nn.BatchNorm3d(channels)


def get_conv3d_layer(use_weight_std=USE_WEIGHT_STANDARDIZATION):
    """Get Conv3d layer (with or without weight standardization)."""
    return WeightStandardizedConv3d if use_weight_std else nn.Conv3d


def get_conv_transpose3d_layer(use_weight_std=USE_WEIGHT_STANDARDIZATION):
    """Get ConvTranspose3d layer (with or without weight standardization)."""
    return WeightStandardizedConvTranspose3d if use_weight_std else nn.ConvTranspose3d

# =============================================================================
# NETWORK COMPONENTS
# =============================================================================


class ResidualBlock(nn.Module):
    """
    Enhanced 3D Residual block with optional Squeeze-and-Excitation and improved normalization.
    
    Safe improvements:
    - GroupNorm instead of BatchNorm for stability
    - Optional Squeeze-and-Excitation for feature calibration
    - Weight standardization for better training dynamics
    - All improvements preserve the residual structure and don't leak spatial information
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Convolution stride. Defaults to 1.
        use_se (bool, optional): Whether to use Squeeze-and-Excitation. Defaults to USE_SQUEEZE_EXCITATION.
        use_dropout (bool, optional): Whether to apply dropout after the block. Defaults to False.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_se: bool = USE_SQUEEZE_EXCITATION, use_dropout: bool = False,
                 dropout_rate: float = ENCODER_DROPOUT_DEEP):
        super().__init__()
        
        Conv3d = get_conv3d_layer()
        
        # First convolution with potential downsampling
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=RESIDUAL_CONV_KERNEL, 
                           stride=stride, padding=RESIDUAL_CONV_PADDING, bias=False)
        self.norm1 = get_norm_layer(out_channels)
        
        # Second convolution maintains spatial dimensions
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=RESIDUAL_CONV_KERNEL,
                           stride=1, padding=RESIDUAL_CONV_PADDING, bias=False)
        self.norm2 = get_norm_layer(out_channels)
        
        # Skip connection projection when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=RESIDUAL_SHORTCUT_KERNEL,
                      stride=stride, bias=False),
                get_norm_layer(out_channels)
            )
        
        # Optional Squeeze-and-Excitation block
        self.se_block = SqueezeExcitation3d(out_channels) if use_se else None
        
        # Optional dropout for deeper layers
        self.dropout = nn.Dropout3d(dropout_rate) if use_dropout else None
    
    def forward(self, x):
        """
        Forward pass through enhanced residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W')
        """
        # Main path
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # Skip connection
        shortcut = self.shortcut(x)
        
        # Add residual connection
        out += shortcut
        
        # Apply SE block if enabled
        if self.se_block is not None:
            out = self.se_block(out)
        
        # Apply ReLU activation
        out = F.relu(out)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class CNNEncoder(nn.Module):
    """
    Enhanced 3D CNN Encoder for feature extraction from volumetric data.
    
    Safe improvements:
    - GroupNorm for better stability with small batch sizes  
    - Weight standardization for improved training dynamics
    - Dropout3d in deeper layers for better generalization
    - Optional SE blocks for feature calibration (default off)
    - Improved weight initialization (Kaiming normal for conv, Xavier for linear)
    
    The architecture maintains the same encoder→latent bottleneck structure,
    ensuring Stage 2 compatibility while improving training stability and performance.
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to INPUT_CHANNELS.
        base_channels (int, optional): Base number of channels. Defaults to BASE_CHANNELS.
        feature_dim (int, optional): Latent feature dimension. Defaults to FEATURE_DIM.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.1.
        use_se (bool, optional): Whether to use SE blocks. Defaults to USE_SQUEEZE_EXCITATION.
    """
    
    def __init__(self, input_channels: int = INPUT_CHANNELS, 
                 base_channels: int = BASE_CHANNELS, 
                 feature_dim: int = FEATURE_DIM,
                 dropout_rate: float = 0.1,
                 use_se: bool = USE_SQUEEZE_EXCITATION):
        super().__init__()
        
        Conv3d = get_conv3d_layer()
        
        # Initial feature extraction with aggressive downsampling
        self.initial_conv = nn.Sequential(
            Conv3d(input_channels, base_channels, kernel_size=ENCODER_KERNEL_INITIAL, 
                  stride=ENCODER_STRIDE_INITIAL, padding=ENCODER_PADDING_INITIAL, bias=False),
            get_norm_layer(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=ENCODER_MAXPOOL_KERNEL, stride=ENCODER_MAXPOOL_STRIDE, 
                        padding=ENCODER_MAXPOOL_PADDING)
        )
        
        # Progressive feature extraction with residual blocks
        # Layer 1-2: No dropout (shallow layers)
        self.layer1 = self._make_layer(base_channels, base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=1, use_se=use_se)
        self.layer2 = self._make_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                                     base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=2, use_se=use_se)
        
        # Layer 3-4: Add dropout for deeper layers (safe regularization)
        self.layer3 = self._make_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                                     base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=2, use_se=use_se, 
                                     use_dropout=True)
        self.layer4 = self._make_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                                     base_channels * ENCODER_CHANNEL_MULTIPLIERS[4], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=2, use_se=use_se,
                                     use_dropout=True)
        
        # Spatial dimension reduction to fixed-size feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Configurable feature dimension with linear projection
        self.feature_dim = feature_dim
        self.feature_projection = nn.Linear(base_channels * 16, feature_dim)  # 256 → 256 for base_channels=16
        
        # Initialize weights with improved schemes
        self._init_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int, use_se: bool = False, 
                    use_dropout: bool = False):
        """Create a layer with residual blocks."""
        layers = []
        
        # First block handles stride and channel change
        layers.append(ResidualBlock(in_channels, out_channels, stride, 
                                  use_se=use_se, use_dropout=use_dropout))
        
        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, 
                                      use_se=use_se, use_dropout=use_dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights with improved schemes."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, WeightStandardizedConv3d)):
                # Kaiming normal for convolutional layers (better for ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through enhanced encoder.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, input_channels, D, H, W)
            
        Returns:
            torch.Tensor: Latent features of shape (batch_size, feature_dim)
        """
        # Progressive feature extraction
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # Includes dropout for regularization
        x = self.layer4(x)  # Includes dropout for regularization
        
        # Global spatial aggregation to fixed-size vector
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout and project to target feature dimension
        x = self.dropout(x)
        x = self.feature_projection(x)
        
        return x


class CNNDecoder(nn.Module):
    """
    Enhanced 3D CNN Decoder for volume reconstruction from encoded features.
    
    Safe improvements:
    - GroupNorm for better stability with small batch sizes
    - Weight standardization for improved training dynamics  
    - Optional SE blocks for feature calibration (default off)
    - Improved weight initialization (Kaiming normal for conv, Xavier for linear)
    
    The decoder maintains the same latent→volume reconstruction structure,
    ensuring Stage 2 compatibility. No skip connections are added to preserve
    the information bottleneck that prevents cheating in Stage 2.
    
    Args:
        feature_dim (int, optional): Dimension of input latent features. Defaults to FEATURE_DIM.
        output_size (Tuple[int, int, int], optional): Target output volume size. Defaults to OUTPUT_SIZE.
        base_channels (int, optional): Base number of channels. Defaults to BASE_CHANNELS.
        use_se (bool, optional): Whether to use SE blocks. Defaults to USE_SQUEEZE_EXCITATION.
    """
    
    def __init__(self, feature_dim: int = FEATURE_DIM, 
                 output_size: Tuple[int, int, int] = OUTPUT_SIZE,
                 base_channels: int = BASE_CHANNELS,
                 use_se: bool = USE_SQUEEZE_EXCITATION):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.base_channels = base_channels
        
        ConvTranspose3d = get_conv_transpose3d_layer()
        Conv3d = get_conv3d_layer()
        
        # Calculate proper initial spatial dimensions to match encoder
        # Encoder path: 64 -> 32 (stride=2) -> 16 (maxpool stride=2) -> 16 -> 8 -> 4 -> 2
        # So decoder should start from 2x2x2 to be symmetric
        self.init_size = DECODER_INIT_SIZE
        
        # Linear projection to expand feature vector to initial 3D volume
        self.fc = nn.Linear(feature_dim, base_channels * ENCODER_CHANNEL_MULTIPLIERS[4] * (self.init_size ** 3))
        
        # Progressive upsampling layers with transposed convolutions
        # Perfectly symmetric to encoder: 2→4→8→16→32→64 (exact mirror of encoder path)
        self.deconv1 = nn.Sequential(
            ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[4], 
                           base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                           kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                           padding=DECODER_TRANSCONV_PADDING, bias=False),
            get_norm_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                           base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                           kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                           padding=DECODER_TRANSCONV_PADDING, bias=False),
            get_norm_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                           base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                           kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                           padding=DECODER_TRANSCONV_PADDING, bias=False),
            get_norm_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.Sequential(
            ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                           base_channels * ENCODER_CHANNEL_MULTIPLIERS[0], 
                           kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                           padding=DECODER_TRANSCONV_PADDING, bias=False),
            get_norm_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to reach 64x64x64 and reduce to 16 channels
        self.deconv5 = nn.Sequential(
            ConvTranspose3d(base_channels, 16, 
                           kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                           padding=DECODER_TRANSCONV_PADDING, bias=False),
            get_norm_layer(16),
            nn.ReLU(inplace=True)
        )
        
        # Optional SE blocks for each decoder layer (but default off for safety)
        self.se_blocks = nn.ModuleList([
            SqueezeExcitation3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3]) if use_se else None,
            SqueezeExcitation3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2]) if use_se else None,
            SqueezeExcitation3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1]) if use_se else None,
            SqueezeExcitation3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[0]) if use_se else None,
            SqueezeExcitation3d(16) if use_se else None
        ])
        
        # Final output layer to dual channel volume (absorption + scattering)
        self.final_conv = Conv3d(16, INPUT_CHANNELS, 
                               kernel_size=DECODER_FINAL_CONV_KERNEL, padding=DECODER_FINAL_CONV_PADDING)
        
        # Initialize weights with improved schemes
        self._init_weights()
        
        logger.info(f"📊 Enhanced CNNDecoder initialized: "
                    f"feature_dim={feature_dim}, output_size={output_size}, base_channels={base_channels}")
    
    def _init_weights(self):
        """Initialize weights with improved schemes."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose3d, WeightStandardizedConvTranspose3d, nn.Conv3d, WeightStandardizedConv3d)):
                # Kaiming normal for convolutional layers (better for ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through enhanced decoder.
        
        Args:
            x (torch.Tensor): Encoded features of shape (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        # Expand feature vector to 3D volume starting from 2x2x2 (matching encoder)
        x = self.fc(x)  # [batch, feature_dim] -> [batch, base_channels*16*8]
        
        x = x.view(x.size(0), self.base_channels * ENCODER_CHANNEL_MULTIPLIERS[4], 
                   self.init_size, self.init_size, self.init_size)  # [batch, 256, 2, 2, 2] for base_channels=16
        
        # Progressive upsampling through transposed convolutions with optional SE blocks
        x = self.deconv1(x)  # 2x2x2 -> 4x4x4, channels: 256->128
        if self.se_blocks[0] is not None:
            x = self.se_blocks[0](x)
        
        x = self.deconv2(x)  # 4x4x4 -> 8x8x8, channels: 128->64
        if self.se_blocks[1] is not None:
            x = self.se_blocks[1](x)
        
        x = self.deconv3(x)  # 8x8x8 -> 16x16x16, channels: 64->32
        if self.se_blocks[2] is not None:
            x = self.se_blocks[2](x)
        
        x = self.deconv4(x)  # 16x16x16 -> 32x32x32, channels: 32->16
        if self.se_blocks[3] is not None:
            x = self.se_blocks[3](x)
        
        x = self.deconv5(x)  # 32x32x32 -> 64x64x64, channels: 16->16
        if self.se_blocks[4] is not None:
            x = self.se_blocks[4](x)
        
        # Generate final dual-channel output (μₐ + μ′s)
        x = self.final_conv(x)  # 64x64x64 -> 64x64x64, channels: 16->2
        
        # Ensure exact output dimensions match target (should be exactly 64x64x64)
        if x.shape[2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='trilinear', 
                              align_corners=False)
        
        return x


class CNNAutoEncoder(nn.Module):
    """
    Enhanced Complete 3D CNN Autoencoder for volumetric reconstruction.
    
    Safe improvements:
    - All normalization, weight standardization, and SE improvements from components
    - Improved weight initialization schemes (Kaiming for conv, Xavier for linear)
    - Comprehensive parameter counting and logging
    - Stage 2 compatibility preserved (latent bottleneck maintained)
    - No skip connections between encoder/decoder (prevents information leakage)
    
    End-to-end autoencoder architecture combining enhanced CNN encoder and decoder for 
    learning compact representations of 3D volumes. Designed for stage 1 
    pre-training in a multi-stage learning approach, focusing on low-level 
    spatial feature extraction and reconstruction.
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to INPUT_CHANNELS.
        output_size (Tuple[int, int, int], optional): Target output volume size. Defaults to OUTPUT_SIZE.
        feature_dim (int, optional): Latent feature dimension. Defaults to FEATURE_DIM.
        base_channels (int, optional): Base number of channels. Defaults to BASE_CHANNELS.
        dropout_rate (float, optional): Dropout rate for encoder regularization. Defaults to 0.1.
        use_se (bool, optional): Whether to use SE blocks. Defaults to USE_SQUEEZE_EXCITATION.
    """
    
    def __init__(self, input_channels: int = INPUT_CHANNELS, 
                 output_size: Tuple[int, int, int] = OUTPUT_SIZE,
                 feature_dim: int = FEATURE_DIM,
                 base_channels: int = BASE_CHANNELS,
                 dropout_rate: float = 0.1,
                 use_se: bool = USE_SQUEEZE_EXCITATION):
        super().__init__()
        
        logger.info(f"🏗️  Initializing Enhanced CNN Autoencoder: input_channels={input_channels}, "
                   f"output_size={output_size}, feature_dim={feature_dim}, base_channels={base_channels}, "
                   f"dropout_rate={dropout_rate}, use_se={use_se}")
        logger.info(f"   Safe improvements: GroupNorm={USE_GROUP_NORM}, WeightStd={USE_WEIGHT_STANDARDIZATION}")
        
        # Initialize enhanced encoder and decoder  
        self.encoder = CNNEncoder(input_channels, base_channels, feature_dim, dropout_rate, use_se)
        self.decoder = CNNDecoder(feature_dim, output_size, base_channels, use_se)
        
        # Initialize network weights (components handle their own init, but we provide central control)
        self._init_weights()
        
        # Log detailed model characteristics
        self._log_detailed_parameter_breakdown()
    
    def _init_weights(self):
        """
        Central weight initialization with improved schemes.
        
        This provides consistent initialization across the entire autoencoder,
        though individual components also handle their own initialization.
        """
        logger.info("🎯 Applying enhanced weight initialization schemes...")
        
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv3d, WeightStandardizedConv3d, nn.ConvTranspose3d, WeightStandardizedConvTranspose3d)):
                # Kaiming normal for convolutional layers (optimal for ReLU networks)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier uniform for linear layers (optimal for linear operations)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                # Standard normalization layer initialization
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _log_detailed_parameter_breakdown(self):
        """Log comprehensive parameter breakdown for analysis."""
        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        # Calculate by layer type
        conv_params = sum(p.numel() for n, p in self.named_parameters() 
                         if 'conv' in n.lower() and p.requires_grad)
        norm_params = sum(p.numel() for n, p in self.named_parameters() 
                         if any(x in n.lower() for x in ['bn', 'norm']) and p.requires_grad)
        linear_params = sum(p.numel() for n, p in self.named_parameters() 
                           if 'fc' in n.lower() or 'linear' in n.lower() and p.requires_grad)
        
        # Log comprehensive breakdown
        logger.info(f"📊 Enhanced CNN Autoencoder Parameter Analysis:")
        logger.info(f"   ├─ Total Parameters: {total_params:,}")
        logger.info(f"   ├─ Trainable Parameters: {trainable_params:,}")
        logger.info(f"   ├─ Encoder Parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
        logger.info(f"   ├─ Decoder Parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
        logger.info(f"   ├─ Convolutional Layers: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
        logger.info(f"   ├─ Normalization Layers: {norm_params:,} ({norm_params/total_params*100:.1f}%)")
        logger.info(f"   └─ Linear Layers: {linear_params:,} ({linear_params/total_params*100:.1f}%)")
        
        # Verify target parameter count (~7M)
        target_params = 7_000_000
        if abs(total_params - target_params) / target_params > 0.2:  # More than 20% difference
            logger.warning(f"⚠️  Parameter count {total_params:,} differs significantly from target {target_params:,}")
        else:
            logger.info(f"✅ Parameter count {total_params:,} is within target range (~{target_params/1_000_000:.1f}M)")
    
    def forward(self, x):
        """
        Complete forward pass through enhanced autoencoder.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, input_channels, D, H, W)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        # Encode to latent space (bottleneck preserves Stage 2 compatibility)
        encoded = self.encoder(x)
        
        # Decode from latent space
        decoded = self.decoder(encoded)
        
        return decoded
    
    def encode(self, x):
        """
        Encode input volume to compact feature representation.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, input_channels, D, H, W)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, feature_dim)
        """
        return self.encoder(x)
    
    def decode(self, features):
        """
        Decode feature representation back to volume.
        
        Args:
            features (torch.Tensor): Encoded features of shape (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        return self.decoder(features)
    
    def get_encoder(self):
        """Get the enhanced encoder for use in stage 2 training."""
        return self.encoder
    
    def get_decoder(self):
        """Get the enhanced decoder for freezing in stage 2."""
        return self.decoder
    
    def count_parameters(self):
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self):
        """Get comprehensive model summary for logging."""
        return {
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
            'feature_dim': self.encoder.feature_dim,
            'input_channels': INPUT_CHANNELS,
            'output_size': OUTPUT_SIZE,
            'base_channels': BASE_CHANNELS,
            'improvements_enabled': {
                'group_norm': USE_GROUP_NORM,
                'weight_standardization': USE_WEIGHT_STANDARDIZATION,
                'squeeze_excitation': USE_SQUEEZE_EXCITATION,
                'deep_dropout': True
            }
        }
