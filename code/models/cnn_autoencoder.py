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

# Initialize module logger
logger = get_model_logger(__name__)

# =============================================================================
# NETWORK COMPONENTS
# =============================================================================


class ResidualBlock(nn.Module):
    """
    3D Residual block with skip connection using GroupNorm.
    
    Implements a residual block with two 3D convolutional layers, group normalization,
    and ReLU activation. The skip connection helps prevent vanishing gradients and 
    improves training stability in deeper networks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Convolution stride. Defaults to 1.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # First convolution with potential downsampling
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=RESIDUAL_CONV_KERNEL, 
                               stride=stride, padding=RESIDUAL_CONV_PADDING, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        # Second convolution maintains spatial dimensions
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=RESIDUAL_CONV_KERNEL,
                               stride=1, padding=RESIDUAL_CONV_PADDING, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        # Skip connection projection when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=RESIDUAL_SHORTCUT_KERNEL, 
                          stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
        
    
    def forward(self, x):
        """
        Forward pass through residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W')
        """
        
        out = F.relu(self.gn1(self.conv1(x)))
        
        out = self.gn2(self.conv2(out))
        
        shortcut = self.shortcut(x)
        
        out += shortcut
        out = F.relu(out)
        
        return out


class CNNEncoder(nn.Module):
    """
    3D CNN Encoder for feature extraction from volumetric data with skip connections.
    
    Progressive downsampling encoder that extracts hierarchical features from 3D volumes.
    Uses residual blocks to maintain gradient flow and group normalization for stable training.
    The architecture follows a typical CNN pattern with increasing channel depth and 
    decreasing spatial resolution. Returns skip connection features for decoder integration.
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to 1.
        base_channels (int, optional): Base number of channels. Defaults to 64.
    """
    
    def __init__(self, input_channels: int = INPUT_CHANNELS, 
                 base_channels: int = BASE_CHANNELS, 
                 feature_dim: int = FEATURE_DIM,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Initial feature extraction with aggressive downsampling
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=ENCODER_KERNEL_INITIAL, 
                      stride=ENCODER_STRIDE_INITIAL, padding=ENCODER_PADDING_INITIAL, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=ENCODER_MAXPOOL_KERNEL, stride=ENCODER_MAXPOOL_STRIDE, 
                        padding=ENCODER_MAXPOOL_PADDING)
        )
        
        # Progressive feature extraction with residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=1)
        self.layer2 = self._make_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                                     base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=2)
        self.layer3 = self._make_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                                     base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=2)
        self.layer4 = self._make_layer(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                                     base_channels * ENCODER_CHANNEL_MULTIPLIERS[4], 
                                     RESIDUAL_BLOCKS_PER_LAYER, stride=2)
        
        # Spatial dimension reduction to fixed-size feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Configurable feature dimension with linear projection
        self.feature_dim = feature_dim
        self.feature_projection = nn.Linear(base_channels * 16, feature_dim)  # 256 → 256 for base_channels=16
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int):
        """
        Create a layer of residual blocks.
        
        Args:
            in_channels (int): Input channels for the first block
            out_channels (int): Output channels for all blocks
            num_blocks (int): Number of residual blocks in the layer
            stride (int): Stride for the first block (subsequent blocks use stride=1)
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, return_skip_connections=False):
        """
        Forward pass through encoder with optional skip connections.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, channels, D, H, W)
            return_skip_connections (bool): Whether to return skip connection features
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, feature_dim)
            or tuple: (features, skip_connections) if return_skip_connections=True
        """
        
        x = self.initial_conv(x)
        
        skip1 = self.layer1(x)  # After layer1 - channels: 16→32
        
        skip2 = self.layer2(skip1)  # After layer2 - channels: 32→64, spatial: /2
        
        skip3 = self.layer3(skip2)  # After layer3 - channels: 64→128, spatial: /2
        
        skip4 = self.layer4(skip3)  # After layer4 - channels: 128→256, spatial: /2
        
        x = self.global_avg_pool(skip4)
        
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, base_channels * 16]
        
        x = self.dropout(x)  # Apply dropout for regularization
        
        features = self.feature_projection(x)  # Project to configurable feature dimension
        
        if return_skip_connections:
            skip_connections = {
                'skip1': skip1,  # 32 channels
                'skip2': skip2,  # 64 channels  
                'skip3': skip3,  # 128 channels
                'skip4': skip4   # 256 channels
            }
            return features, skip_connections
        
        return features


class CNNDecoder(nn.Module):
    """
    3D CNN Decoder for volume reconstruction with skip connections and refinement head.
    
    Progressive upsampling decoder that reconstructs 3D volumes from compact feature
    representations with skip connections from encoder for detail preservation.
    Uses transposed convolutions for learnable upsampling, group normalization,
    and a refinement head for high-quality output.
    
    Args:
        feature_dim (int, optional): Dimension of input features. Defaults to 512.
        output_size (Tuple[int, int, int], optional): Target output volume size. 
            Defaults to (64, 64, 64).
        base_channels (int, optional): Base number of channels. Defaults to 64.
    """
    
    def __init__(self, feature_dim: int = FEATURE_DIM, 
                 output_size: Tuple[int, int, int] = OUTPUT_SIZE,
                 base_channels: int = BASE_CHANNELS):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.base_channels = base_channels
        
        # Calculate proper initial spatial dimensions to match encoder
        # Encoder path: 64 -> 32 (stride=2) -> 16 (maxpool stride=2) -> 16 -> 8 -> 4 -> 2
        # So decoder should start from 2x2x2 to be symmetric
        self.init_size = DECODER_INIT_SIZE
        
        # Linear projection to expand feature vector to initial 3D volume
        self.fc = nn.Linear(feature_dim, base_channels * ENCODER_CHANNEL_MULTIPLIERS[4] * (self.init_size ** 3))
        
        # Skip connection projection layers (1x1 conv to match channels if needed)
        self.skip4_proj = nn.Conv3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[4], 
                                   base_channels * ENCODER_CHANNEL_MULTIPLIERS[4], 
                                   kernel_size=1)  # 256→256 (identity)
        self.skip3_proj = nn.Conv3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                                   base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                                   kernel_size=1)  # 128→128 (identity)
        self.skip2_proj = nn.Conv3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                                   base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                                   kernel_size=1)  # 64→64 (identity)
        self.skip1_proj = nn.Conv3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                                   base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                                   kernel_size=1)  # 32→32 (identity)
        
        # Progressive upsampling layers with transposed convolutions and skip connections
        # Handle both skip connection and no-skip cases
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[4],  # 256 input (no skip concat)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[3]),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection branch for deconv1
        self.deconv1_with_skip = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[4] * 2,  # 512 input (with skip concat)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[3], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[3]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3],  # 128 input (no skip)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[2]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2_with_skip = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[3] * 2,  # 256 input (with skip)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[2], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[2]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2],  # 64 input (no skip)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[1]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3_with_skip = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[2] * 2,  # 128 input (with skip)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[1], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[1]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1],  # 32 input (no skip)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[0], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[0]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4_with_skip = nn.Sequential(
            nn.ConvTranspose3d(base_channels * ENCODER_CHANNEL_MULTIPLIERS[1] * 2,  # 64 input (with skip)
                               base_channels * ENCODER_CHANNEL_MULTIPLIERS[0], 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * ENCODER_CHANNEL_MULTIPLIERS[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to reach 64x64x64 and reduce to 16 channels
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(base_channels, 16, 
                               kernel_size=DECODER_TRANSCONV_KERNEL, stride=DECODER_TRANSCONV_STRIDE, 
                               padding=DECODER_TRANSCONV_PADDING),
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.ReLU(inplace=True)
        )
        
        # Refinement head for high-quality output (replaces final_conv)
        self.refinement_head = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, INPUT_CHANNELS, kernel_size=1)
        )
        
        logger.info(f"📊 CNNDecoder initialized with skip connections and refinement head: "
                    f"feature_dim={feature_dim}, output_size={output_size}, base_channels={base_channels}")
    
    def forward(self, x, skip_connections=None):
        """
        Forward pass through decoder with optional skip connections.
        
        Args:
            x (torch.Tensor): Encoded features of shape (batch_size, feature_dim)
            skip_connections (dict, optional): Skip connection features from encoder
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        
        # Expand feature vector to 3D volume starting from 2x2x2 (matching encoder)
        x = self.fc(x)  # [batch, feature_dim] -> [batch, base_channels*8*8]
        
        x = x.view(x.size(0), self.base_channels * ENCODER_CHANNEL_MULTIPLIERS[4], 
                   self.init_size, self.init_size, self.init_size)  # [batch, 256, 2, 2, 2] for base_channels=16
        
        # Progressive upsampling through transposed convolutions with skip connections
        
        # Deconv1: 2x2x2 -> 4x4x4, channels: 256->128 (with/without skip4 connection)
        if skip_connections and 'skip4' in skip_connections:
            skip4 = self.skip4_proj(skip_connections['skip4'])
            x = torch.cat([x, skip4], dim=1)  # Concatenate along channel dimension: 256+256=512
            x = self.deconv1_with_skip(x)  # Input: 512 -> Output: 128
        else:
            x = self.deconv1(x)  # Input: 256 -> Output: 128
        
        # Deconv2: 4x4x4 -> 8x8x8, channels: 128->64 (with/without skip3 connection) 
        if skip_connections and 'skip3' in skip_connections:
            skip3 = self.skip3_proj(skip_connections['skip3'])
            x = torch.cat([x, skip3], dim=1)  # Concatenate along channel dimension: 128+128=256
            x = self.deconv2_with_skip(x)  # Input: 256 -> Output: 64
        else:
            x = self.deconv2(x)  # Input: 128 -> Output: 64
        
        # Deconv3: 8x8x8 -> 16x16x16, channels: 64->32 (with/without skip2 connection)
        if skip_connections and 'skip2' in skip_connections:
            skip2 = self.skip2_proj(skip_connections['skip2'])
            x = torch.cat([x, skip2], dim=1)  # Concatenate along channel dimension: 64+64=128
            x = self.deconv3_with_skip(x)  # Input: 128 -> Output: 32
        else:
            x = self.deconv3(x)  # Input: 64 -> Output: 32
        
        # Deconv4: 16x16x16 -> 32x32x32, channels: 32->16 (with/without skip1 connection)
        if skip_connections and 'skip1' in skip_connections:
            skip1 = self.skip1_proj(skip_connections['skip1'])
            x = torch.cat([x, skip1], dim=1)  # Concatenate along channel dimension: 32+32=64
            x = self.deconv4_with_skip(x)  # Input: 64 -> Output: 16
        else:
            x = self.deconv4(x)  # Input: 32 -> Output: 16
        
        # Deconv5: 32x32x32 -> 64x64x64, channels: 16->16 (no skip connection)
        x = self.deconv5(x)
        
        # Apply refinement head for high-quality final output
        x = self.refinement_head(x)  # 64x64x64 -> 64x64x64, channels: 16->2
        
        # Ensure exact output dimensions match target (should be exactly 64x64x64)
        if x.shape[2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='trilinear', 
                              align_corners=False)
        
        return x


class CNNAutoEncoder(nn.Module):
    """
    Complete 3D CNN Autoencoder for volumetric reconstruction.
    
    End-to-end autoencoder architecture combining CNN encoder and decoder for 
    learning compact representations of 3D volumes. Designed for stage 1 
    pre-training in a multi-stage learning approach, focusing on low-level 
    spatial feature extraction and reconstruction.
    
    The encoder progressively downsamples the input volume to a compact feature
    representation, while the decoder reconstructs the original volume from these
    features using learnable upsampling.
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to 1.
        output_size (Tuple[int, int, int], optional): Target output volume size. 
            Defaults to (64, 64, 64).
        base_channels (int, optional): Base number of channels for the network. 
            Defaults to 64.
    """
    
    def __init__(self, input_channels: int = INPUT_CHANNELS, 
                 output_size: Tuple[int, int, int] = OUTPUT_SIZE,
                 feature_dim: int = FEATURE_DIM,
                 base_channels: int = BASE_CHANNELS,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        logger.info(f"🏗️  Initializing CNN Autoencoder: input_channels={input_channels}, "
                   f"output_size={output_size}, feature_dim={feature_dim}, base_channels={base_channels}, "
                   f"dropout_rate={dropout_rate}")
        
        self.encoder = CNNEncoder(input_channels, base_channels, feature_dim, dropout_rate)
        self.decoder = CNNDecoder(feature_dim, output_size, base_channels)
        
        # Initialize network weights
        self._init_weights()
        
        # Log model characteristics
        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        logger.info(f"📊 CNN Autoencoder initialized: {total_params:,} total params")
        logger.info(f"   ├─ Encoder: {encoder_params:,} params")
        logger.info(f"   ├─ Decoder: {decoder_params:,} params")
        logger.info(f"   └─ All trainable: {trainable_params:,} params")
    
    def _init_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        Applies Xavier uniform initialization to convolutional and linear layers,
        and standard initialization to group normalization layers. This helps
        maintain stable gradients during training.
        """
        conv_count = 0
        gn_count = 0
        linear_count = 0
        
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(m.weight)
                conv_count += 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                gn_count += 1
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                linear_count += 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        logger.debug(f"🎲 Initialized weights: {conv_count} conv layers, {gn_count} GroupNorm layers, {linear_count} linear layers")
        
    
    def forward(self, x):
        """
        Complete forward pass through autoencoder with skip connections.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, channels, D, H, W)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        
        # Encode with skip connections
        encoded, skip_connections = self.encoder(x, return_skip_connections=True)
        
        # Decode with skip connections
        decoded = self.decoder(encoded, skip_connections)
        
        return decoded
    
    def encode(self, x):
        """
        Encode input volume to compact feature representation.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, channels, D, H, W)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, feature_dim)
        """
        features = self.encoder(x, return_skip_connections=False)  # Only return features for latent interface
        return features
    
    def decode(self, features):
        """
        Decode feature representation back to volume (without skip connections for Stage 2 compatibility).
        
        Args:
            features (torch.Tensor): Encoded features of shape (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        volume = self.decoder(features, skip_connections=None)  # No skip connections for Stage 2
        return volume
    
    def get_encoder(self):
        """Get the encoder for use in stage 2 training"""
        return self.encoder
    
    def get_decoder(self):
        """Get the decoder for freezing in stage 2"""
        return self.decoder
