"""
CNN Autoencoder for NIR-DOT reconstruction.

This module implements a 3D convolutional autoencoder for near-infrared diffuse optical 
tomography (NIR-DOT) volume reconstruction. The architecture uses residual blocks for 
improved gradient flow and progressive downsampling/upsampling for spatial feature learning.

The autoencoder is designed for stage 1 pre-training in a two-stage hybrid approach,
focusing on learning low-level spatial features from DOT measurements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..utils.logging_config import get_model_logger

# Initialize logger for this module
logger = get_model_logger(__name__)


class ResidualBlock(nn.Module):
    """
    3D Residual block with skip connection.
    
    Implements a residual block with two 3D convolutional layers, batch normalization,
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
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Second convolution maintains spatial dimensions
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection projection when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        logger.debug(f"ResidualBlock initialized: {in_channels}→{out_channels}, stride={stride}")
    
    def forward(self, x):
        """
        Forward pass through residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W')
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNNEncoder(nn.Module):
    """
    3D CNN Encoder for feature extraction from volumetric data.
    
    Progressive downsampling encoder that extracts hierarchical features from 3D volumes.
    Uses residual blocks to maintain gradient flow and batch normalization for stable training.
    The architecture follows a typical CNN pattern with increasing channel depth and 
    decreasing spatial resolution.
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to 1.
        base_channels (int, optional): Base number of channels. Defaults to 64.
    """
    
    def __init__(self, input_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        # Initial feature extraction with aggressive downsampling
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=7, 
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Progressive feature extraction with residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Spatial dimension reduction to fixed-size feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final feature dimension
        self.feature_dim = base_channels * 8
        
        logger.debug(f"CNNEncoder initialized: {input_channels} input channels, "
                    f"{base_channels} base channels, feature_dim={self.feature_dim}")
    
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
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, channels, D, H, W)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, feature_dim)
        """
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)  # Flatten to [batch_size, feature_dim]


class CNNDecoder(nn.Module):
    """
    3D CNN Decoder for volume reconstruction from encoded features.
    
    Progressive upsampling decoder that reconstructs 3D volumes from compact feature
    representations. Uses transposed convolutions for learnable upsampling and skip
    connections for detail preservation.
    
    Args:
        feature_dim (int, optional): Dimension of input features. Defaults to 512.
        output_size (Tuple[int, int, int], optional): Target output volume size. 
            Defaults to (64, 64, 64).
        base_channels (int, optional): Base number of channels. Defaults to 64.
    """
    
    def __init__(self, feature_dim: int = 512, output_size: Tuple[int, int, int] = (64, 64, 64),
                 base_channels: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.base_channels = base_channels
        
        # Calculate initial spatial dimensions for feature map reconstruction
        self.init_size = 4  # Start with 4x4x4 feature maps after linear expansion
        
        # Linear projection to expand feature vector to initial 3D volume
        self.fc = nn.Linear(feature_dim, base_channels * 8 * (self.init_size ** 3))
        
        # Progressive upsampling layers with transposed convolutions
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(base_channels, base_channels // 2, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer to dual channel volume (absorption + scattering)
        self.final_conv = nn.Conv3d(base_channels // 2, 2, kernel_size=3, padding=1)
        
        logger.debug(f"CNNDecoder initialized: feature_dim={feature_dim}, "
                    f"output_size={output_size}, base_channels={base_channels}")
    
    def forward(self, x):
        """
        Forward pass through decoder.
        
        Args:
            x (torch.Tensor): Encoded features of shape (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 1, D, H, W)
        """
        # Expand feature vector to 3D volume
        x = self.fc(x)
        x = x.view(x.size(0), self.base_channels * 8, 
                   self.init_size, self.init_size, self.init_size)
        
        # Progressive upsampling through transposed convolutions
        x = self.deconv1(x)  # 4x4x4 -> 8x8x8
        x = self.deconv2(x)  # 8x8x8 -> 16x16x16
        x = self.deconv3(x)  # 16x16x16 -> 32x32x32
        x = self.deconv4(x)  # 32x32x32 -> 64x64x64
        
        # Generate final single-channel output
        x = self.final_conv(x)
        
        # Ensure exact output dimensions if needed
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
    
    def __init__(self, input_channels: int = 2, 
                 output_size: Tuple[int, int, int] = (60, 60, 60),
                 base_channels: int = 64):
        super().__init__()
        
        logger.info(f"🏗️  Initializing CNN Autoencoder: input_channels={input_channels}, "
                   f"output_size={output_size}, base_channels={base_channels}")
        
        self.encoder = CNNEncoder(input_channels, base_channels)
        self.decoder = CNNDecoder(self.encoder.feature_dim, output_size, base_channels)
        
        # Initialize network weights
        self._init_weights()
        
        # Log model characteristics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"📊 CNN Autoencoder initialized: {total_params:,} total params, "
                   f"{trainable_params:,} trainable params")
    
    def _init_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        Applies Xavier uniform initialization to convolutional and linear layers,
        and standard initialization to batch normalization layers. This helps
        maintain stable gradients during training.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Complete forward pass through autoencoder.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, channels, D, H, W)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 1, D, H, W)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """
        Encode input volume to compact feature representation.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, channels, D, H, W)
            
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
            torch.Tensor: Reconstructed volume of shape (batch_size, 1, D, H, W)
        """
        return self.decoder(features)
    
    def get_encoder(self):
        """Get the encoder for use in stage 2 training"""
        return self.encoder
    
    def get_decoder(self):
        """Get the decoder for freezing in stage 2"""
        return self.decoder
