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
import sys
import os

# Add parent directories to path for logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_config import get_model_logger

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
    
    def __init__(self, input_channels: int = 1, base_channels: int = 64, feature_dim: int = 512):
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
        
        # Configurable feature dimension with linear projection
        self.feature_dim = feature_dim
        self.feature_projection = nn.Linear(base_channels * 8, feature_dim)
        
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
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, base_channels * 8]
        x = self.feature_projection(x)  # Project to configurable feature dimension
        return x


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
    
    def __init__(self, feature_dim: int = 512, output_size: Tuple[int, int, int] = (60, 60, 60),
                 base_channels: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.base_channels = base_channels
        
        # Calculate proper initial spatial dimensions to match encoder
        # Encoder path: 60 -> 30 (stride=2) -> 15 (maxpool stride=2) -> 15 -> 8 -> 4 -> 2
        # So decoder should start from 2x2x2 to be symmetric
        self.init_size = 2
        
        # Linear projection to expand feature vector to initial 3D volume
        self.fc = nn.Linear(feature_dim, base_channels * 8 * (self.init_size ** 3))
        
        # Progressive upsampling layers with transposed convolutions
        # Symmetric to encoder: 2->4->8->16->32->60 (with final adjustment)
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
        
        # Additional layer to get closer to target size (32->64)
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(base_channels // 2, base_channels // 4, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer to dual channel volume (absorption + scattering)
        self.final_conv = nn.Conv3d(base_channels // 4, 2, kernel_size=3, padding=1)
        
        logger.debug(f"CNNDecoder initialized: feature_dim={feature_dim}, "
                    f"output_size={output_size}, base_channels={base_channels}")
    
    def forward(self, x):
        """
        Forward pass through decoder.
        
        Args:
            x (torch.Tensor): Encoded features of shape (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        # Expand feature vector to 3D volume starting from 2x2x2 (matching encoder)
        x = self.fc(x)  # [batch, feature_dim] -> [batch, base_channels*8*8]
        x = x.view(x.size(0), self.base_channels * 8, 
                   self.init_size, self.init_size, self.init_size)  # [batch, 512, 2, 2, 2]
        
        # Progressive upsampling through transposed convolutions
        x = self.deconv1(x)  # 2x2x2 -> 4x4x4, channels: 512->256
        x = self.deconv2(x)  # 4x4x4 -> 8x8x8, channels: 256->128
        x = self.deconv3(x)  # 8x8x8 -> 16x16x16, channels: 128->64
        x = self.deconv4(x)  # 16x16x16 -> 32x32x32, channels: 64->32
        x = self.deconv5(x)  # 32x32x32 -> 64x64x64, channels: 32->16
        
        # Generate final dual-channel output (μₐ + μ′s)
        x = self.final_conv(x)  # 64x64x64 -> 64x64x64, channels: 16->2
        
        # Ensure exact output dimensions match target (64x64x64 -> 60x60x60)
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
                 feature_dim: int = 512,
                 base_channels: int = 64):
        super().__init__()
        
        logger.info(f"🏗️  Initializing CNN Autoencoder: input_channels={input_channels}, "
                   f"output_size={output_size}, feature_dim={feature_dim}, base_channels={base_channels}")
        
        self.encoder = CNNEncoder(input_channels, base_channels, feature_dim)
        self.decoder = CNNDecoder(feature_dim, output_size, base_channels)
        
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Complete forward pass through autoencoder.
        
        Args:
            x (torch.Tensor): Input volume of shape (batch_size, channels, D, H, W)
            
        Returns:
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
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
            torch.Tensor: Reconstructed volume of shape (batch_size, 2, D, H, W)
        """
        return self.decoder(features)
    
    def get_encoder(self):
        """Get the encoder for use in stage 2 training"""
        return self.encoder
    
    def get_decoder(self):
        """Get the decoder for freezing in stage 2"""
        return self.decoder


def test_cnn_autoencoder():
    """Test function to verify the autoencoder works correctly"""
    print("🧪 Testing CNN Autoencoder...")
    
    # Create model
    model = CNNAutoEncoder(
        input_channels=2,
        output_size=(60, 60, 60),
        feature_dim=512,
        base_channels=64
    )
    
    # Test with sample input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 2, 60, 60, 60)
    
    # Test encoding
    encoded = model.encode(input_tensor)
    print(f"✅ Encoding: {input_tensor.shape} → {encoded.shape}")
    
    # Test decoding
    decoded = model.decode(encoded)
    print(f"✅ Decoding: {encoded.shape} → {decoded.shape}")
    
    # Test full forward pass
    output = model(input_tensor)
    print(f"✅ Full pass: {input_tensor.shape} → {output.shape}")
    
    # Verify shapes
    assert encoded.shape == (batch_size, 512), f"Encoded shape wrong: {encoded.shape}"
    assert decoded.shape == (batch_size, 2, 60, 60, 60), f"Decoded shape wrong: {decoded.shape}"
    assert output.shape == (batch_size, 2, 60, 60, 60), f"Output shape wrong: {output.shape}"
    
    print("🎉 All tests passed!")
    return True


if __name__ == "__main__":
    test_cnn_autoencoder()
