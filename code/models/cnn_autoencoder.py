"""
CNN Autoencoder following Robin Dale's architecture for stage 1 pre-training.
Focuses on learning low-level spatial features from DOT measurements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow in deeper networks"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNNEncoder(nn.Module):
    """CNN Encoder following Robin Dale's design"""
    
    def __init__(self, input_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=7, 
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks with progressive downsampling
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Global average pooling to create fixed-size feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final feature dimension
        self.feature_dim = base_channels * 8
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)  # Flatten to [batch_size, feature_dim]


class CNNDecoder(nn.Module):
    """CNN Decoder to reconstruct volumes from encoded features"""
    
    def __init__(self, feature_dim: int = 512, output_size: Tuple[int, int, int] = (64, 64, 64),
                 base_channels: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.base_channels = base_channels
        
        # Calculate initial spatial dimensions after upsampling
        self.init_size = 4  # Start with 4x4x4 feature maps
        
        # Linear layer to expand features to initial volume
        self.fc = nn.Linear(feature_dim, base_channels * 8 * (self.init_size ** 3))
        
        # Transpose convolutions for upsampling
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
        
        # Final layer to output single channel volume
        self.final_conv = nn.Conv3d(base_channels // 2, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Expand features to 3D volume
        x = self.fc(x)
        x = x.view(x.size(0), self.base_channels * 8, 
                   self.init_size, self.init_size, self.init_size)
        
        # Progressive upsampling
        x = self.deconv1(x)  # 4x4x4 -> 8x8x8
        x = self.deconv2(x)  # 8x8x8 -> 16x16x16
        x = self.deconv3(x)  # 16x16x16 -> 32x32x32
        x = self.deconv4(x)  # 32x32x32 -> 64x64x64
        
        # Final output
        x = self.final_conv(x)
        
        # Interpolate to exact output size if needed
        if x.shape[2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='trilinear', 
                              align_corners=False)
        
        return x


class CNNAutoEncoder(nn.Module):
    """
    Complete CNN Autoencoder for stage 1 pre-training following Robin Dale's approach.
    Learns low-level spatial feature representations from DOT measurements.
    """
    
    def __init__(self, input_channels: int = 1, 
                 output_size: Tuple[int, int, int] = (64, 64, 64),
                 base_channels: int = 64):
        super().__init__()
        
        self.encoder = CNNEncoder(input_channels, base_channels)
        self.decoder = CNNDecoder(self.encoder.feature_dim, output_size, base_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
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
        """Forward pass through encoder-decoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to feature representation"""
        return self.encoder(x)
    
    def decode(self, features):
        """Decode features to volume"""
        return self.decoder(features)
    
    def get_encoder(self):
        """Get the encoder for use in stage 2 training"""
        return self.encoder
    
    def get_decoder(self):
        """Get the decoder for freezing in stage 2"""
        return self.decoder
