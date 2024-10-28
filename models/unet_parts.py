""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Attention(nn.Module):
    """Attention Layer"""
    def __init__(self, x1_channels, x2_channels):
        super().__init__()
        # Convolutions to reduce channels of x1 and x2
        self.x1_conv = nn.Conv2d(x1_channels, x2_channels, kernel_size=1)
        self.x2_conv = nn.Conv2d(x2_channels, x2_channels, kernel_size=1)

        # Psi layer (1x1 convolution) followed by sigmoid
        self.psi = nn.Conv2d(x2_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
            # Apply convolution to reduce channels
            g1 = self.x1_conv(x1)  # Transform x1
            x2_ = self.x2_conv(x2)  # Transform x2
            
            # Add them and apply ReLU activation
            attention = F.relu(g1 + x2_)
            
            # Apply psi (1x1 convolution) and sigmoid
            psi = self.psi(attention)
            attention = self.sigmoid(psi)
            
            # Multiply the attention map with x2 (element-wise)
            x2 = x2 * attention
            return x2
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + out_channels, out_channels)
        self.attention = Attention(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)