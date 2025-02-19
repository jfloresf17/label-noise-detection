import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Basic U-Net block:
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """
    Downsample: applies MaxPool followed by DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)           # reduces HxW by half
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upsample: upsampling (bilinear) + DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # in_channels is the sum of channels from the skip and the upsampled branch
        # but first we spatially reduce with ConvTranspose2d or Upsample
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: feature map to upsample
        x2: skip connection
        """
        x1 = self.up(x1)    

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel axis
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x   


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        Channel Attention Block where the importance of channels is calculated.

        Args:
            in_channels (int): Number of input channels.
            reduction (int): Reduction factor for the size of hidden layers (MLP).
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # To maintain dimensions
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # To maintain dimensions

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Combine the outputs and apply sigmoid activation
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        """
        Spatial Attention Block where the importance of spatial features is calculated.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Kernel size for the convolution. Must be odd.
        """
        super(SpatialAttention, self).__init__()    
        padding = (kernel_size - 1) // 2  # To maintain dimensions

        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)  # To maintain dimensions
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)  # To maintain dimensions
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()       

    def forward(self, x):
        # Apply average and max pooling along the channel
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # Concatenate the outputs and apply convolution
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Attention Block that combines Channel Attention and Spatial Attention.

        Args:
            in_channels (int): Number of input channels.
            reduction (int): Reduction factor for Channel Attention.
            kernel_size (int): Kernel size for Spatial Attention.
        """
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels, kernel_size)

    def forward(self, x):
        # Apply Channel Attention
        ca = self.channel_attention(x)
        x = x * ca  # Scale features by channel attention

        # Apply Spatial Attention
        sa = self.spatial_attention(x)
        x = x * sa  # Scale features by spatial attention

        return x