
import torch.nn as nn
from models.unet_parts import DoubleConv, Down, Up, AttentionBlock

"""
__title__ = "NRN-RSSEG: A Deep Neural Network Model for Combating Label Noise in Semantic Segmentation of Remote Sensing"
__author__ = "Xi, M., Li, J., He, Z., Yu, M., & Qin, F. (2023)"
__paper__ = "https://doi.org/10.3390/rs15010108"
"""

class NRNRSSEGTeacher(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        """
        Modified UNet Teacher with Attention Block (CBAM: Channel and Spatial Attention Blocks) 
        concatenating the skip connections.
        """
        super(NRNRSSEGTeacher, self).__init__()

        # ENCODER 1
        self.inc = DoubleConv(in_channels, base_filters)        # [3 -> 32]
        self.down1 = Down(base_filters, base_filters * 2)       # [32 -> 64]

        # ENCODER 2
        self.down2 = Down(base_filters * 2, base_filters * 4)   # [64 -> 128]

        # ENCODER 3
        self.down3 = Down(base_filters * 4, base_filters * 8)   # [128 -> 256]

        # BRIDGE
        self.bridge = DoubleConv(base_filters * 8, base_filters * 16)  # [256 -> 512]

        # DECODER 1
        self.up1 = Up(base_filters * 16, base_filters * 8)       # [512 -> 256]
        self.att1 = AttentionBlock(in_channels= base_filters * 8, # concat with skip=256 => total=512 -> conv->256
                                   reduction=16, 
                                   kernel_size=7)
        # DECODER 2
        self.up2 = Up(base_filters * 8, base_filters * 4)       # [256 -> 128]
        self.att2 = AttentionBlock(in_channels= base_filters * 4, # concat with skip=128 => total=256 -> conv->128
                                   reduction=16, 
                                   kernel_size=7)
        # DECODER 3
        self.up3 = Up(base_filters * 4, base_filters * 2)       # [128 -> 64] 
        self.att3 = AttentionBlock(in_channels= base_filters * 2, # concat with skip=128 => total=256 -> conv->128
                                   reduction=16, 
                                   kernel_size=7)
        # DECODER 4
        self.up4 = Up(base_filters * 2, base_filters)           # [64 -> 32] 
        self.att4 = AttentionBlock(in_channels= base_filters,  # concat with skip=64 => total=128 -> conv->64
                                   reduction=16, 
                                   kernel_size=7)
        # OUTPUT
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # ====== ENCODER ======
        x1 = self.inc(x)        # [B, 32,  H, W]
        x2 = self.down1(x1)     # [B, 64,  H/2, W/2]
        x3 = self.down2(x2)     # [B, 128, H/4, W/4]
        x4 = self.down3(x3)     # [B, 256, H/8, W/8]
        x5 = self.bridge(x4)    # [B, 512, H/16, W/16]
        
        # 1) up1
        up1_att = self.att1(x4)       # [B, 256, H/8, W/8]
        up1 = self.up1(x5, up1_att)         # [B, 256, H/8, W/8]
              

        # 2) up2
        up2_att = self.att2(x3)        # [B, 128, H/4, W/4]
        up2 = self.up2(up1, up2_att)    # [B, 128, H/4, W/4]
              

        # 3) up3
        up3_att = self.att3(x2)        # [B, 64, H/2, W/2]
        up3 = self.up3(up2, up3_att)    # [B, 64, H/2, W/2]

        # 4) up4
        up4_att = self.att4(x1)        # [B, 32, H, W]
        up4 = self.up4(up3, up4_att)    # [B, 32, H, W]

        # OUTPUT
        logits = self.outc(up4)    # [B, 1, H, W]
        
        return logits, x5



class NRNRSSEGStudent(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        """
        Ejemplo simple de U-Net con 4 niveles + CBAM en los bloques de decodificación.
        """
        super(NRNRSSEGStudent, self).__init__()

        # ENCODER 1
        self.inc = DoubleConv(in_channels, base_filters)        # [3 -> 32]
        self.down1 = Down(base_filters, base_filters * 2)       # [32 -> 64]

        # ENCODER 2
        self.down2 = Down(base_filters * 2, base_filters * 4)   # [64 -> 128]

        # ENCODER 3
        self.down3 = Down(base_filters * 4, base_filters * 8)   # [128 -> 256]

        # BRIDGE
        self.bridge = DoubleConv(base_filters * 8, base_filters * 16)  # [256 -> 512]

        # DECODER 1
        self.up1 = Up(base_filters * 16, base_filters * 8)       # [512 -> 256]
        self.att1 = AttentionBlock(in_channels= base_filters * 8, # concat with skip=256 => total=512 -> conv->256
                                   reduction=16, 
                                   kernel_size=7)
        # DECODER 2
        self.up2 = Up(base_filters * 8, base_filters * 4)       # [256 -> 128]
        self.att2 = AttentionBlock(in_channels= base_filters * 4, # concat with skip=128 => total=256 -> conv->128
                                   reduction=16, 
                                   kernel_size=7)
        # DECODER 3
        self.up3 = Up(base_filters * 4, base_filters * 2)       # [128 -> 64] 
        self.att3 = AttentionBlock(in_channels= base_filters * 2, # concat with skip=128 => total=256 -> conv->128
                                   reduction=16, 
                                   kernel_size=7)
        # DECODER 4
        self.up4 = Up(base_filters * 2, base_filters)           # [64 -> 32] 
        self.att4 = AttentionBlock(in_channels= base_filters,  # concat with skip=64 => total=128 -> conv->64
                                   reduction=16, 
                                   kernel_size=7)
        # OUTPUT
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # ====== ENCODER ======
        x1 = self.inc(x)        # [B, 32,  H, W]
        x2 = self.down1(x1)     # [B, 64,  H/2, W/2]
        x3 = self.down2(x2)     # [B, 128, H/4, W/4]
        x4 = self.down3(x3)     # [B, 256, H/8, W/8]
        x5 = self.bridge(x4)    # [B, 512, H/16, W/16]
        
        # 1) up1
        up1_att = self.att1(x4)       # [B, 256, H/8, W/8]
        up1 = self.up1(x5, up1_att)         # [B, 256, H/8, W/8]
              

        # 2) up2
        up2_att = self.att2(x3)        # [B, 128, H/4, W/4]
        up2 = self.up2(up1, up2_att)    # [B, 128, H/4, W/4]
              

        # 3) up3
        up3_att = self.att3(x2)        # [B, 64, H/2, W/2]
        up3 = self.up3(up2, up3_att)    # [B, 64, H/2, W/2]

        # 4) up4
        up4_att = self.att4(x1)        # [B, 32, H, W]
        up4 = self.up4(up3, up4_att)    # [B, 32, H, W]

        # OUTPUT
        logits = self.outc(up4)    # [B, 1, H, W]
        
        return logits, x5
