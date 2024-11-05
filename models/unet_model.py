
import torch
import torch.nn as nn
from models.unet_parts import ResidualConv, Upsample, AttentionBlock

"""
__title__ = "Road Extraction by Deep Residual U-Net"
__author__ = "Zhengxin Zhang, Qingjie Liu &  Yunhong Wang"
__paper__ = "https://arxiv.org/pdf/1711.10684"
"""

class ResUnetTeacher(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256]):
        super(ResUnetTeacher, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.attention_1 =AttentionBlock(filters[2], filters[3], filters[3])
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.attention_2 = AttentionBlock(filters[1], filters[2], filters[2])
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.attention_3 = AttentionBlock(filters[0], filters[1], filters[1])
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1)
        )

    def forward(self, x):
        # Encode
        d_x1 = self.input_layer(x) + self.input_skip(x)
        d_x2 = self.residual_conv_1(d_x1)
        d_x3 = self.residual_conv_2(d_x2)
        
        # Bridge
        d_x4 = self.bridge(d_x3)


        # Decode
        u_x4 = self.upsample_1(d_x4)
        a_x3 = self.attention_1(d_x3, u_x4)
        c_x3 = torch.cat([a_x3, d_x3], dim=1)

        r_x5 = self.up_residual_conv1(c_x3)

        u_x2 = self.upsample_2(r_x5)
        a_x2 = self.attention_2(d_x2, u_x2)
        c_x2 = torch.cat([a_x2, d_x2], dim=1)

        r_x6 = self.up_residual_conv2(c_x2)

        u_x1 = self.upsample_3(r_x6)
        a_x1 = self.attention_3(d_x1, u_x1)
        c_x1 = torch.cat([a_x1, d_x1], dim=1)

        r_x7 = self.up_residual_conv3(c_x1)

        output = self.output_layer(r_x7)

        return output, d_x4


class ResUnetStudent(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256]):
        super(ResUnetStudent, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.attention_1 =AttentionBlock(filters[2], filters[3], filters[3])
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.attention_2 = AttentionBlock(filters[1], filters[2], filters[2])
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.attention_3 = AttentionBlock(filters[0], filters[1], filters[1])
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1)
        )

    def forward(self, x):
        # Encode
        d_x1 = self.input_layer(x) + self.input_skip(x)
        d_x2 = self.residual_conv_1(d_x1)
        d_x3 = self.residual_conv_2(d_x2)
        
        # Bridge
        d_x4 = self.bridge(d_x3)


        # Decode
        u_x4 = self.upsample_1(d_x4)
        a_x3 = self.attention_1(d_x3, u_x4)
        c_x3 = torch.cat([a_x3, d_x3], dim=1)

        r_x5 = self.up_residual_conv1(c_x3)

        u_x2 = self.upsample_2(r_x5)
        a_x2 = self.attention_2(d_x2, u_x2)
        c_x2 = torch.cat([a_x2, d_x2], dim=1)

        r_x6 = self.up_residual_conv2(c_x2)

        u_x1 = self.upsample_3(r_x6)
        a_x1 = self.attention_3(d_x1, u_x1)
        c_x1 = torch.cat([a_x1, d_x1], dim=1)

        r_x7 = self.up_residual_conv3(c_x1)

        output = self.output_layer(r_x7)

        return output, d_x4