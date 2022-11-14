import torch
import torch.nn as nn
import torch.nn.functional as F

class LSSFPN(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4,
                 extra_upsample=2,
                 lateral=None):
        super().__init__()
        self.extra_upsample = extra_upsample is not None
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample , mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral=  lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral,
                          kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        x2, x1 = feats
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = F.interpolate(x1,size=x2.shape[2:])
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.extra_upsample:
            x = self.up2(x)
        return x