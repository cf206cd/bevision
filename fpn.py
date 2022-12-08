import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=512, out_ids=[0]):
        super().__init__()
        self.in_channels = in_channels
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.out_ids = out_ids
        for _,in_channel in enumerate(in_channels):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channel,out_channels,1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
            self.lateral_convs.append(lateral_conv)
            if _ in out_ids:
                fpn_conv =  nn.Sequential(
                    nn.Conv2d(out_channels,out_channels,3,padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU())
                self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(laterals)-1,0,-1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = F.interpolate(laterals[i], size=prev_shape)+laterals[i - 1]
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        return outs

if __name__ == "__main__":
    input1 = [torch.zeros(6,56,80,80),
            torch.zeros(6,152,40,40),
            torch.zeros(6,368,20,20)]
    net1 = FPN(in_channels=[56,152,368],out_channels=64)
    output1 = net1(input1)
    print([i.shape for i in output1])

    input2 = [torch.zeros(4,152,64,64),
            torch.zeros(4,368,32,32)]
    net2 = FPN(in_channels=[152,368],out_channels=64,out_ids=[0])
    output2 = net2(input2)
    print([i.shape for i in output2])