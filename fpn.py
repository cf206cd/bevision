import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=512, out_ids=[0]):
        super().__init__()
        self.in_channels = in_channels
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.out_ids = out_ids
        for i,in_channel in enumerate(in_channels):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channel,out_channels,1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
            self.lateral_convs.append(lateral_conv)
            if i in out_ids:
                fpn_conv =  nn.Sequential(
                    nn.Conv2d(out_channels,out_channels,3,padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU())
                self.fpn_convs.append(fpn_conv)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, inputs:List[torch.Tensor]):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(laterals)-1,0,-1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] = F.interpolate(laterals[i], size=prev_shape, mode="bilinear")+laterals[i-1]
        outs = [
            fpn_conv(laterals[self.out_ids[i]])
            for i,fpn_conv in enumerate(self.fpn_convs)]        
        return outs

if __name__ == "__main__":
    x1 = [torch.randn(6,240,64,36),
            torch.randn(6,720,32,18),
            torch.randn(6,1920,16,9)]
    net1 = FPN(in_channels=[240, 720, 1920],out_channels=64)
    output1 = net1(x1)
    print([i.shape for i in output1])
    jit_model1 = torch.jit.script(net1,x1)
    print(jit_model1)

    x2 = [torch.randn(4,152,100,100),
            torch.randn(4,368,50,50)]
    net2 = FPN(in_channels=[152,368],out_channels=64)
    output2 = net2(x2)
    print([i.shape for i in output2])
    jit_model2 = torch.jit.script(net2,x2)
    print(jit_model2)