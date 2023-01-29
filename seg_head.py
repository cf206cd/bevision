import torch
import torch.nn as nn
from utils import SeparateHead
class VanillaSegmentHead(nn.Module):
    def __init__(self,in_channels,output_channels):
        super().__init__()
        self.head = SeparateHead(in_channels, output_channels, 2, True)

    def forward(self,x):
        return self.head(x)

if __name__ == "__main__":
    x = torch.zeros(4,64,200,400)
    net = VanillaSegmentHead(64,10)
    res = net(x)
    print(res.shape)
    jit_model = torch.jit.script(net,x)
    print(jit_model)