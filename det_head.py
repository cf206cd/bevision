import torch
import torch.nn as nn
from utils import SeparateHead
class CenterPointHead(nn.Module):
    """CenterHead for CenterPoint."""
    def __init__(self,in_channels,num_classes,
                num_task_channel=5,#offset 2,volume 2,rotation 1,all 5
                shared_conv_channel=64,
                num_shared_convs=2,
                num_seperate_convs=2,
                bn=True,
                init_bias=-2.19):#heat map focal loss init trick
        super().__init__()
        self.shared_conv = SeparateHead(in_channels,shared_conv_channel,num_shared_convs,bn)
        self.heatmap_head = SeparateHead(shared_conv_channel, num_classes, num_seperate_convs, bn, init_bias=init_bias)
        self.regression_head = SeparateHead(shared_conv_channel, num_task_channel, num_seperate_convs, bn)

    def forward(self,x):
        x = self.shared_conv(x)
        heatmap = self.heatmap_head(x)
        regression  = self.regression_head(x)
        return heatmap,regression

if __name__ == "__main__":
    x = torch.zeros(4,64,128,128)
    net = CenterPointHead(64,10)
    res = net(x)
    print(res[0].shape,res[1].shape)