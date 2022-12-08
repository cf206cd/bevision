import torch
import torch.nn as nn
class SeparateHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs,
        bn=False,
        init_bias=-2.19,
        head_channels=64
    ):
        super().__init__()

        conv_lists = []
        c_in = in_channels
        for i in range(num_convs-1):
            conv_lists.append(nn.Conv2d(c_in, head_channels,
                kernel_size=3, stride=1, 
                padding=3 // 2, bias=True))
            if bn:
                conv_lists.append(nn.BatchNorm2d(head_channels))
            conv_lists.append(nn.ReLU())
            c_in = head_channels
        
        conv_lists.append(nn.Conv2d(head_channels, out_channels,
                kernel_size=3, stride=1, 
                padding=3 // 2, bias=True))    

        self.conv_layers = nn.Sequential(*conv_lists)
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if init_bias is not None:
            self.conv_layers[-1].bias.data.fill_(init_bias)

    def forward(self, x):     
        ret = self.conv_layers(x)
        return ret

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