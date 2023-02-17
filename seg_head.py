import torch
import torch.nn as nn

class ConvHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs,
        bn=True,
        init_bias=None,
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
                kernel_size=1, stride=1, bias=True))    

        self.conv_layers = nn.Sequential(*conv_lists)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        nn.init.xavier_uniform_(self.conv_layers[-1].weight)
        if init_bias is not None:
            self.conv_layers[-1].bias.data.fill_(init_bias)

    def forward(self, x):   
        ret = self.conv_layers(x)
        return ret
class VanillaSegmentHead(nn.Module):
    def __init__(self,in_channels,output_channels):
        super().__init__()
        self.head = ConvHead(in_channels, output_channels, 2, True, init_bias=-5)

    def forward(self,x):
        return self.head(x)

if __name__ == "__main__":
    x = torch.randn(4,64,200,200)
    net = VanillaSegmentHead(64,10)
    res = net(x)
    print(res.shape)
    jit_model = torch.jit.script(net,x)
    print(jit_model)