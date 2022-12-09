import torch
import torch.nn as nn

class VanillaSegmentHead(nn.Module):
    def __init__(self,in_channels,output_channels):
        super().__init__()
        self.head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, output_channels, kernel_size=1, padding=0)
            )
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        return self.head(x)

if __name__ == "__main__":
    x = torch.zeros(4,64,200,400)
    net = VanillaSegmentHead(64,10)
    res = net(x)
    print(res.shape)