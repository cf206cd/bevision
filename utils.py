import numpy as np
import torch.nn as nn

def generate_grid(row):
    grid = np.linspace(*row,endpoint=False, dtype=np.float32)+(row[1]-row[0])/row[2]*0.5
    return grid

def generate_step(rows):
    start = np.array([row[0] for row in rows], dtype=np.float32)
    interval = np.array([(row[1]-row[0])/row[2] for row in rows], dtype=np.float32)
    count = np.array([row[2] for row in rows], dtype=np.float32)
    return start,interval,count

class SeparateHead(nn.Module):
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