import torch
import torch.nn as nn

class PredictHead(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        self.predict_future_flow = predict_future_flow
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)

        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None
        return {
            'segmentation': segmentation_output.view(b, s, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, s, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, s, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, s, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

if __name__ == "__main__":
    x = torch.zeros(4,7,64,200,200)
    net = PredHead(64,4,True)
    res = net(x)
    print([(i[0],i[1].shape) for i in res.items()])