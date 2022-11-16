import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_birds_eye_view_parameters

class GridSampler(nn.Module):
    def __init__(self,input_grid_conf,target_grid_conf):
        super().__init__()
        input_resolution, input_start_position, input_dimension = calculate_birds_eye_view_parameters(
                input_grid_conf['xbound'], input_grid_conf['ybound'], input_grid_conf['zbound'],
            )

        target_resolution, target_start_position, target_dimension = calculate_birds_eye_view_parameters(
                target_grid_conf['xbound'], target_grid_conf['ybound'], target_grid_conf['zbound'],
            )

        self.map_x = torch.arange(
                target_start_position[0], target_grid_conf['xbound'][1], target_resolution[0])

        self.map_y = torch.arange(
                target_start_position[1], target_grid_conf['ybound'][1], target_resolution[1])

        # convert to normalized coords
        self.norm_map_x = self.map_x / (- input_start_position[0])
        self.norm_map_y = self.map_y / (- input_start_position[1])

        self.map_grid = torch.stack(torch.meshgrid(
            self.norm_map_x, self.norm_map_y, indexing='xy'), dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        grid = self.map_grid.unsqueeze(0).type_as(x).repeat(x.shape[0], 1, 1, 1)
        return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    confs = {
    'det_grid_conf':{
        'xbound': [-51.2, 51.2, 0.8],
        'ybound': [-51.2, 51.2, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],
    },
    'mot_grid_conf': {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],
    },
    'seg_grid_conf': {
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],
    }
    }
    grid_conf = confs['det_grid_conf']
    input = torch.zeros(4,64,64,64)
    res={}
    for name,conf in confs.items():
        net = GridSampler(grid_conf,conf)
        res[name] = net(input)
    print([i.shape for i in res.values()])
