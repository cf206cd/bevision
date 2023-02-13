import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_grid,generate_step

class GridSampler(nn.Module):
    def __init__(self,input_grid_conf,target_grid_conf):
        super().__init__()

        self.mesh_x = generate_grid(target_grid_conf['xbound'])
        self.mesh_y = generate_grid(target_grid_conf['ybound'])

        #convert to normalized coords
        self.norm_mesh_x = torch.tensor((self.mesh_x-input_grid_conf['xbound'][0]) / (input_grid_conf['xbound'][1]-input_grid_conf['xbound'][0])*2-1)
        self.norm_mesh_y = torch.tensor((self.mesh_y-input_grid_conf['ybound'][0]) / (input_grid_conf['ybound'][1]-input_grid_conf['ybound'][0])*2-1)
        
        #remember xy coordination of grid_conf(where x is forward,y is left) is different from torch.grid_sample(where x is right,y is backward)
        self.mesh_grid = torch.stack(torch.meshgrid(-self.norm_mesh_x, -self.norm_mesh_y, indexing='ij'), dim=2)

    def forward(self, x):
        # x: bev feature mesh tensor of shape (b, c, h, w)
        grid = self.mesh_grid.unsqueeze(0).type_as(x).repeat(x.shape[0], 1, 1, 1)
        result = F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)
        return result

if __name__ == "__main__":
    base_grid = {
        'xbound': [-10.0, 50.0, 0.125],
        'ybound': [-15.0, 15.0, 0.125],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],
    }
    task_grids = {
    'det': {
        'xbound': [-10.0, 50.0, 0.5],
        'ybound': [-10.0, 10.0, 0.5],
    },
    'seg': {
        'xbound': [-10.0, 50.0, 0.25],
        'ybound': [-15.0, 15.0, 0.25],
    }
    }
    x = torch.zeros(2,64,120,60)
    for name,conf in task_grids.items():
        net = GridSampler(base_grid,conf)
        res = net(x)
        jit_model = torch.jit.script(net,x)
        print(res.shape)
        print(jit_model)
