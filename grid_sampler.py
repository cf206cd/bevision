import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_birds_eye_view_parameters

class Grid_sampler(nn.Module):
    def __init__(self,input_grid_cell,target_grid_cell):
        super().__init__()
        input_resolution, input_start_position, input_dimension = calculate_birds_eye_view_parameters(
                input_grid_cell['xbound'], input_grid_cell['ybound'], input_grid_cell['zbound'],
            )

        target_resolution, target_start_position, target_dimension = calculate_birds_eye_view_parameters(
                target_grid_cell['xbound'], target_grid_cell['ybound'], target_grid_cell['zbound'],
            )

        self.map_x = torch.arange(
                target_start_position[0], target_grid_cell['xbound'][1], target_resolution[0])

        self.map_y = torch.arange(
                target_start_position[1], target_grid_cell['ybound'][1], target_resolution[1])

        # convert to normalized coords
        self.norm_map_x = self.map_x / (- input_start_position[0])
        self.norm_map_y = self.map_y / (- input_start_position[1])

        self.map_grid = torch.stack(torch.meshgrid(
            self.norm_map_x, self.norm_map_y, indexing='xy'), dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        grid = self.map_grid.unsqueeze(0).type_as(x).repeat(x.shape[0], 1, 1, 1)
        return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)