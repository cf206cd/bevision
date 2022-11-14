import torch

def calculate_birds_eye_view_parameters(xbound, ybound, zbound):
    bev_resolution = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) #分别为x, y, z三个方向上的网格间距
    bev_start_position = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]]) #分别为x, y, z三个方向上第一个格子中心的坐标
    bev_dimension = torch.Tensor([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]]) #分别为x, y, z三个方向上格子的数量

    return bev_resolution, bev_start_position, bev_dimension

