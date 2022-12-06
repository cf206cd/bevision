import torch

def generate_grid(bound):
    bev_resolution = torch.Tensor([row[2] for row in bound]) #分别为每个方向上的网格间距
    bev_start_position = torch.Tensor(
        [row[0] + row[2]/2.0 for row in bound]) #分别为每个方向上第一个格子中心的坐标
    bev_dimension = torch.Tensor([(row[1] - row[0]) / row[2]
                      for row in bound]) #分别为每个方向上格子的数量

    return bev_resolution, bev_start_position, bev_dimension

