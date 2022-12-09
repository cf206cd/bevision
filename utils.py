import numpy as np
import quaternion

def generate_grid(bound):
    resolution = np.array([row[2] for row in bound]) #分别为每个方向上的网格间距
    start_position = np.array(
        [row[0] + row[2]*0.5 for row in bound]) #分别为每个方向上第一个格子中心的坐标
    dimension = np.array([(row[1] - row[0]) / row[2]
                      for row in bound]) #分别为每个方向上格子的数量

    return resolution, start_position, dimension

def to_rotation_matrix(rotation):
    return quaternion.as_rotation_matrix(quaternion.from_float_array(rotation)).astype(np.float32)

def to_euler_angles(rotation):
    return quaternion.as_euler_angles(quaternion.from_float_array(rotation)).astype(np.float32)