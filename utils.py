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