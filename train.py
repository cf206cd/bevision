from config import Config
from trainer import Trainer
import torch
import numpy as np
import random

# setup random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(42)
config = Config
trainer = Trainer(config)
trainer.train()
