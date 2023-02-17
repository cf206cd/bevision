import torch
from model import BEVision
from dataset import NuScenesDataset
from torch.utils.data import DataLoader
from loss import Loss
from config import Config
import random
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

import cv2
class Trainer:
    def __init__(self,config):
        self.config = config
        self.setup_seed(config.RANDOM_SEED)
        self.model = BEVision(config.GRID_CONFIG,num_seg_classes=config.NUM_SEG_CLASSES,num_cameras=config.NUM_CAMERAS,image_size=config.INPUT_IMAGE_SIZE).to(torch.device(config.DEVICE))
        self.epoch = config.EPOCH
        self.dataset = NuScenesDataset()
        self.dataloader = DataLoader(self.dataset,batch_size=config.BATCH_SIZE,shuffle=True)
        self.loss = Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=config.LR_SCHE_STEP_SIZE,gamma=config.LR_SCHE_GAMMA)
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)

    # setup random seed
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        iteration = 0
        for epoch in range(self.epoch):
            print("training epoch:",epoch)
            self.model.train()
            for data in self.dataloader:
                print("training iterateion:",iteration)
                x,rots,trans,intrins,segment_gt = [var.to(self.config.DEVICE) for var in data]
                cv2.imwrite("./imgs/{0}seg_gt.png".format(iteration),segment_gt.squeeze(0).sum(dim=0).cpu().detach().numpy()*255)
                self.optimizer.zero_grad()
                seg_res = self.model(x,rots,trans,intrins)
                loss = self.loss(seg_res,segment_gt)
                print("loss:",loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                self.optimizer.step()
                self.scheduler.step()
                self.writer.add_scalar('train/loss', loss.item(),iteration,new_style=True)
                cv2.imwrite("./imgs/{0}seg_res.png".format(iteration),seg_res.squeeze(0).sigmoid().sum(dim=0).cpu().detach().numpy()*255)
                iteration+=1
            torch.save(self.model.state_dict(),os.path.join(self.config.LOG_DIR,"model_epoch{0}.pt".format(epoch)))
            torch.save(self.model.state_dict(),os.path.join(self.config.LOG_DIR,"model_last.pt"))
            
if __name__ == '__main__':
    config = Config
    trainer = Trainer(config)
    trainer.train()
