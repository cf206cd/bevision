import torch
from model import BEVerse
from dataset import NuScenesDataset
from torch.utils.data import DataLoader
from loss import Loss
from config import Config

class Trainer:
    def __init__(self,config):
        self.config = config
        self.model = BEVerse(config.GRID_CONFIG,num_det_classes=config.NUM_DET_CLASSES,num_seg_classes=config.NUM_SEG_CLASSES,image_size=config.INPUT_IMAGE_SIZE).to(torch.device(config.DEVICE),)
        self.epoch = config.EPOCH
        self.dataset = NuScenesDataset()
        self.dataloader = DataLoader(self.dataset,batch_size=config.BATCH_SIZE)
        self.loss = Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),config.LEARNING_RATE,config.MOMENTUM)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=config.LR_SCHE_STEP_SIZE,gamma=config.LR_SCHE_GAMMA)

    def train(self):
        for epoch in range(self.epoch):
            print("training epoch:",epoch)
            for iter,data in enumerate(self.dataloader):
                with torch.autograd.set_detect_anomaly(True):
                    print("training iterateion:",iter)
                    x,rots,trans,intrins,heatmap_gt,regression_gt,segment_gt = data
                    self.optimizer.zero_grad()
                    predicts = self.model(x,rots,trans,intrins)
                    targets = [heatmap_gt,regression_gt,segment_gt]
                    loss = self.loss(predicts,targets)
                    print("loss:",loss.item())
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                
    
if __name__ == '__main__':
    config = Config
    trainer = Trainer(config)
    trainer.train()
