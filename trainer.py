import torch
from model import BEVerse
from dataset import NuScenesDataset
from torch.utils.data import DataLoader
from loss import Loss
from config import Config

class Trainer:
    def __init__(self,config):
        self.config = config
        self.model = BEVerse(config.GRID_CONFIG,image_size=config.INPUT_IMAGE_SIZE).to(torch.device(config.DEVICE))
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
                print("training iterateion:",iter)
                self.optimizer.zero_grad()
                x,rots,trans,intrins = self.generate_inputs(data['data'])
                predicts = self.model(x,rots,trans,intrins)
                targets = self.generate_targets(data['sample_count'],data['instances'],data['map'])
                loss = self.loss(predicts,targets)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def generate_inputs(self,data):
        x = torch.stack([i['raw'] for i in data]).permute(1,0,2,3,4)
        x = x.reshape(-1,*x.shape[2:])
        rots =  torch.stack([i['rotation'] for i in data]).permute(1,0,2,3)
        trans =  torch.stack([i['translation'] for i in data]).permute(1,0,2)
        intrins =  torch.stack([i['camera_intrinsic'] for i in data]).permute(1,0,2,3)
        return x,rots,trans,intrins

    def generate_targets(self,sample_count,instances,map_token):
        pass

if __name__ == '__main__':
    config = Config
    trainer = Trainer(config)
    trainer.train()
