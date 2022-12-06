import torch
import torch.nn as nn
from regnet import regnetx_002
from fpn import FPN
from lss_transform import LSSTransform,LSSTransformWithFixedParam
from grid_sampler import GridSampler
from det_head import CenterPointHead
from seg_head import VanillaSegmentHead

class BEVerse(nn.Module):
    def __init__(self,grid_confs,num_det_classes=10,num_seg_classes=10,num_images=6,image_size=(640,640)):
        super().__init__()
        self.image_encoder = regnetx_002()
        self.image_fpn = FPN(in_channels=[56,152,368],out_channels=64)
        self.grid_conf = grid_confs['base']
        self.image_size = image_size
        self.lss_transformer = LSSTransform(grid_conf=self.grid_conf,image_size=self.image_size,numC_input=64,numC_trans=64,downsample=8)
        self.bev_encoder = regnetx_002(input_channel=64,out_indices=[2,3],replace_stride_with_dilation=[True,True,True,False])
        self.bev_fpn = FPN(in_channels=[152,368],out_channels=64,out_ids=[0])
        self.grid_samplers = {}
        for task,conf in grid_confs.items():
            self.grid_samplers[task] = GridSampler(self.grid_conf,conf)
        self.det_head = CenterPointHead(64,num_det_classes)
        self.seg_head = VanillaSegmentHead(64,num_seg_classes)
        self.num_images = num_images

    def forward(self, x, rots, trans, intrins):
        x = x.reshape(-1,*x.shape[2:])
        image_feature = self.image_encoder(x)
        image_fpn_feature = self.image_fpn(image_feature)[0]
        image_fpn_feature = image_fpn_feature.reshape(-1,self.num_images,*image_fpn_feature.shape[1:])
        lss_feature = self.lss_transformer(image_fpn_feature,rots,trans,intrins)
        bev_feature = self.bev_encoder(lss_feature)
        bev_fpn_feature = self.bev_fpn(bev_feature)[0]
        grid_cells = {}
        for task,grid_sampler in self.grid_samplers.items():
            grid_cells[task] = grid_sampler(bev_fpn_feature)
        det_res = self.det_head(grid_cells['det'])
        seg_res = self.seg_head(grid_cells['seg'])
        return det_res,seg_res

class BEVerseWithFixedParam(BEVerse):
    def __init__(self,rots,trans,intrins,**kwargs):
        super().__init__(**kwargs)
        self.lss_transformer = LSSTransformWithFixedParam(rots,trans,intrins,image_size=self.image_size,numC_input=64,numC_trans=64,downsample=8,grid_conf=self.grid_conf)
    
    def forward(self, x):
        x = x.reshape(-1,*x.shape[2:])
        image_feature = self.image_encoder(x)
        image_fpn_feature = self.image_fpn(image_feature)[0]
        image_fpn_feature = image_fpn_feature.reshape(-1,self.num_images,*image_fpn_feature.shape[1:])
        lss_feature = self.lss_transformer(image_fpn_feature)
        bev_feature = self.bev_encoder(lss_feature)
        bev_fpn_feature = self.bev_fpn(bev_feature)[0]
        grid_cells = {}
        for task,grid_sampler in self.grid_samplers.items():
            grid_cells[task] = grid_sampler(bev_fpn_feature)
        det_res = self.det_head(grid_cells['det'])
        seg_res = self.seg_head(grid_cells['seg'])
        return det_res,seg_res

if __name__ == '__main__':
    grid_confs = {
    'base':{
        'xbound': [-51.2, 51.2, 0.8],
        'ybound': [-51.2, 51.2, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],
    },
    'det': {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
    },
    'seg': {
        'xbound': [-15.0, 15.0, 0.15],
        'ybound': [-30.0, 30.0, 0.15],
    }
    }
    import time
    device = torch.device("cpu")
    x = torch.zeros(2,6,3,640,640).to(device)
    rots = torch.zeros(2,6,3,3)
    trans =torch.zeros(2,6,3)
    intrins = torch.zeros(2,6,3,3)
    for i in range(3):
        rots[:,:,i,i] = 1
        intrins[:,:,i,i] = 1

    net1 = BEVerse(grid_confs).to(device)
    start = time.time()
    for i in range(100):
        res1 = net1(x,rots.to(device),trans.to(device),intrins.to(device))
    end = time.time()
    print("FPS:",100/(end-start))
    print(res1[0][0].shape,res1[0][1].shape,res1[1].shape)
    net2 = BEVerseWithFixedParam(rots,trans,intrins,grid_confs=grid_confs).to(device)
    start = time.time()
    for i in range(100):
        res2 = net2(x)
    end = time.time()
    print("FPS:",100/(end-start))
    print(res2[0][0].shape,res2[0][1].shape,res2[1].shape)