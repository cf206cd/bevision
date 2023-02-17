import torch
import torch.nn as nn
from regnet import regnetx_002,regnetx_080
from fpn import FPN
from lift_splat import LiftSplat,LiftSplatWithFixedParam
from seg_head import VanillaSegmentHead

class BEVision(nn.Module):
    def __init__(self,grid_conf,num_seg_classes,num_cameras,image_size):
        super().__init__()
        self.image_encoder = regnetx_080()
        self.image_fpn = FPN(in_channels=[self.image_encoder.widths[i] for i in self.image_encoder.out_indices],out_channels=64)
        self.grid_conf = grid_conf
        self.image_size = image_size
        self.transformer = LiftSplat(grid_conf=self.grid_conf,image_size=self.image_size,numC_input=64,numC_trans=64,downsample=8)
        self.bev_encoder = regnetx_002(input_channel=64,out_indices=[2,3],replace_stride_with_dilation=[True,True,True,False])
        self.bev_fpn = FPN(in_channels=[64]+[self.bev_encoder.widths[i] for i in self.bev_encoder.out_indices],out_channels=64,out_ids=[0])
        self.seg_head = VanillaSegmentHead(64,num_seg_classes)
        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.num_cameras = num_cameras

    def forward(self, x, rots, trans, intrins):
        x = x.reshape(-1,x.shape[2],x.shape[3],x.shape[4])
        image_feature = self.image_encoder(x)
        image_fpn_feature = self.image_fpn(image_feature)[0]
        image_fpn_feature = image_fpn_feature.reshape(-1,self.num_cameras,image_fpn_feature.shape[1],image_fpn_feature.shape[2],image_fpn_feature.shape[3])
        bev_image = self.transformer(image_fpn_feature,rots,trans,intrins)
        bev_feature = self.bev_encoder(bev_image)
        bev_fpn_feature = self.bev_fpn([bev_image]+bev_feature)[0]
        seg_res = self.seg_head(bev_fpn_feature)
        return seg_res

class BEVisionWithFixedParam(BEVision):
    def __init__(self,rots,trans,intrins,**kwargs):
        super().__init__(**kwargs)
        self.lss_transformer = LiftSplatWithFixedParam(rots,trans,intrins,image_size=self.image_size,numC_input=64,numC_trans=64,downsample=8,grid_conf=self.grid_conf,intrins_is_inverse=True)
    
    def forward(self, x):
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        image_feature = self.image_encoder(x)
        image_fpn_feature = self.image_fpn(image_feature)[0]
        image_fpn_feature = image_fpn_feature.reshape(-1,self.num_cameras,image_fpn_feature.shape[1],image_fpn_feature.shape[2],image_fpn_feature.shape[3])
        bev_image = self.transformer(image_fpn_feature)
        bev_feature = self.bev_encoder(bev_image)
        bev_fpn_feature = self.bev_fpn([bev_image]+bev_feature)[0]
        seg_res = self.seg_head(bev_fpn_feature)
        return seg_res

if __name__ == '__main__':
    grid_conf = {
        'xbound': [-50.0, 50.0, 200],
        'ybound': [-50.0, 50.0, 200],
        'zbound': [-10.0, 10.0, 1],
        'dbound': [1.0, 50.0, 49],
    }
    import time
    device = torch.device("cpu")
    x = torch.zeros(2,6,3,228,512).to(device)
    rots = torch.zeros(2,6,3,3)
    trans =torch.zeros(2,6,3)
    intrins = torch.zeros(2,6,3,3)
    for i in range(3):
        rots[:,:,i,i] = 1
        intrins[:,:,i,i] = 1

    net1 = BEVision(grid_conf,2,6,(228,512)).to(device)
    start = time.time()
    for i in range(100):
        res1 = net1(x,rots.to(device),trans.to(device),intrins.to(device))
    end = time.time()
    print("FPS:",100/(end-start))
    print(res1.shape)
    net2 = BEVisionWithFixedParam(rots,trans,intrins,grid_conf,2,6,(228,512)).to(device)
    start = time.time()
    for i in range(100):
        res2 = net2(x)
    end = time.time()
    print("FPS:",100/(end-start))
    print(res2.shape)