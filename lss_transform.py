import torch
import torch.nn as nn
from utils import generate_grid
class LSSTransform(nn.Module):
    def __init__(self, grid_conf=None, image_size=None, #image_size: origin image count, H x W
                numC_input=512,numC_trans=512, 
                downsample=16, use_quickcumsum=True):

        super().__init__()
        self.grid_conf = grid_conf
        lower_bound, interval, count = [torch.tensor(x) for x in generate_grid([self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound']])]
        self.interval = nn.Parameter(interval, requires_grad=False)
        self.lower_bound = nn.Parameter(lower_bound, requires_grad=False)
        self.count = nn.Parameter(count, requires_grad=False)
        self.image_size = image_size
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_trans = numC_trans
        self.depthnet = nn.Conv2d(
            self.numC_input, self.D + self.numC_trans, kernel_size=1, padding=0)
        self.use_quickcumsum = use_quickcumsum

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.image_size
        fH, fW = (ogfH+self.downsample-1) // self.downsample, (ogfW+self.downsample-1) // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).reshape(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).reshape(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).reshape(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    def get_geometry(self, rots, trans, intrins):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """

        N = trans.shape[1]

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) # 将像素坐标(u,v,1)根据深度d变成齐次坐标(du,dv,d)

        # flatten (batch & sequence)
        rots = rots.flatten(0, 1).to(points.device)
        trans = trans.flatten(0, 1).to(points.device)
        intrins = intrins.flatten(0, 1).float().to(points.device)
        # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]^T
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)，这里需要倒过来
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.reshape(-1, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points = trans.reshape(-1, N, 1, 1, 1, 3)+points
        return points

    def get_volume(self,x):
        B, N, C, H, W = x.shape
        # flatten (batch, num_cam)
        x = x.reshape(B * N, C, H, W)
        x = self.depthnet(x)

        # 前D个通道估计深度
        depth = self.get_depth_dist(x[:, :self.D])

        # 后numC_trans个通道提取特征
        cvt_feature = x[:, self.D:(self.D + self.numC_trans)]

        # 深度乘特征，见LSS论文
        volume = depth.unsqueeze(1) * cvt_feature.unsqueeze(2)

        volume = volume.reshape(B, N, self.numC_trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)
        return volume

    def voxel_pooling_prepare(self, geom):
        # flatten indices
        lower_bound = self.lower_bound.type_as(geom)
        interval = self.interval.type_as(geom)
        count = self.count.type_as(geom).long()

        # 将[m,n]的范围转换到[0,n-m]，计算栅格坐标并取整
        geom_grid = ((geom - (lower_bound - interval * 0.5)) / interval).long()  
        B = geom_grid.shape[0]

        # 将像素映射关系同样展平，一共有(B*N*D*H*W)个点 
        geom_grid = geom_grid.reshape(-1, 3)
        batch_ix = torch.cat([torch.full([geom_grid.shape[0]//B,1], ix,
                                         device=geom_grid.device, dtype=torch.long) for ix in range(B)])

        geom_grid = torch.cat((geom_grid, batch_ix), 1)

        # filter out points that are outside box
        # 过滤掉在边界线之外的点
        kept = (geom_grid[:, 0] >= 0) & (geom_grid[:, 0] < count[0]) \
            & (geom_grid[:, 1] >= 0) & (geom_grid[:, 1] < count[1]) \
            & (geom_grid[:, 2] >= 0) & (geom_grid[:, 2] < count[2])

        # [count, ny, nz, n_batch]
        geom_grid = geom_grid[kept]

        # get tensors from the same voxel next to each other
        # 给每一个点一个rank值，同一个batch且同一个格子里面的点rank值相等
        ranks = geom_grid[:, 0] * (count[1] * count[2] * B) \
            + geom_grid[:, 1] * (count[2] * B) \
            + geom_grid[:, 2] * B \
            + geom_grid[:, 3]
        sorts = ranks.argsort()
        geom_grid = geom_grid[sorts]
        ranks = ranks[sorts]
        return geom_grid,ranks,kept,sorts

    def voxel_pooling(self, geom, x):
        B, N, D, H, W, C = x.shape
        geom,ranks,kept,sorts = self.voxel_pooling_prepare(geom)
        
        # flatten x
        # 将图像特征展平，一共有 (B*N*D*H*W)*C个点
        x = x.reshape(-1, x.shape[-1])
        x = x[kept][sorts]
        geom = geom.to(x.device)
        ranks = ranks.to(x.device)

        # 一个batch的一个格子里只留一个点
        if self.use_quickcumsum:
            x, geom = QuickCumsum.apply(x, geom, ranks)
        else:
            # cumsum trick
            x, geom = cumsum_trick(x, geom, ranks)

        # griddify (B x C x Z x X x Y)
        # 将x按照栅格坐标放到final中
        count = self.count.long()
        final = torch.zeros((B, C, count[2], count[0], count[1]), device=x.device)

        final[geom[:, 3], :, geom[:, 2],
              geom[:, 0], geom[:, 1]] = x

        # collapse Z
        # 消除掉z维
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, x, rots, trans, intrins):
        # 每个像素对应的视锥体的车体坐标[B, N, D, H, W, 3]
        geom = self.get_geometry(rots, trans, intrins)

        # 每个特征图上的像素对应的深度上的特征
        volume = self.get_volume(x)

        # 将图像特征沿着pillars方向进行sum pooling，其中使用了cumsum trick，参考LSS论文4.2节
        bev_feat = self.voxel_pooling(geom, volume)

        return bev_feat

class LSSTransformWithFixedParam(LSSTransform):
    def __init__(self, rots,trans,intrins, **kwargs):
        super().__init__(**kwargs)
        geom = self.get_geometry(rots, trans, intrins)
        self.geom,self.ranks,self.kept,self.sorts = self.voxel_pooling_prepare(geom)

    def forward(self,x):
        # 每个特征图上的像素对应的深度上的特征
        x = self.get_volume(x)
        B, N, D, H, W, C = x.shape

        # flatten x
        # 将图像特征展平，一共有 (B*N*D*H*W)*C个点
        x = x.reshape(-1, x.shape[-1])
        x = x[self.kept][self.sorts]
        geom = self.geom.to(x.device)
        ranks = self.ranks.to(x.device)

        # 一个batch的一个格子里只留一个点
        if self.use_quickcumsum:
            x, geom = QuickCumsum.apply(x, geom, ranks)
        else:
            # cumsum trick
            x, geom = cumsum_trick(x, geom, ranks)

        # griddify (B x C x Z x X x Y)
        # 将x按照栅格坐标放到final中
        count = self.count.long()
        bev_feat = torch.zeros((B, C, count[2], count[0], count[1]), device=x.device)

        bev_feat[geom[:, 3], :, geom[:, 2],
              geom[:, 0], geom[:, 1]] = x

        # collapse Z
        # 消除掉z维
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

def cumsum_trick(x, geom, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom = x[kept], geom[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom = x[kept], geom[kept]
        # Calculate sum of features within a voxel.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom
        ctx.mark_non_differentiable(geom)

        return x, geom 

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)

        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        back[kept] -= 1
        val = gradx[back]

        return val, None, None

if __name__ == "__main__":
    grid_conf = {
        'xbound': [-10.0, 50.0, 0.125],
        'ybound': [-15.0, 15.0, 0.125],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],}
    input = torch.zeros(2,6,64,80,80)
    rots = torch.zeros(2,6,3,3)
    trans = torch.zeros(2,6,3)
    intrins = torch.zeros(2,6,3,3)
    for i in range(3):
        rots[:,:,i,i] = 1
        intrins[:,:,i,i] = 1
    net1 = LSSTransform(grid_conf=grid_conf,image_size=(640,640),numC_input=64,numC_trans=64,downsample=8)
    output1 = net1(input,rots,trans,intrins)
    print(output1.shape)
    net2 = LSSTransformWithFixedParam(rots,trans,intrins,grid_conf=grid_conf,image_size=(640,640),numC_input=64,numC_trans=64,downsample=8)
    output2 = net2(input)
    print(output2.shape)
   
