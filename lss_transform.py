import torch
import torch.nn as nn
from utils import calculate_birds_eye_view_parameters
class LSSTransform(nn.Module):
    def __init__(self, grid_conf=None, input_dim=None, #input_dim: origin image size
                numC_input=512,numC_trans=512, 
                downsample=16, use_quickcumsum=True):

        super().__init__()
        if grid_conf is None:
            grid_conf = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [4.0, 45.0, 1.0], }

        self.grid_conf = grid_conf
        self.dx, self.bx, self.nx = calculate_birds_eye_view_parameters(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )

        self.input_dim = input_dim
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_trans = numC_trans
        self.depthnet = nn.Conv2d(
            self.numC_input, self.D + self.numC_trans, kernel_size=1, padding=0)
        self.use_quickcumsum = use_quickcumsum

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.input_dim
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """

        B, N, _ = trans.shape

        # flatten (batch & sequence)
        rots = rots.flatten(0, 1)
        trans = trans.flatten(0, 1)
        intrins = torch.inverse(intrins).flatten(0, 1).float()
        # inverse can only work for float32
        post_rots = post_rots.flatten(0, 1).float()
        post_trans = post_trans.flatten(0, 1)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)) 

        # cam_to_ego
        
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) # 将像素坐标(u,v,1)根据深度d变成齐次坐标(du,dv,d)

        # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]^T
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)，这里需要倒过来
        combine = rots.matmul(intrins)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def voxel_pooling(self, geom_feats, x):#将图像特征沿着pillars方向进行sum pooling，其中使用了cumsum trick，参考LSS论文4.2节
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        # 将图像特征展平，一共有 (B*N*D*H*W)*C个点
        x = x.reshape(Nprime, C)

        # flatten indices
        bx = self.bx.type_as(geom_feats)
        dx = self.dx.type_as(geom_feats)
        nx = self.nx.type_as(geom_feats).long()

        # 将[-m,m]的范围转换到[0,m*2]，计算栅格坐标并取整
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()  

        # 将像素映射关系同样展平，一共有(B*N*D*H*W)*3个点 
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=geom_feats.device, dtype=torch.long) for ix in range(B)])

        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        geom_feats = geom_feats.type_as(x).long()

        # filter out points that are outside box
        # 过滤掉在边界线之外的点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        # [nx, ny, nz, n_batch]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
            + geom_feats[:, 1] * (nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 一个batch的一个格子里只留一个点
        if self.use_quickcumsum:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        else:
            # cumsum trick
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        # 将x按照栅格坐标放到final中
        final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        # 消除掉z维
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_volume(self,x):
        B, N, C, H, W = x.shape
        # flatten (batch, num_cam)
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)

        # 前D个通道估计深度
        depth = self.get_depth_dist(x[:, :self.D])

        # 后numC_trans个通道提取特征
        cvt_feature = x[:, self.D:(self.D + self.numC_trans)]

        # 深度乘特征，见LSS论文
        volume = depth.unsqueeze(1) * cvt_feature.unsqueeze(2)

        volume = volume.view(B, N, self.numC_trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)
        return volume

    def forward(self, x, rots=None, trans=None, intrins=None, post_rots=None, post_trans=None, flip_x=False, flip_y=False, use_pre_geom=False):
        if use_pre_geom is False:
            # 每个像素对应的视锥体的车体坐标[B, N, D, H, W, 3]
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            if flip_x:
                geom[..., 0] = -geom[..., 0]
            if flip_y:
                geom[..., 1] = -geom[..., 1]

            volume = self.get_volume(x)
            bev_feat = self.voxel_pooling(geom, volume)
            bev_feat = bev_feat.view(x.shape[0], *bev_feat.shape[1:])
        else:
            assert self.pre_geom
            geom_feats = self.geom_feats.to(x.device)
            ranks = self.ranks.to(x.device)
            bev_feat = self.bev_feat.to(x.device)
            kept = self.kept.to(x.device)
            if flip_x:
                geom_feats[..., 0] = -geom_feats[..., 0]
            if flip_y:
                geom_feats[..., 1] = -geom_feats[..., 1]
            
            volume = self.get_volume(x)
            B, N, D, H, W, C = volume.shape
            Nprime = B * N * D * H * W
            # flatten x
            # 将图像特征展平，一共有 (B*N*D*H*W)*C个点
            volume = volume.reshape(Nprime, C)
            volume = volume[kept]
            volume = volume[self.sorts]

            # 一个batch的一个格子里只留一个点
            if self.use_quickcumsum:
                volume, geom_feats = QuickCumsum.apply(volume, geom_feats, ranks)
            else:
                # cumsum trick
                volume, geom_feats = cumsum_trick(volume, geom_feats, ranks)

            # griddify (B x C x Z x X x Y)
            # 将x按照栅格坐标放到final中
            bev_feat[geom_feats[:, 3], :, geom_feats[:, 2],
                geom_feats[:, 1], geom_feats[:, 0]] = volume

            # collapse Z
            # 消除掉z维
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
            bev_feat = bev_feat.view(x.shape[0], *bev_feat.shape[1:])
        return bev_feat

    def set_geometry(self,feature_shape, rots, trans, intrins, post_rots, post_trans):
        self.pre_geom = True
        geom_feats = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        B, N, D, H, W, C = feature_shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)

        # flatten indices
        bx = self.bx.type_as(geom_feats)
        dx = self.dx.type_as(geom_feats)
        nx = self.nx.type_as(geom_feats).long()

        # 将[-m,m]的范围转换到[0,m*2]，计算栅格坐标并取整
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()  

        # 将像素映射关系同样展平，一共有(B*N*D*H*W)*3个点 
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=geom_feats.device, dtype=torch.long) for ix in range(B)])

        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        geom_feats = geom_feats.long()

        # filter out points that are outside box
        # 过滤掉在边界线之外的点
        self.kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])

        # [nx, ny, nz, n_batch]
        geom_feats = geom_feats[self.kept]

        # get tensors from the same voxel next to each other
        # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
            + geom_feats[:, 1] * (nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        self.sorts = ranks.argsort()
        self.geom_feats, self.ranks = geom_feats[self.sorts], ranks[self.sorts]
        self.bev_feat = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=geom_feats.device)


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats 

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

if __name__ == "__main__":
    input = torch.zeros(4,6,64,20,20)
    rots = torch.zeros(4,6,3,3)
    trans = torch.zeros(4,6,3)
    intrins = torch.zeros(4,6,3,3)
    post_rots = torch.zeros(4,6,3,3)
    post_trans = torch.zeros(4,6,3)
    for i in range(3):
        rots[:,:,i,i] = 1
        intrins[:,:,i,i] = 1
        post_rots[:,:,i,i] = 1
    net = LSSTransform(input_dim=(160,160),numC_input=64,numC_trans=64,downsample=8)
    output1 = net(input,rots,trans,intrins,post_rots,post_trans)
    print(output1.shape)
    net.eval()
    net.set_geometry(net.get_volume(input).shape,rots,trans,intrins,post_rots,post_trans)
    output2 = net(input,use_pre_geom = True)
    print(output2.shape)
    print(torch.sum(output2-output1))
   
