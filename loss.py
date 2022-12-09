import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self,alpha=-1,gamma=1, reduction = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        Arguments:
            inputs: batch x dim x h x w, without sigmoid
            target: batch x dim x h x w, only 0 and 1
        '''
        shifted_inputs = self.gamma * (inputs * (2 * targets - 1))
        loss = -(F.logsigmoid(shifted_inputs)) / self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss *= alpha_t

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class HeatmapFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred,gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w) without sigmoid 
            gt (batch x c x h x w) between 0 and 1
        '''
        pos_inds = gt.gt(0).float()
        neg_inds = gt.eq(0).float()
        neg_weights = torch.pow(1 - gt, 4)
        pos_loss = F.logsigmoid(pred) * torch.pow(1 - pred.sigmoid(), 2) * pos_inds
        neg_loss = F.logsigmoid(-pred) * torch.pow(pred.sigmoid(), 2) * neg_weights * neg_inds
        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = - neg_loss
        else:
            loss = - (pos_loss + neg_loss) / num_pos
        return loss

class SmoothL1Loss(torch.nn.SmoothL1Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self,inputs,targets,mask):
        return super().forward(inputs*mask,targets*mask)

class Loss(nn.Module):
    def __init__(self,gamma1=1.0,gamma2=1.0):
        super().__init__()
        self.regression_loss =  SmoothL1Loss(reduction="mean")
        self.heatmap_loss = HeatmapFocalLoss()
        self.segment_loss = BinaryFocalLoss(reduction="mean")
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def forward(self,inputs,targets):
        heatmap,regression,segment = inputs
        heatmap_gt,regression_gt,segment_gt = targets
        mask = regression_gt!=0
        heatmap_loss = self.heatmap_loss(heatmap,heatmap_gt)
        regression_loss = self.regression_loss(regression,regression_gt,mask)
        segment_loss = self.segment_loss(segment,segment_gt)
        loss = heatmap_loss+self.gamma1*regression_loss+self.gamma2*segment_loss
        return loss



