import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,alpha=-1,gamma=1, reduction = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        Arguments:
            out: batch x dim x h x w
            target: batch x dim x h x w
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

class SmoothL1Loss(torch.nn.SmoothL1Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self,inputs,targets,mask):
        return super().forward(inputs*mask,targets*mask)

class Loss(nn.Module):
    def __init__(self,gamma1=1.0,gamma2=1.0):
        super().__init__()
        self.regression_loss =  SmoothL1Loss(reduction="mean")
        self.heatmap_loss = FocalLoss(reduction="mean")
        self.segment_loss = FocalLoss(reduction="mean")
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def forward(self,inputs,targets):
        heatmap,regression,segment = inputs
        heatmap_gt,regression_gt,segment_gt = targets
        mask = regression_gt!=0
        loss = self.heatmap_loss(heatmap,heatmap_gt)+ \
                self.gamma1*self.regression_loss(regression,regression_gt,mask)+ \
                self.gamma2*self.segment_loss(segment,segment_gt)
        return loss



