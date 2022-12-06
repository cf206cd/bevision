import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,alpha=-1,gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets,reduction = "none"):
        '''
        Arguments:
            out: batch x dim x h x w
            target: batch x dim x h x w
        '''

        inputs = inputs.float()
        shifted_inputs = self.gamma * (inputs * (2 * targets - 1))
        loss = -(F.logsigmoid(shifted_inputs)) / self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss *= alpha_t

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
    
class Loss(nn.Module):
    def __init__(self,gamma1=1.0,gamma2=1.0):
        super().__init__()
        self.regression_loss = torch.nn.SmoothL1Loss()
        self.heatmap_loss = FocalLoss()
        self.segment_loss = FocalLoss()
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def forward(self,inputs,heatmap_gt,regression_gt,segment_gt):
        heatmap,regression,segment = inputs
        loss = self.heatmap_loss(heatmap,heatmap_gt)+ \
                self.gamma1*self.regression_loss(regression,regression_gt)+ \
                self.gamma2*self.segment_loss(segment,segment_gt)
        return loss



