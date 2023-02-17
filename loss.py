import torch
import torch.nn as nn
import torch.nn.functional as F
class BinaryLoss(torch.nn.Module):
    def __init__(self, reduction, pos_weight=2.13):
        super(BinaryLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction=reduction,pos_weight=torch.tensor(pos_weight))

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss
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

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.segment_loss = BinaryLoss(reduction="mean")

    def forward(self,inputs,targets):
        segment_loss = self.segment_loss(inputs,targets)
        return segment_loss
