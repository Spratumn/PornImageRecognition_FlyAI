import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class CELoss(nn.Module):
    def __init__(self, gamma=2):
        super(CELoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, target):
        """
        inputs: shape of (N,C)
        target: shape of (N,C)
        """
        inputs = torch.clamp(inputs, min=1e-4, max=1 - 1e-4)

        class_num = inputs.size(1)
        target = F.one_hot(target, num_classes=class_num)
        pos_inds = target == 1
        neg_inds = target == 0

        loss = 0
        pos_pred = inputs[pos_inds]
        neg_pred = inputs[neg_inds]

        pos_loss = torch.log(pos_pred)
        neg_loss = torch.log(1 - neg_pred)

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = loss - (pos_loss + neg_loss)
        return loss


if __name__ == '__main__':
    inputs = torch.rand(2, 5).softmax(dim=1)
    target = [2, 4]
    print(inputs)
    print(target)
    print(CELoss()(inputs, target))


