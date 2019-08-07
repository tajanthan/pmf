""" simplex constrained quantized classifier model: LeNet-300-100-10
"""

import torch
import torch.nn as nn

import SimplexLayers as sl

class SLeNet300(nn.Module):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (dataset_size, input_dim)
    Returns:
      y: a tensor of shape (dataset_size, output_dim), with values
      equal to the logits of classifying the digit into one of output_dim classes
    """
    def __init__(self, input_dim, output_dim, Q_l):
        super(SLeNet300, self).__init__()
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w1 = sl.SLinear(input_dim, 300, Q_l)
        self.bn1 = nn.BatchNorm1d(300, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.w2 = sl.SLinear(300, 100, Q_l)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.w3 = sl.SLinear(100, output_dim, Q_l)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.w1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.w2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.w3(x)
        return x
