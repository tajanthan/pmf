""" custom layers to handle simplex methods --> u-expansion
"""

import torch
import torch.nn as nn

class SLinear(nn.Module):
    """modified linear layer to handle u-expansion, q times more parameters than nn.Linear
    :math: `y = (xA^T + b)q`
    """
    def __init__(self, in_features, out_features, Q_l, bias=True):
        super(SLinear, self).__init__()
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        self.linear = nn.Linear(in_features, out_features * self.qlevels, bias=bias)

    def ucollapse(self, x):
        """x: N x dm --> N x m
        """
        x = x.view(x.size(0), -1, self.qlevels) # picks d consecutive elements for each row
        return x.matmul(self.Q_l)

    def forward(self, x):
        x = self.linear(x)
        return self.ucollapse(x)

class SConv2d(nn.Module):
    """modified conv2d layer to handle u-expansion, q times more parameters than nn.conv2d
    :math: `y = (x ** A + b)q`
    """
    def __init__(self, in_channels, out_channels, kernel_size, Q_l, 
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SConv2d, self).__init__()
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        self.conv2d = nn.Conv2d(in_channels, out_channels * self.qlevels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups,bias=bias)

    def ucollapse(self, x):
        """x: N x dm x f1 x f2 --> N x m x f1 x f2
        """
        sz = x.size()
        x = x.permute(0, 2, 3, 1).view(sz[0], sz[2], sz[3], -1, self.qlevels) # picks d consecutive elements for each row
        x = x.matmul(self.Q_l)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.conv2d(x)
        return self.ucollapse(x)
