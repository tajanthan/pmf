""" simplex constrained quantized classifier model: LeNet-5
"""

import torch
import torch.nn as nn

from collections import OrderedDict

import SimplexLayers as sl

class SLeNet5(nn.Module):
    """
    Input - 1x28x28
    conv1 - 20@24x24 (5x5 kernel)
    bn1
    relu1
    pool1 - 20@12x12 (2x2 kernel, stride=2)
    conv2 - 50@8x8 (5x5 kernel)
    bn2
    relu2 
    pool2 - 50@4x4 (2x2 kernel, stride=2)
    fc1   - 500x1
    bn3
    relu3
    fc2   - 10x1 (Output)
    """
    def __init__(self, input_channels, imsize, output_dim, Q_l):
        super(SLeNet5, self).__init__()
        self.input_channels = input_channels    # 1 or 3
        self.imsize = imsize                    # square image -- even
        self.output_dim = output_dim
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        assert(imsize % 2 == 0)

        self.cnn = nn.Sequential(OrderedDict([        
            ('conv1', sl.SConv2d(input_channels, 20, 5, Q_l)),   # imsize-4
            ('bn1', nn.BatchNorm2d(20, affine=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),                # (imsize-4)/2
            ('conv2', sl.SConv2d(20, 50, 5, Q_l)),              # (imsize-4)/2 - 4
            ('bn2', nn.BatchNorm2d(50, affine=False)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2))                # ((imsize-4)/2 - 4)/2
        ]))

        self.ftsize = ((imsize - 4)/2 - 4)/2
        self.fc = nn.Sequential(OrderedDict([
           ('fc1', sl.SLinear(50 * self.ftsize * self.ftsize, 500, Q_l)),
           ('bn3', nn.BatchNorm1d(500, affine=False)),
           ('relu3', nn.ReLU(inplace=True)),
           ('fc2', sl.SLinear(500, output_dim, Q_l))
        ]))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 50 * self.ftsize * self.ftsize)
        x = self.fc(x)
        return x
