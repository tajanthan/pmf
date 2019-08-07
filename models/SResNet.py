""" simplex constrained model: ResNet18/34/50/101/152
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import SimplexLayers as sl

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, Q_l, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = sl.SConv2d(in_planes, planes, 3, Q_l, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.conv2 = sl.SConv2d(planes, planes, 3, Q_l, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                sl.SConv2d(in_planes, self.expansion*planes, 1, Q_l, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, Q_l, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = sl.SConv2d(in_planes, planes, 1, Q_l, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.conv2 = sl.SConv2d(planes, planes, 3, Q_l, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.conv3 = sl.SConv2d(planes, self.expansion*planes, 1, Q_l, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                sl.SConv2d(in_planes, self.expansion*planes, 1, Q_l, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SResNet(nn.Module):
    def __init__(self, block, num_blocks, Q_l, input_channels=3, imsize=32, output_dim=10):
        super(SResNet, self).__init__()
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        self.in_planes = 64

        self.input_channels = input_channels
        self.imsize = imsize
        self.output_dim = output_dim
        self.stride1 = 1
        if imsize == 64:    # tinyimagenet
            self.stride1 = 2

        self.conv1 = sl.SConv2d(3, 64, 3, Q_l, stride=self.stride1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = sl.SLinear(512*block.expansion, self.output_dim, Q_l)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.Q_l, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SResNet18(Q_l, input_channels=3, imsize=32, output_dim=10):
    return SResNet(BasicBlock, [2,2,2,2], Q_l, input_channels, imsize, output_dim)

def SResNet34(Q_l, input_channels=3, imsize=32, output_dim=10):
    return SResNet(BasicBlock, [3,4,6,3], Q_l, input_channels, imsize, output_dim)

def SResNet50(Q_l, input_channels=3, imsize=32, output_dim=10):
    return SResNet(Bottleneck, [3,4,6,3], Q_l, input_channels, imsize, output_dim)

def SResNet101(Q_l, input_channels=3, imsize=32, output_dim=10):
    return SResNet(Bottleneck, [3,4,23,3], Q_l, input_channels, imsize, output_dim)

def SResNet152(Q_l, input_channels=3, imsize=32, output_dim=10):
    return SResNet(Bottleneck, [3,8,36,3], Q_l, input_channels, imsize, output_dim)


def test():
    net = SResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
