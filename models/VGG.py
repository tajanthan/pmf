""" VGG11/13/16/19
    Based on: https://github.com/kuangliu/pytorch-cifar
"""
import torch
import torch.nn as nn

from collections import OrderedDict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, input_channels=3, imsize=32, output_dim=10):
        super(VGG, self).__init__()
        self.input_channels = input_channels
        self.imsize = imsize
        self.output_dim = output_dim
        self.stride1 = 1
        self.fc_size = 512
        if imsize == 64:    # tinyimagenet
            self.stride1 = 2
        if imsize != 32 and imsize != 64:   # not cifar and not tiny imagenet, then original vgg size (imagenet)
            self.fc_size = 4096

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(OrderedDict([
           ('fc1', nn.Linear(512, self.fc_size)),
           ('bn1', nn.BatchNorm1d(self.fc_size, affine=False)),
           ('relu1', nn.ReLU(inplace=True)),
           ('fc2', nn.Linear(self.fc_size, self.fc_size)),
           ('bn2', nn.BatchNorm1d(self.fc_size, affine=False)),
           ('relu2', nn.ReLU(inplace=True)),
           ('fc3', nn.Linear(self.fc_size, self.output_dim))
        ]))
#        self.classifier = nn.Linear(self.fc_size, self.output_dim)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_channels
        stride = self.stride1   # diffent stride only for the first one
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=stride, padding=1),
                           nn.BatchNorm2d(x, affine=False),
                           nn.ReLU(inplace=True)]
                in_channels = x
                stride = 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
