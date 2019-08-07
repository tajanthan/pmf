""" model: LeNet-300-100-10
"""

import torch
import torch.nn as nn

class LeNet300(nn.Module):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (dataset_size, input_dim)
    Returns:
      y: a tensor of shape (dataset_size, output_dim), with values
      equal to the logits of classifying the digit into one of output_dim classes
    """
    def __init__(self, input_dim, output_dim):
        super(LeNet300, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w1 = nn.Linear(input_dim, 300)
        self.bn1 = nn.BatchNorm1d(300, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.w2 = nn.Linear(300, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.w3 = nn.Linear(100, output_dim)

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
