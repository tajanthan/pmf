""" simplex constrained quantized classifier model: 1-hidden layer MLP
"""

import torch
import torch.nn as nn

import SimplexLayers as sl

class SMLP(nn.Module):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (dataset_size, input_dim)
    Returns:
      y: a tensor of shape (dataset_size, output_dim), with values
      equal to the logits of classifying the digit into one of output_dim classes
    """
    def __init__(self, input_dim, hidden_dim, output_dim, Q_l):
        super(SMLP, self).__init__()
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w1 = sl.SLinear(input_dim, hidden_dim, Q_l)
        self.relu1 = nn.ReLU(inplace=True)
        self.w2 = sl.SLinear(hidden_dim, output_dim, Q_l)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.w1(x)
        x = self.relu1(x)
        x = self.w2(x)
        return x

