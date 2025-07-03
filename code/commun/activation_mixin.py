import torch
from torch import nn

class ActivationMixin:
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        if activation == 'leakyrelu':
            return nn.LeakyReLU()
        if activation == 'sigmoid':
            return nn.Sigmoid()
        if activation == 'tanh':
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {activation}")