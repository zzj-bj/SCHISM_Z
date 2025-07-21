"""
Activation Mixin for PyTorch Modules
This module provides a mixin class that can be used to add activation functions
to PyTorch modules.
"""
from torch import nn

# pylint: disable=too-few-public-methods
class ActivationMixin:
    """
    Mixin class to add activation functions to a PyTorch module.

    This class provides methods to add activation functions to a PyTorch module.
    It supports common activation functions like ReLU, LeakyReLU, Sigmoid, and Tanh.
    """
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
