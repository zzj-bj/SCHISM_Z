"""
UnetVanilla: A simple U-Net implementation in PyTorch.

UnetVanilla is a PyTorch module that implements a U-Net architecture for image segmentation.
This module defines a U-Net architecture for image segmentation tasks.
It includes configurable parameters for the number of blocks, channels, kernel size,
activation functions, and the number of output classes.
It consists of encoder blocks, decoder blocks, and a bridge layer.

@author: Florent.BRONDOLO
"""

import torch
from torch import nn
import torch.nn.functional as F

class UnetVanilla(nn.Module):
    """
    UnetVanilla: A simple U-Net implementation in PyTorch.
    """
    REQUIRED_PARAMS = {
        'num_classes': int,
    }

    OPTIONAL_PARAMS = {
        'n_block': int,
        'channels': int,
        'k_size': int,  
        'activation': str 
    }

    def __init__(self,
                 n_block=4,
                 channels=8,
                 num_classes=3,
                 k_size=3,
                 activation='relu'
                 ):
        super(UnetVanilla, self).__init__()
        self.n_block = n_block
        self.channels = channels
        self.num_classes = num_classes
        self.k_size = k_size
        self.activation = activation.lower()

        self.encoder_blocks = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        for i in range(n_block):
            self.encoder_blocks.append(self.encoder_block(channels * (2 ** i)))
            self.max_pools.append(nn.MaxPool2d(kernel_size=2, ceil_mode=True))

        # Define up sampling path
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_block, -1, -1):
            if i > 0:
                self.decoder_blocks.append(self.decoder_block(channels * (2 ** i)))
                self.up_convs.append(nn.ConvTranspose2d(
                    channels * (2 ** i), channels * (2 ** i) // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1  # Allow a small correction
                ))

        # Define output layer
        self.output_conv = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

        self.bridge = self.simple_conv(self.channels * (2 ** n_block)// 2,
                                       self.channels * (2 ** n_block))
        self.start = self.simple_conv(3, self.channels)

    def _get_activation(self):
        """
        This method returns the activation function based on the specified type.

        Supported activation types: 'relu', 'leakyrelu', 'sigmoid', 'tanh'.
        Returns:
            nn.Module: Activation function module.
        Raises:
            ValueError: If the specified activation function is not supported.
        """
        if self.activation == 'relu':
            return nn.ReLU()
        if self.activation == 'leakyrelu':
            return nn.LeakyReLU()
        if self.activation == 'sigmoid':
            return nn.Sigmoid()
        if self.activation == 'tanh':
            return nn.Tanh()   
        raise ValueError(f"Unsupported activation function: {self.activation}")

    def simple_conv(self, in_channels, out_channels):
        """
        Creates a simple convolutional block with activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        Returns:
            nn.Sequential: A sequential block containing a convolutional layer
            and an activation function.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.k_size, padding=self.k_size // 2),
            self._get_activation()
        )

    def encoder_block(self, in_channels):
        """
        Creates an encoder block with convolution, batch normalization, and activation.

        Args:
            in_channels (int): Number of input channels.
        Returns:
            nn.Sequential: A sequential block containing a convolutional layer,
            batch normalization and an activation function.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2,
                      kernel_size=self.k_size,
                      padding=self.k_size // 2),
            nn.BatchNorm2d(in_channels * 2),
            self._get_activation()
        )

    def decoder_block(self, in_channels):
        """
        Creates a decoder block with convolution, batch normalization and activation.

        Args:
            in_channels (int): Number of input channels.
        Returns:
            nn.Sequential: A sequential block containing a convolutional layer,
            batch normalization and an activation function.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2,
                      kernel_size=self.k_size,
                      padding=self.k_size // 2),
            nn.BatchNorm2d(in_channels // 2),
            self._get_activation()
        )

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        This method processes the input tensor through the encoder blocks,
        bridge, decoder blocks, and outputs the final segmentation map.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        encodings = []
        x = self.start(x)
        encodings.append(x)

        for i in range(self.n_block):
            x = self.max_pools[i](x)
            if i < self.n_block - 1:
                x = self.encoder_blocks[i](x)
                encodings.append(x)

        x = self.bridge(x)

        for i in range(self.n_block):
            x = self.up_convs[i](x)
            enc = F.interpolate(encodings[-(i + 1)],
                                size=(x.shape[2],
                                x.shape[3]),
                                mode='bilinear',
                                align_corners=True)
            x = torch.cat([x, enc], dim=1)
            x = self.decoder_blocks[i](x)

        x = self.output_conv(x)

        return x
