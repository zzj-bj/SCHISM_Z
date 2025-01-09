# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:57:46 2024

@author: Florent.BRONDOLO
"""

import torch
import torch.nn as nn


class UnetVanilla(nn.Module):
    REQUIRED_PARAMS = {
        'n_block': int,
        'channels': int,
        'num_classes': int
    }

    OPTIONAL_PARAMS = {
        'k_size': int,  # Kernel size for convolutions
        'activation': str  # Activation function type
    }

    def __init__(self, n_block=4, channels=64, num_classes=2):
        """
        Initializes the UnetVanilla model.

        Args:
            n_block (int): Number of blocks in the U-Net architecture.
            channels (int): Number of channels in the first layer.
            num_classes (int): Number of output classes for segmentation.

        Raises:
            ValueError: If num_classes is less than or equal to 0.
        """
        super(UnetVanilla, self).__init__()
        self.n_block = n_block
        self.channels = channels
        if num_classes <= 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes

        # Define down sampling path
        self.encoder_blocks = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        for i in range(n_block):
            self.encoder_blocks.append(self.encoder_block(channels * (2 ** i)))
            self.max_pools.append(nn.MaxPool2d(kernel_size=2))

        # Define up sampling path
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_block, -1, -1):
            if i > 0:
                self.decoder_blocks.append(self.decoder_block(channels * (2 ** i)))
                self.up_convs.append(nn.ConvTranspose2d(channels * (2 ** i),
                                                        channels * (2 ** i) // 2,
                                                        kernel_size=3,
                                                        stride=2,
                                                        padding=1,
                                                        output_padding=1))

        # Define output layer
        self.output_conv = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

        self.bridge = self.simple_conv(self.channels * (2 ** n_block) // 2, self.channels * (2 ** n_block))
        self.start = self.simple_conv(3, self.channels)

    def simple_conv(self, in_channels, out_channels, k_size=3):
        """
        Creates a simple convolutional block with a convolutional layer followed by ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            k_size (int): Kernel size for the convolutional layer.

        Returns:
            nn.Sequential: A sequential model containing the convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=1),
            nn.ReLU()
        )

    def encoder_block(self, in_channels, k_size=3):
        """
        Creates an encoder block consisting of a convolutional layer, batch normalization, and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            k_size (int): Kernel size for the convolutional layer.

        Returns:
            nn.Sequential: A sequential model containing the encoder block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=k_size, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU()
        )

    def decoder_block(self, in_channels, k_size=3):
        """
        Creates a decoder block consisting of a convolutional layer, batch normalization, and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            k_size (int): Kernel size for the convolutional layer.

        Returns:
            nn.Sequential: A sequential model containing the decoder block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=k_size, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Defines the forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,
                             C is the number of channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes, H_out, W_out), where H_out and W_out
                          are the height and width of the output after processing through the U-Net.
        """
        encodings = []
        x = self.start(x)
        encodings.append(x)

        # Down sampling path
        for i in range(self.n_block):
            x = self.max_pools[i](x)
            if i < self.n_block - 1:
                x = self.encoder_blocks[i](x)
                encodings.append(x)

        x = self.bridge(x)

        # Up sampling path
        for i in range(self.n_block):
            x = self.up_convs[i](x)
            x = torch.cat([x, encodings[-(i + 1)]], dim=1)  # skip connection
            x = self.decoder_blocks[i](x)

        x = self.output_conv(x)

        return x
