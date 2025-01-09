# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:02:53 2024

@author: Florent.BRONDOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as nn_func


class UnetSegmentor(nn.Module):
    def __init__(self, n_blocks=4, n_filter=32, num_classes=1, p=0.5):
        """
        Initializes the U-Net segmentation model.

        Args:
            n_blocks (int): Number of blocks in the U-Net architecture. Default is 4.
            n_filter (int): Number of filters in the first convolutional layer. Default is 32.
            num_classes (int): Number of output classes for segmentation. Default is 1.
            p (float): Dropout probability. Default is 0.5.
        """
        super(UnetSegmentor, self).__init__()
        self.n_blocks = n_blocks
        self.p = p
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=n_filter, kernel_size=3, padding=1)
        self.encoder_convs = nn.ModuleList([
            self._create_encoder_conv_block(channels=n_filter * 2 ** i, kernel_size=3)
            for i in range(0, n_blocks - 1)
        ])
        self.mid_conv = self._create_encoder_conv_block(
            channels=n_filter * 2 ** (n_blocks - 1),
            kernel_size=3
        )
        self.decoder_deconvs = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=n_filter * 2 ** (i + 1),
                out_channels=n_filter * 2 ** i,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
            for i in reversed(range(n_blocks))
        ])
        self.decoder_convs = nn.ModuleList([
            self._create_decoder_conv_block(
                channels=n_filter * 2 ** i,
                kernel_size=3
            )
            for i in reversed(range(n_blocks))
        ])
        self.seg_conv = nn.Conv2d(
            in_channels=n_filter,
            out_channels=num_classes,
            kernel_size=3,
            padding=1
        )

    def _create_encoder_conv_block(self, channels, kernel_size):
        """
        Creates a convolutional block for the encoder part of the U-Net.

        Args:
            channels (int): Number of input channels for the convolutional block.
            kernel_size (int): Size of the convolutional kernel.

        Returns:
            nn.Sequential: A sequential block containing convolutional layers,
            batch normalization, and ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
        )

    def _create_decoder_conv_block(self, channels, kernel_size):
        """
        Creates a convolutional block for the decoder part of the U-Net.

        Args:
            channels (int): Number of input channels for the convolutional block.
            kernel_size (int): Size of the convolutional kernel.

        Returns:
            nn.Sequential: A sequential block containing convolutional layers,
            batch normalization, and ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Defines the forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width) representing
            the segmentation map.
        """
        feature_list = []
        x = self.input_conv(x)
        feature_list.append(x)
        x = nn_func.max_pool2d(x, kernel_size=2)
        x = nn_func.dropout(x, p=self.p)
        for i in range(self.n_blocks - 1):
            x = self.encoder_convs[i](x)
            feature_list.append(x)
            x = nn_func.max_pool2d(x, kernel_size=2)
            x = nn_func.dropout(x, p=self.p)

        x = self.mid_conv(x)

        for i in range(self.n_blocks):
            x = self.decoder_deconvs[i](x)
            x = nn_func.dropout(x, p=self.p)
            x = self.decoder_convs[i](x + feature_list[::-1][i])

        return self.seg_conv(x)
