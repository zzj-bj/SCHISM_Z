# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:57:46 2024

@author: Florent.BRONDOLO
"""

import torch
import torch.nn as nn


class UNet_vanilla(nn.Module):
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
        super(UNet_vanilla, self).__init__()
        self.n_block = n_block
        self.channels = channels
        if num_classes <=2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
            
        # Define downsampling path
        self.encoder_blocks = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        for i in range(n_block):
            self.encoder_blocks.append(self.encoder_block(channels * (2 ** i)))
            self.max_pools.append(nn.MaxPool2d(kernel_size=2))

        # Define upsampling path
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_block , -1, -1):
            if i>0:
              self.decoder_blocks.append(self.decoder_block(channels * (2 ** i)))
              self.up_convs.append(nn.ConvTranspose2d(channels * (2**i), 
                                                      channels * (2**i)//2, 
                                                      kernel_size=3, 
                                                      stride=2, 
                                                      padding=1, 
                                                      output_padding=1))

        # Define output layer
        self.output_conv = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

        self.bridge = self.simple_conv(self.channels * (2 ** n_block)//2, self.channels * (2 ** n_block))
        self.start = self.simple_conv(3, self.channels)

    def simple_conv(self, in_channels, out_channels, k_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=1),
            nn.ReLU()
        )

    def encoder_block(self, in_channels, k_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2,  kernel_size=k_size, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU()
        )

    def decoder_block(self, in_channels, k_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=k_size, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )

    def forward(self, x):
        encodings = []
        x = self.start(x)
        encodings.append(x)

        # Downsampling path
        for i in range(self.n_block):
            x = self.max_pools[i](x)
            if i < self.n_block - 1:
              x = self.encoder_blocks[i](x)
              encodings.append(x)

        x = self.bridge(x)

        # Upsampling path
        for i in range(self.n_block):
            x = self.up_convs[i](x)
            x = torch.cat([x, encodings[-(i + 1)]], dim=1)  # skip connection
            x = self.decoder_blocks[i](x)

        x= self.output_conv(x)

        return x