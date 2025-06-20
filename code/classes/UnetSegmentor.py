"""
UnetSegmentor: A U-Net based segmentation model for image segmentation tasks.

This module implements a U-Net architecture for semantic segmentation tasks.
UnetSegmentor is a PyTorch module that implements a U-Net architecture for image segmentation.
It includes configurable parameters for the number of blocks, channels, kernel size,
activation functions, and the number of output classes.
It consists of encoder blocks, decoder blocks, and a bridge layer.

@author: Florent.BRONDOLO
"""
from torch import nn
import torch.nn.functional as nn_func

class UnetSegmentor(nn.Module):
    """
    UnetSegmentor: A U-Net based segmentation model for image segmentation tasks.
    """
    REQUIRED_PARAMS = {
        'num_classes': int
    }

    OPTIONAL_PARAMS = {
        'n_block': int,
        'channels': int,
        'k_size': int,  
        'activation': str, 
        'p': float
    }

    def __init__(self,
                 n_block=4,
                 channels=8,
                 num_classes=3,
                 p=0.5,
                 k_size=3,
                 activation='relu'
                 ):
        super(UnetSegmentor, self).__init__()
        self.n_block = n_block
        self.channels = channels
        self.num_classes = num_classes
        self.p = p
        self.k_size = k_size
        self.activation = activation.lower()
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=self.channels,
                                    kernel_size=self.k_size, padding=1)
        self.encoder_convs = nn.ModuleList([
            self._create_encoder_conv_block(channels= self.channels * 2 ** i)
            for i in range(0, self.n_block - 1)
        ])
        self.mid_conv = self._create_encoder_conv_block(
            channels=self.channels*2**(self.n_block - 1))
        self.decoder_deconvs = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels= self.channels * 2 ** (i + 1),
                out_channels= self.channels * 2 ** i,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
            for i in reversed(range(self.n_block))
        ])
        self.decoder_convs = nn.ModuleList([
            self._create_decoder_conv_block(channels=self.channels * 2 ** i)
            for i in reversed(range(self.n_block))
        ])
        self.seg_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.num_classes,
            kernel_size=3,
            padding=1
        )

    def _create_encoder_conv_block(self, channels):
        """
        Creates a convolutional block for the encoder part of the U-Net.

        Args:
            channels (int): Number of input channels for the convolutional block.
        Returns:
            nn.Sequential: A sequential block containing convolutional layers,
            batch normalization, and the specified activation function.
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=self.k_size, padding=1),
            nn.BatchNorm2d(channels * 2),
            self._get_activation(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=self.k_size, padding=1),
            nn.BatchNorm2d(channels * 2),
            self._get_activation(),
        )

    def _create_decoder_conv_block(self, channels):
        """
        Creates a convolutional block for the decoder part of the U-Net.

        Args:
            channels (int): Number of input channels for the convolutional block.
        Returns:
            nn.Sequential: A sequential block containing convolutional layers,
            batch normalization, and the specified activation function.
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=self.k_size, padding=1),
            nn.BatchNorm2d(channels),
            self._get_activation(),
            nn.Conv2d(channels, channels, kernel_size=self.k_size, padding=1),
            nn.BatchNorm2d(channels),
            self._get_activation(),
        )

    def _get_activation(self):
        """
        Returns the specified activation function.

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
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x):
        """
        Defines the forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width) 
            representing the segmentation map.
        """
        feature_list = []
        x = self.input_conv(x)
        feature_list.append(x)
        x = nn_func.max_pool2d(x, kernel_size=2)
        x = nn_func.dropout(x, p=self.p)
        for i in range(self.n_block - 1):
            x = self.encoder_convs[i](x)
            feature_list.append(x)
            x = nn_func.max_pool2d(x, kernel_size=2)
            x = nn_func.dropout(x, p=self.p)

        x = self.mid_conv(x)

        for i in range(self.n_block):
            x = self.decoder_deconvs[i](x)
            x = nn_func.dropout(x, p=self.p)
            x = self.decoder_convs[i](x + feature_list[::-1][i])

        return self.seg_conv(x)
