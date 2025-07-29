"""
CNNHead Class for Semantic Segmentation

This class implements a CNN-based head for semantic segmentation tasks.
It includes configurable parameters for the number of blocks, channels, kernel size,
activation functions, and the number of output classes.

@author: Florent.BRONDOLO
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from AI.activation_mixin import ActivationMixin


@dataclass
class CNNHeadConfig:
    """
    CNNHeadConfig Class for Configuring CNNHead

    This class defines the configuration parameters for the CNNHead.
    
    It includes parameters such as embedding size, number of channels,
    number of classes, kernel size, number of features, and activation function.
    """
    embedding_size: int
    channels: int = 512
    num_classes: int = 3
    k_size: int = 3
    n_features: int = 1
    activation: str = 'relu'

class CNNHead(nn.Module, ActivationMixin):
    """
    CNNHead Class for Semantic Segmentation
    """

    def __init__(
        self,
        cnn_head_config : CNNHeadConfig
    ):
        super().__init__()
        self.n_features = cnn_head_config.n_features
        self.embedding_size = cnn_head_config.embedding_size * self.n_features
        self.n_block = 4 # Hardcoded for now
        self.channels = cnn_head_config.channels
        self.k_size = cnn_head_config.k_size
        self.num_classes = cnn_head_config.num_classes
        self.activation = cnn_head_config.activation

        self.input_conv = nn.Conv2d(
            in_channels=self.embedding_size,
            out_channels=self.channels,
            kernel_size=self.k_size,
            padding=1,
        )

        self.decoder_convs = nn.ModuleList()
        self.upscale_fn = ["interpolate", "interpolate",
                           "pixel_shuffle", "pixel_shuffle"]

        for i in range(self.n_block):
            if self.upscale_fn[i] == "interpolate":
                self.decoder_convs.append(
                    self._create_decoder_conv_block(
                        channels=self.channels, kernel_size=self.k_size)
                )
            else:
                channels_by_4 = self.channels // 4
                self.decoder_convs.append(
                    self._create_decoder_up_conv_block(
                        channels=channels_by_4, kernel_size=self.k_size)
                )

        self.seg_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.num_classes,
                      kernel_size=self.k_size, padding=1)
        )

    def _create_decoder_conv_block(self, channels, kernel_size):
        """
        Creates a decoder convolutional block with batch normalization.

        Args:
            channels (int): Number of input/output channels.
            kernel_size (int): Size of the convolutional kernel.
        Returns:
            nn.Sequential: A sequential block containing batch normalization 
            and a convolutional layer.
        """
        return nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
        )

    def _create_decoder_up_conv_block(self, channels, kernel_size):
        """
        Creates a decoder upsampling convolutional block.

        Args:
            channels (int): Number of input/output channels.
            kernel_size (int): Size of the convolutional kernel.
        Returns:
            nn.Sequential: A sequential block containing a pixel shuffle layer 
            and a convolutional layer."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
        )

    def forward(self, inputs):
        """
        Forward pass of the CNNHead.

        Args:
            inputs (dict): Dictionary containing 'features' and 'image'.
                'features' is a list of feature maps, and 'image' is the input image tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the CNNHead.
        """
        features = inputs["features"]
        patch_feature_size = inputs["image"].shape[-1] // 14
        if self.n_features > 1:
            features = torch.cat(features, dim=-1)
        features = features[:, 1:].permute(0, 2, 1).reshape(
            -1, self.embedding_size, patch_feature_size, patch_feature_size
        )
        x = self.input_conv(features)
        for i in range(self.n_block):
            if self.upscale_fn[i] == "interpolate":
                resize_shape = x.shape[-1] * 2 if i >= 1 else x.shape[-1] * 1.75
                x = F.interpolate(input=x, size=(int(resize_shape),
                                                 int(resize_shape)),
                                                 mode="bicubic")
            else:
                x = nn.PixelShuffle(2)(x)
            x = x + self.decoder_convs[i](x)
            if i % 2 == 1 and i != 0:
                x = F.dropout(x, p=0.2)
                x = self._get_activation(self.activation)(x)
        return self.seg_conv(x)
