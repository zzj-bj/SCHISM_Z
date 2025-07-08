# pylint: disable=too-many-instance-attributes
"""
UnetVanilla: A simple U-Net implementation in PyTorch.

UnetVanilla is a PyTorch module that implements a U-Net architecture for image segmentation.
This module defines a U-Net architecture for image segmentation tasks.
It includes configurable parameters for the number of blocks, channels, kernel size,
activation functions, and the number of output classes.
It consists of encoder blocks, decoder blocks, and a bridge layer.

@author: Florent.BRONDOLO
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from commun.activation_mixin import ActivationMixin

@dataclass
class UnetVanillaConfig:
    """
    UnetVanillaConfig Class for Configuring UnetVanilla

    This class defines the configuration parameters for the UnetVanilla.

    It includes parameters such as number of blocks, number of channels,
    number of classes, kernel size, and activation function."""
    n_block: int = 4
    channels: int = 8
    num_classes: int = 3
    k_size: int = 3
    activation: str = 'relu'

class UnetVanilla(nn.Module,ActivationMixin):
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
                 unet_vanilla_config: UnetVanillaConfig
                 ):
        super().__init__()
        self.config = {
            "n_block" : unet_vanilla_config["n_block"],
            "channels" : unet_vanilla_config["channels"],
            "num_classes" : unet_vanilla_config["num_classes"],
            "k_size" : unet_vanilla_config["k_size"],
            "activation" : unet_vanilla_config["activation"].lower(),
            }

        self.encoder_blocks = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        for i in range(self.config["n_block"]):
            self.encoder_blocks.append(self.encoder_block(self.config["channels"] * (2 ** i)))
            self.max_pools.append(nn.MaxPool2d(kernel_size=2, ceil_mode=True))

        # Define up sampling path
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.config["n_block"], -1, -1):
            if i > 0:
                self.decoder_blocks.append(self.decoder_block(self.config["channels"] * (2 ** i)))
                self.up_convs.append(
                    nn.ConvTranspose2d(
                        self.config["channels"] * (2 ** i),
                        self.config["channels"] * (2 ** i) // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1  # Allow a small correction
                    ))

        # Define output layer
        self.output_conv = nn.Conv2d(
            self.config["channels"],
            self.config["num_classes"],
            kernel_size=1)

        self.bridge = self.simple_conv(
            self.config["channels"] * (2 ** self.config["n_block"])// 2,
            self.config["channels"] * (2 ** self.config["n_block"]))

        self.start = self.simple_conv(3, self.config["channels"])

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
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.config["k_size"],
                padding=self.config["k_size"] // 2),
                self._get_activation(self.config["activation"]
                )
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
            nn.Conv2d(
                in_channels,
                in_channels * 2,
                kernel_size=self.config["k_size"],
                padding=self.config["k_size"] // 2
                ),
            nn.BatchNorm2d(
                in_channels * 2),
                self._get_activation(self.config["activation"]
                )
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
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=self.config["k_size"],
                padding=self.config["k_size"] // 2
                ),
            nn.BatchNorm2d(
                in_channels // 2),
                self._get_activation(self.config["activation"]
                )
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
