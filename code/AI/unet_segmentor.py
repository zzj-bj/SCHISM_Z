"""
UnetSegmentor: A U-Net based segmentation model for image segmentation tasks.

This module implements a U-Net architecture for semantic segmentation tasks.
UnetSegmentor is a PyTorch module that implements a U-Net architecture for image segmentation.
It includes configurable parameters for the number of blocks, channels, kernel size,
activation functions, and the number of output classes.
It consists of encoder blocks, decoder blocks, and a bridge layer.

@author: Florent.BRONDOLO
"""
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as nn_func
from AI.activation_mixin import ActivationMixin

@dataclass
class UnetSegmentorConfig:
    """
    LinearHeadConfig Class for Configuring CNNHead

    This class defines the configuration parameters for the CNNHead.
    
    It includes parameters such as embedding size, number of channels,
    number of classes, kernel size, number of features, and activation function.
    """
    num_classes:int = 3
    n_block: int=4
    channels: int=8
    k_size:int =3
    activation:int ='relu'
    p:int =0.5

class UnetSegmentor(nn.Module,ActivationMixin):
    """
    UnetSegmentor: A U-Net based segmentation model for image segmentation tasks.
    """
    REQUIRED_PARAMS = {
        "num_classes": int
    }

    OPTIONAL_PARAMS = {
        "n_block": int,
        "channels": int,
        "k_size": int,  
        "activation": str, 
        "p": float
    }

    def __init__(self,
                 unet_segmentor_config: UnetSegmentorConfig
                 ):
        super().__init__()
        self.n_block = unet_segmentor_config.n_block
        self.channels = unet_segmentor_config.channels
        self.num_classes = unet_segmentor_config.num_classes
        self.p = unet_segmentor_config.p
        self.k_size = unet_segmentor_config.k_size
        self.activation = unet_segmentor_config.activation.lower()
        self.activation_mixin = ActivationMixin()

        self.input_conv = nn.Conv2d(in_channels=3, out_channels=self.channels,
                                kernel_size=self.k_size, padding=1)

        self.encoder_convs = nn.ModuleList([
                                self._create_encoder_conv_block(
                                    channels= self.channels * 2 ** i)
                                    # Z: n_block - 1 because the last block is the bridge
                                    for i in range(0, self.n_block - 1)])
        # Z: Bridge layer
        self.mid_conv = self._create_encoder_conv_block(
            channels=self.channels*2**(self.n_block - 1))

        self.decoder_deconvs = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels = self.channels * 2 ** (i + 1),
                out_channels = self.channels * 2 ** i,
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
            self.activation_mixin._get_activation(self.activation),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=self.k_size, padding=1),
            nn.BatchNorm2d(channels * 2),
            self.activation_mixin._get_activation(self.activation),
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
            self.activation_mixin._get_activation(self.activation),
            nn.Conv2d(channels, channels, kernel_size=self.k_size, padding=1),
            nn.BatchNorm2d(channels),
            self.activation_mixin._get_activation(self.activation),
        )

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
        # Z: size/2
        x = nn_func.max_pool2d(x, kernel_size=2)
        x = nn_func.dropout(x, p=self.p)
        for i in range(self.n_block - 1):
            # Z: channels*2 for each block
            x = self.encoder_convs[i](x)
            feature_list.append(x)
            # Z: size/2 for each block
            x = nn_func.max_pool2d(x, kernel_size=2)
            x = nn_func.dropout(x, p=self.p)

        x = self.mid_conv(x)

        for i in range(self.n_block):
            # Z: size*2, channels/2 for each block
            x = self.decoder_deconvs[i](x)
            x = nn_func.dropout(x, p=self.p)
            # Z: element-wise addition with corresponding encoder feature map then decoder conv
            x = self.decoder_convs[i](x + feature_list[::-1][i])

        return self.seg_conv(x)
