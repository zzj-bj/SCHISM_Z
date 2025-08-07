
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import nn
import torch.nn.functional as F
from AI.activation_mixin import ActivationMixin

@dataclass
class CNNHeadConfig:
    """
    Configuration for the CNNHead. All formerly "magic" numbers are now fields here.
    """
    embedding_size: int                # size of the incoming feature vector
    n_features: int = 1                # how many feature maps to concat
    conv_channels: List[int] = (1024, 512, 256, 128)  
                                       # sequence of channel counts for conv blocks
    fc_channels: List[int] = (128,)    # channels for the final conv before output
    num_classes: int = 3
    block_kernel: int = 3               # kernel size for conv blocks
    fc_kernel: int = 3                  # kernel size for final conv
    pooling_out: int = 1                # output size for AdaptiveAvgPool2d
    activation: str = 'relu'            # activation name
    
    @property
    def total_embedding(self) -> int:
        # after concatenating n_features of embeddings
        return self.embedding_size * self.n_features

class CNNHead(nn.Module, ActivationMixin):
    """
    CNN-based head for semantic segmentation, fully driven by CNNHeadConfig.
    """
    def __init__(self, cfg: CNNHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.activation_mixin = ActivationMixin()
        
        in_channels = cfg.total_embedding
        layers: List[nn.Module] = []
        
        # build conv blocks
        for out_ch in cfg.conv_channels:
            layers.append(nn.Conv2d(
                in_channels, out_ch,
                kernel_size=cfg.block_kernel, padding=cfg.block_kernel // 2,
                bias=False
            ))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(self.activation_mixin._get_activation(cfg.activation)(inplace=True))
            in_channels = out_ch
        
        # global pooling to a fixed spatial size
        layers.append(nn.AdaptiveAvgPool2d(cfg.pooling_out))
        self.conv = nn.Sequential(*layers)
        
        # final classifier conv(s)
        fc_in = cfg.conv_channels[-1]
        fc_layers = []
        for out_ch in cfg.fc_channels:
            fc_layers.append(nn.Conv2d(
                fc_in, out_ch,
                kernel_size=cfg.fc_kernel, padding=cfg.fc_kernel // 2
            ))
            fc_layers.append(self.activation_mixin._get_activation(cfg.activation)(inplace=True))
            fc_in = out_ch
        
        # output conv to num_classes
        fc_layers.append(nn.Conv2d(
            fc_in, cfg.num_classes,
            kernel_size=cfg.fc_kernel, padding=cfg.fc_kernel // 2
        ))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = inputs["features"]
        image   = inputs["image"]
        
        # flatten list of features and remove CLS tokens
        if isinstance(features, list):
            feats = [f[:,1:,:] for f in features]
            feats = torch.cat(feats, dim=-1)
        else:
            feats = features[:,1:,:]
        
        B, S, D = feats.shape
        # assume features form a sqrt grid
        patch_size = image.shape[-1] // int(S**0.5)
        
        x = feats.permute(0,2,1).contiguous() \
             .view(B, D, patch_size, patch_size)
        
        x = self.conv(x)               # [B, last_conv_ch, 1, 1]
        x = x.view(B, -1, 1, 1)        # preserve as 4D for conv
        x = self.fc(x)                 # [B, num_classes, 1, 1]
        
        return x.squeeze(-1).squeeze(-1)  # [B, num_classes]
