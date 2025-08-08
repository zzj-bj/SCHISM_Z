# AI/cnn_head.py

from dataclasses import dataclass
import torch
from torch import nn
from AI.activation_mixin import ActivationMixin

@dataclass
class CNNHeadConfig:
    embedding_size: int
    channels: int = 256      # width of the head, comes from the backbone config
    num_classes: int = 3
    k_size: int = 3
    n_features: int = 1
    activation: str = 'relu'

class CNNHead(nn.Module):
    """
    Segmentation head
    produces logits with shape [B, num_classes, patch_sz, patch_sz]
    uses cfg.channels to size the conv stack, no hardcoded widths
    """
    def __init__(self, cfg: CNNHeadConfig) -> None:
        super().__init__()
        self.n_features     = cfg.n_features
        self.embedding_size = cfg.embedding_size * self.n_features
        self.base_ch        = cfg.channels
        self.num_classes    = cfg.num_classes
        self.k_size         = cfg.k_size
        self.activation     = cfg.activation

        self.activation_mixin = ActivationMixin()

        if self.n_features == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(self.embedding_size, self.base_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.base_ch),
                self.activation_mixin._get_activation(self.activation),

                nn.Conv2d(self.base_ch, self.base_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.base_ch),
                self.activation_mixin._get_activation(self.activation),
            )
            final_in = self.base_ch
        else:
            ch1 = max(32, self.base_ch * 2)
            ch2 = max(32, self.base_ch)
            ch3 = max(32, self.base_ch // 2)

            self.conv = nn.Sequential(
                nn.Conv2d(self.embedding_size, ch1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch1),
                self.activation_mixin._get_activation(self.activation),

                nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch2),
                self.activation_mixin._get_activation(self.activation),

                nn.Conv2d(ch2, ch3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch3),
                self.activation_mixin._get_activation(self.activation),
            )
            final_in = ch3

        self.classifier = nn.Conv2d(
            final_in,
            self.num_classes,
            kernel_size=self.k_size,
            padding=self.k_size // 2,
            bias=True
        )
    
    def forward(self, inputs: dict) -> torch.Tensor:
        feats = inputs["features"]
        img   = inputs["image"]
        patch_sz = img.shape[-1] // 14
    
        if isinstance(feats, list):
            feats = torch.cat([f[:, 1:, :] for f in feats], dim=-1)
        else:
            feats = feats[:, 1:, :]
    
        B, S, D = feats.shape
        feats = feats.permute(0, 2, 1).contiguous()
        feats = feats.view(B, D, patch_sz, patch_sz)
    
        x = self.conv(feats)                      # [B, final_in, patch_sz, patch_sz]
        logits = self.classifier(x)               # [B, num_classes, patch_sz, patch_sz]

        # make logits match labels spatial size
        if logits.shape[-2:] != img.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=img.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits
