# AI/cnn_head.py

from dataclasses import dataclass
import torch
from torch import nn
from AI.activation_mixin import ActivationMixin

@dataclass
class CNNHeadConfig:
    embedding_size: int
    channels: int = 512
    num_classes: int = 3
    k_size: int = 3
    n_features: int = 1
    activation: str = 'relu'

class CNNHead(nn.Module):
    def __init__(self, cfg: CNNHeadConfig) -> None:
        super().__init__()
        # base params
        self.n_features     = cfg.n_features
        self.embedding_size = cfg.embedding_size * self.n_features
        self.base_ch        = cfg.channels
        self.num_classes    = cfg.num_classes
        self.k_size         = cfg.k_size
        self.activation     = cfg.activation

        # instantiate mixin for activations
        self.activation_mixin = ActivationMixin()

        # build conv stack
        if self.n_features == 1:
            # single‐block head
            self.conv = nn.Sequential(
                nn.Conv2d(self.embedding_size, self.base_ch,
                          kernel_size=3, stride=1, padding=1),
                self.activation_mixin._get_activation(self.activation),
                nn.AdaptiveAvgPool2d(1)
            )
            final_in = self.base_ch
        else:
            # multi‐block head: embedding → 2x → 1x → 0.5x → 0.25x
            ch1 = self.base_ch * 2
            ch2 = self.base_ch
            ch3 = self.base_ch // 2
            ch4 = max(1, self.base_ch // 4)

            self.conv = nn.Sequential(
                nn.Conv2d(self.embedding_size, ch1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch1),
                self.activation_mixin._get_activation(self.activation, inplace=True),

                nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch2),
                self.activation_mixin._get_activation(self.activation, inplace=True),

                nn.Conv2d(ch2, ch3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch3),
                self.activation_mixin._get_activation(self.activation, inplace=True),

                nn.Conv2d(ch3, ch4, kernel_size=1, bias=False),  # bottleneck
                nn.BatchNorm2d(ch4),
                self.activation_mixin._get_activation(self.activation, inplace=True),

                nn.AdaptiveAvgPool2d(1)
            )
            final_in = ch4

        # final classifier conv
        self.fc = nn.Conv2d(
            final_in,
            self.num_classes,
            kernel_size=self.k_size,
            padding=self.k_size // 2
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        feats = inputs["features"]
        img   = inputs["image"]
        # derive patch size
        patch_sz = img.shape[-1] // 14

        # concatenate features (drop CLS tokens)
        if isinstance(feats, list):
            trimmed = [f[:, 1:, :] for f in feats]
            feats = torch.cat(trimmed, dim=-1)
        else:
            feats = feats[:, 1:, :]

        B, S, D = feats.shape
        feats = feats.permute(0, 2, 1).contiguous()
        feats = feats.view(B, D, patch_sz, patch_sz)

        x = self.conv(feats)
        x = x.view(x.size(0), -1, 1, 1)  # keep conv2d shape
        return self.fc(x)
