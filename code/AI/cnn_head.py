import torch
from torch import nn
from dataclasses import dataclass
from AI.activation_mixin import ActivationMixin
import math

@dataclass
class CNNHeadConfig:
    embedding_size: int
    img_res: int
    num_classes: int = 3
    n_features: int = 1
    n_blocks: int = 4
    k_size: int = 3
    activation: str = "relu"
    dropout: float = 0.5
    channel_reduction: str = "gradual"

def _gn_groups(c: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn, kernel_size=3, dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.activation = activation_fn
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)


    def forward(self, x):
        identity = self.proj(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.activation(self.norm2(self.conv2(out)))
        out = out + identity
        return out

class CNNHead(nn.Module):
    def __init__(self, cfg: CNNHeadConfig) -> None:
        super().__init__()
        self.n_features = cfg.n_features
        self.n_blocks = cfg.n_blocks
        self.embedding_size = cfg.embedding_size * self.n_features
        self.num_classes = cfg.num_classes
        self.dropout = cfg.dropout
        self.k_size = cfg.k_size
        self.channel_reduction = cfg.channel_reduction
        self.img_res = cfg.img_res
        self.activation = cfg.activation.lower()
        self.activation_mixin = ActivationMixin()

        channels_list = [self.embedding_size]
        for i in range(1, self.n_blocks + 1):
            if self.channel_reduction == "gradual":
                next_c = max(self.embedding_size // (2 ** i), 8)
            elif self.channel_reduction == "aggressive":
                next_c = max(self.embedding_size // (4 ** i), 8)
            else:
                next_c = self.embedding_size
            channels_list.append(next_c)
        
        print(channels_list)
        dinov2_feat_size = self.img_res // 14
        scale_factors = self.compute_scale_factors(dinov2_feat_size, self.img_res, self.n_blocks)

        layers = []
        act_fn = self.activation_mixin._get_activation(self.activation)
        for i in range(self.n_blocks):
            layers.append(
                ConvBlock(
                    channels_list[i],
                    channels_list[i + 1],
                    activation_fn=act_fn,
                    kernel_size=self.k_size,
                    dropout=self.dropout
                )
            )
            if i < self.n_blocks - 1:
                layers.append(
                    nn.Upsample(scale_factor=scale_factors[i], mode="bilinear", align_corners=False)
                )
            else:
                layers.append(
                    nn.Upsample(size=(self.img_res, self.img_res), mode="bilinear", align_corners=False)
                )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(channels_list[-1], self.num_classes, kernel_size=1)
      
    def compute_scale_factors(self, input_size, output_size, n_blocks):
        factors = []
        current = input_size
        for i in range(n_blocks):
            remaining = output_size / current
            blocks_left = n_blocks - i
            if blocks_left == 1:
                f = remaining
            else:
                target = remaining ** (1 / blocks_left)
                chosen = None
                for test in [3, 2]:  # prefer 3 then 2
                    if target >= test and current * (test ** blocks_left) <= output_size * 1.001:
                        chosen = test
                        break
                f = chosen if chosen is not None else max(2.0, round(target))
            factors.append(float(f))
            current *= f
        cumulative = input_size
        for i in range(n_blocks - 1):
            cumulative *= factors[i]
        factors[-1] = output_size / cumulative
        return factors

    def forward(self, inputs: dict) -> torch.Tensor:
        feats = inputs["features"]
        img = inputs["image"]
        img_res = img.shape[-1]
        patch_sz = img_res // 14
    
        feats = torch.cat([f[:, 1:, :] for f in feats], dim=-1) if isinstance(feats, list) else feats[:, 1:, :]
        B, S, D = feats.shape
        assert S == patch_sz * patch_sz, f"S={S} patch={patch_sz}"
        x = feats.view(B, D, patch_sz, patch_sz)
        #print(f"in {x.shape}")
    
        out = x
        for b in range(self.n_blocks):
            conv = self.features[2*b]
            up   = self.features[2*b + 1]
            #print(f"block {b} in {out.shape}")
            out = conv(out)
            #print(f"block {b} after conv {out.shape}")
            out = up(out)
            #print(f"block {b} after up {out.shape}")
    
        out = self.classifier(out)
        #print(f"class out {out.shape}")
        return out
