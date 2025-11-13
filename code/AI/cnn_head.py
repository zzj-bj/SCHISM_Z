import torch
from torch import nn
from dataclasses import dataclass
from AI.activation_mixin import ActivationMixin
import math

@dataclass
class CNNHeadConfig:
    # Z: embedding_size: single feature embedding size = number of channels of 1 Transformer layer output
    embedding_size: int
    img_res: int
    num_classes: int = 3
    # Z: n_features: number of transformer layers used for feature aggregation
    n_features: int = 1
    n_block: int = 4
    k_size: int = 3
    activation: str = "relu"
    dropout: float = 0.1
    channel_reduction: str = "gradual"
# Z: choose number of groups for GroupNorm based on channels
def _gn_groups(c: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn, kernel_size=3, dropout=0.0):
        super().__init__()
        # Z: equivalent to 'same' padding
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
        # Z: element wise addition for residual connection
        out = out + identity
        return out

class CNNHead(nn.Module):
    def __init__(self, cfg: CNNHeadConfig) -> None:
        super().__init__()
        self.n_features = cfg.n_features
        self.n_block = cfg.n_block
        # Z: total embedding size after concatenating features from multiple layers
        self.embedding_size = cfg.embedding_size * self.n_features
        self.num_classes = cfg.num_classes
        self.dropout = cfg.dropout
        self.channel_reduction = cfg.channel_reduction
        self.img_res = cfg.img_res
        self.activation = cfg.activation.lower()
        self.activation_mixin = ActivationMixin()
        # Z: first element is input channels, others are output channels for each block
        channels_list = [self.embedding_size]
        for i in range(1, self.n_block + 1):
            if self.channel_reduction == "gradual":
                # Z: reduce channels by half each block, minimum 8
                next_c = max(self.embedding_size // (2 ** i), 8)
            elif self.channel_reduction == "aggressive":
                # Z: reduce channels by factor of 4 each block, minimum 8
                next_c = max(self.embedding_size // (4 ** i), 8)
            else:
                # Z: keep channels constant
                next_c = self.embedding_size
            channels_list.append(next_c)
        # Z: In ViT one image is diveded into 14*14 patches
        dinov2_feat_size = self.img_res // 14
        scale_factors = self.compute_scale_factors(dinov2_feat_size, self.img_res, self.n_block)
        layers = []
        act_fn = self.activation_mixin._get_activation(self.activation)
        for i in range(self.n_block):
            layers.append(
                ConvBlock(
                    channels_list[i],
                    channels_list[i + 1],
                    activation_fn=act_fn,
                    kernel_size=self.k_size,
                    dropout=self.dropout
                )
            )
            # Z: if not the last block
            if i < self.n_block - 1:
                layers.append(
                    # Z: upsample by the computed scale factor
                    nn.Upsample(scale_factor=scale_factors[i], mode="bilinear", align_corners=False)
                )
            else:
                layers.append(
                    nn.Upsample(size=(self.img_res, self.img_res), mode="bilinear", align_corners=False)
                )
        # Z: sequentially stack all layers
        self.features = nn.Sequential(*layers)
        # Z: use 1x1 conv to map channels to num_classes
        self.classifier = nn.Conv2d(channels_list[-1], self.num_classes, kernel_size=1)

    def compute_scale_factors(self, input_size, output_size, n_block):
        """
        Z: determine the upscaling factor for each upsampling layer. It first estimates a uniform scaling ratio
        based on the input and output resolutions and the number of remaining blocks,
        giving priority to a *3 or *2 factor and accumulating them step by step.
        The final factor is then automatically adjusted to an exact value to ensure that,
        after all upscaling, the output size matches the target resolution precisely.
        """
        factors = []
        current = input_size
        for i in range(self.n_block):
            remaining = output_size / current
            blocks_left = self.n_block - i
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
        for i in range(self.n_block - 1):
            cumulative *= factors[i]
        factors[-1] = output_size / cumulative
        return factors

    def forward(self, inputs: dict) -> torch.Tensor:
        feats = inputs["features"]
        img = inputs["image"]
        img_res = img.shape[-1]
        patch_sz = img_res // 14

        # Z: if multiple outputs from different transformer layers are provided
        # Z: remove the CLS token at position 0 along the sequence dimension, only keep patch tokens
        # Z: then concatenate features by channels/embeddings
        # Z: tensor shape: [batch, sequence_length, embedding_dim]
        feats = torch.cat([f[:, 1:, :] for f in feats], dim=-1) if isinstance(feats, list) else feats[:, 1:, :]
        B, S, D = feats.shape
        assert S == patch_sz * patch_sz, f"S={S} patch={patch_sz}"
        # Z: permute to [batch, embedding_dim, sequence_length] and reshape to [batch, embedding_dim, patch height, patch width]
        x = feats.permute(0,2,1).reshape(-1,self.embedding_size, patch_sz, patch_sz)    
        out = x

        for b in range(self.n_block):
            # Z: extract conv and upsample layers from the sequential container
            conv = self.features[2*b]
            # Z: extract upsample layer from the sequential container
            up  = self.features[2*b + 1]
            out = conv(out)
            out = up(out)
    
        out = self.classifier(out)

        return out
