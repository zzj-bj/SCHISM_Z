import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNHead(nn.Module):
    def __init__(
        self,
        embedding_size,
        n_block=4,
        channels=512,
        num_classes=2,
        k_size=3,
        n_features=1,
    ):
        super(CNNHead, self).__init__()
        self.n_features = n_features
        self.embedding_size = embedding_size * n_features
        self.n_block = n_block
        self.channels = channels
        self.k_size = int(k_size)
        self.num_classes = num_classes

        self.input_conv = nn.Conv2d(
            in_channels=self.embedding_size,
            out_channels=channels,
            kernel_size=self.k_size,
            padding=1,
        )
        self.decoder_convs = nn.ModuleList()
        self.upscale_fn = ["interpolate", "interpolate", "pixel_shuffle", "pixel_shuffle"]

        for i in range(n_block):
            if self.upscale_fn[i] == "interpolate":
                self.decoder_convs.append(
                    self._create_decoder_conv_block(channels=channels, kernel_size=self.k_size)
                )
            else:
                channels = channels // 4
                self.decoder_convs.append(
                    self._create_decoder_up_conv_block(channels=channels, kernel_size=self.k_size)
                )

        self.seg_conv = nn.Sequential(
            nn.Conv2d(channels, num_classes, kernel_size=self.k_size, padding=1)
        )

    def _create_decoder_conv_block(self, channels, kernel_size):
        return nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
        )

    def _create_decoder_up_conv_block(self, channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
        )

    def forward(self, inputs):
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
                x = F.interpolate(input=x, size=(int(resize_shape), int(resize_shape)), mode="bicubic")
            else:
                x = F.pixel_shuffle(x, 2)
            x = x + self.decoder_convs[i](x)
            if i % 2 == 1 and i != 0:
                x = F.dropout(x, p=0.2)
                x = F.leaky_relu(x)
        return self.seg_conv(x)
