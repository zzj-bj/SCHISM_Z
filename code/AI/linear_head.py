"""
LinearHead class for image classification tasks.

This class implements a linear head for image classification tasks.
It includes configurable parameters for the embedding size, number of classes,
and the number of features.

@author: Florent.BRONDOLO
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class LinearHeadConfig:
    """
    LinearHeadConfig Class for Configuring LinearHead

    This class defines the configuration parameters for the LinearHead.
    
    It includes parameters such as embedding size, number of features,
    and number of classes.
    """
    # Z: embedding_size: number of channels of 1 Transformer layer output
    embedding_size: int = 512
    n_features: int = 1
    num_classes: int = 3


class LinearHead(nn.Module):
    """
    LinearHead class for image classification tasks.
    """
    def __init__(self,
                 linear_head_config : LinearHeadConfig
                 ):
        super().__init__()
        # Z: n_features: number of transformer layers used for feature aggregation
        self.n_features = linear_head_config.n_features
        self.embedding_size = linear_head_config.embedding_size * self.n_features
        self.num_classes = linear_head_config.num_classes
        self.head = nn.Sequential(
            nn.BatchNorm2d(self.embedding_size),
            nn.Conv2d(self.embedding_size, self.num_classes, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, inputs):
        """
        Forward pass of the LinearHead.

        Processes the input features and returns the output logits.
        Args:
            inputs (dict): A dictionary containing the input features and image shape.
        Returns:
            torch.Tensor: The output logits after processing the input features.
        """
        feats = inputs["features"]
        img_shape = inputs["image"].shape[-1]
        patch_feature_size = img_shape // 14
        
        if isinstance(feats, list):
            # Z: remove the CLS token at position 0 along the sequence dimension, then concatenate features by channels
            # Z: tensor shape: [batch, sequence_length, embedding_dim]
            feats = torch.cat([f[:, 1:, :] for f in feats], dim=-1)
        else:
            feats = feats[:, 1:, :]
        
        B, S, D = feats.shape
        # Z: permute to [batch, embedding_dim, sequence_length] and reshape to [batch, embedding_dim, height, width]
        x = feats.permute(0,2,1).reshape(-1,self.embedding_size, patch_feature_size, patch_feature_size)

        logits = self.head(x)

        logits = F.interpolate(
            input=logits, size=(int(img_shape),int(img_shape)),
                                mode="bilinear",align_corners=False
        )
        return logits
   