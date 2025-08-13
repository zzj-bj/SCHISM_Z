# AI/cnn_head.py
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from AI.activation_mixin import ActivationMixin


@dataclass
class CNNHeadConfig:
    embedding_size: int
    num_classes: int = 3
    n_features: int = 1
    k_size: int = 3
    activation: str = "relu"
    up_factors: Optional[Tuple[float, ...]] = None
    dropout: float = 0.1


class CNNHead(nn.Module):
    def __init__(self, cfg: CNNHeadConfig) -> None:
        super().__init__()
        self.n_features = cfg.n_features
        self.embedding_size = cfg.embedding_size * self.n_features
        self.num_classes = cfg.num_classes
        self.dropout = cfg.dropout
        
        #self.activation_mixin = ActivationMixin()
        #self.activation = self.activation_mixin.activation.lower()
        self.up_factors = cfg.up_factors  # Can be None initially
        
        # Layers will be built dynamically on first forward pass

         # Build layers on first forward pass if not already built
            # Calculate up_factors if not provided
        self.up_factors = self._calculate_up_factors_dynamic(self.n_features, 560)
            
        device = "cuda"
        self._build_layers(device)

    
    def _calculate_up_factors_dynamic(self, n_features: int, img_res: int) -> Tuple[float, ...]:
        """
        ONE simple function to calculate up_factors mathematically.
        
        Logic: 
        - total_factor = img_res / (img_res // 14) = exactly 14
        - Use (n_features-1) factors of 2.0 
        - First factor = 14 / (2^(n_features-1))
        """
        total_factor = 14.0  # Always 14 for DINOv2
        
        if n_features == 1:
            return (total_factor,)
        
        # Use (n_features-1) powers of 2, calculate first factor
        num_twos = n_features - 1
        first_factor = total_factor / (2.0 ** num_twos)
        
        return (first_factor,) + tuple(2.0 for _ in range(num_twos))
    
    def _build_layers(self, device: torch.device):
        """Build all CNN layers - with proper stages and projections."""
        # Check that up_factors is set
        if self.up_factors is None:
            raise ValueError("up_factors must be set before building layers")
        
        # Scale channels based on input features
        initial_channels = self.embedding_size // 2
        
        self.initial_conv = nn.Conv2d(
            in_channels=self.embedding_size,
            out_channels=initial_channels,
            kernel_size=3,
            padding=1
        ).to(device)
        
        # Build stages and projections
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()  # Initialize this!

        current_channels = initial_channels
        for i in range(len(self.up_factors)):
            # Determine next channel count
            if i == len(self.up_factors) - 1:
                # Last stage - prepare for classifier
                next_channels = max(64, current_channels // 2)
            else:
                # Keep same or reduce gradually
                next_channels = current_channels // 2
            
            # Main stage (conv block) - ensure spatial dims are preserved
            stage = nn.Sequential(
                nn.BatchNorm2d(current_channels),
                nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
            ).to(device)
            self.stages.append(stage)
            
            # Projection for residual connection if channels change
            # Use 1x1 conv to preserve spatial dimensions exactly
            if current_channels != next_channels:
                projection = nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=0, bias=False).to(device)
            else:
                projection = nn.Identity().to(device)
            self.projections.append(projection)
            
            # Also build decoder_convs for compatibility
            decoder_block = nn.Sequential(
                nn.BatchNorm2d(current_channels),
                nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
            ).to(device)
            self.decoder_convs.append(decoder_block)
            
            current_channels = next_channels
        
        # Final classifier
        self.classifier = nn.Conv2d(current_channels, self.num_classes, kernel_size=3, padding=1).to(device)
    
    def forward(self, inputs: dict) -> torch.Tensor:
        feats = inputs["features"]
        img = inputs["image"]
        img_res = img.shape[-1]
        patch_sz = img_res // 14
             
        # Handle features
        if isinstance(feats, list):
            feats = torch.cat([f[:, 1:, :] for f in feats], dim=-1)
        else:
            feats = feats[:, 1:, :]
        
        B, S, D = feats.shape
        x = feats.view(B, D, patch_sz, patch_sz)
        # Initial conv
        x = self.initial_conv(x)

        # Calculate all target sizes upfront to avoid accumulation errors
        target_sizes = []
        current_size = patch_sz
        for factor in self.up_factors:
            current_size = int(round(current_size * factor))
            target_sizes.append(current_size)
        
        # Ensure final size matches image resolution
        if target_sizes and target_sizes[-1] != img_res:
            target_sizes[-1] = img_res
        
        # Upsampling stages - smart residual with projections
        for i, target_size in enumerate(target_sizes):
            # Interpolate to exact target size
            x = F.interpolate(x, size=(target_size, target_size), 
                            mode="bicubic", align_corners=False)

            # Apply convolution first
            conv_out = self.stages[i](x)

            # Apply projection for residual connection
            identity = self.projections[i](x)

            # Ensure sizes match before addition (handle any edge cases)
            if identity.shape != conv_out.shape:
                # This shouldn't happen with proper padding, but just in case
                h, w = identity.shape[-2:]
                conv_out = F.interpolate(conv_out, size=(h, w), mode='bilinear', align_corners=False)
            
            x = identity + conv_out

            
            if i % 2 == 1 and i != 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = F.leaky_relu(x)
        
        # Final classifier
        logits = self.classifier(x)

        # Ensure exact size match
        if logits.shape[-2:] != (img_res, img_res):
            logits = F.interpolate(logits, size=(img_res, img_res), 
                                 mode="bicubic", align_corners=False)
        
        return logits