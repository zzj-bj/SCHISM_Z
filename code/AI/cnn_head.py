# AI/cnn_head.py
from dataclasses import dataclass
from typing import Tuple, Optional, List
import torch
from torch import nn
import torch.nn.functional as F
import math
from AI.activation_mixin import ActivationMixin


@dataclass
class CNNHeadConfig:
    embedding_size: int
    num_classes: int = 3
    n_features: int = 1
    n_blocks: int = 4  # Number of CNN blocks to stack
    k_size: int = 3
    activation: str = "relu"
    dropout: float = 0.1
    channel_reduction: str = "gradual"  # "gradual", "aggressive", "maintain"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


class CNNHead(nn.Module):
    def __init__(self, cfg: CNNHeadConfig) -> None:
        super().__init__()
        self.n_features = cfg.n_features
        self.n_blocks = cfg.n_blocks
        self.embedding_size = cfg.embedding_size * self.n_features
        self.num_classes = cfg.num_classes
        self.dropout = cfg.dropout
        self.activation = cfg.activation.lower()
        self.channel_reduction = cfg.channel_reduction
        self.activation_mixin = ActivationMixin()
        
        # Smart calculation of upsampling factors
        self.up_factors = self._calculate_smart_up_factors(
            n_blocks=self.n_blocks,
            n_features=self.n_features
        )
        
        print(f"Upsampling factors for {self.n_blocks} blocks: {self.up_factors}")
        
        self._build_layers(cfg.device)
    
    def _calculate_smart_up_factors(self, n_blocks: int, n_features: int) -> Tuple[float, ...]:
        """
        Intelligently calculate upsampling factors based on multiple parameters.
        
        Strategy:
        1. Prefer powers of 2 (2x, 4x) for clean upsampling
        2. Consider n_features to balance early vs late upsampling
        3. Minimize fractional factors
        4. Put any odd factors at strategic positions
        """
        total_factor = 14.0  # Always 14 for DINOv2 (from patch to image)
        
        if n_blocks == 1:
            return (total_factor,)
        
        # Define preferred factor combinations for common cases
        factor_strategies = {
            # (n_blocks, n_features): factors
            (2, 1): (2.0, 7.0),  # Early small, late large
            (2, 2): (4.0, 3.5),  # More balanced
            (2, 3): (3.5, 4.0),  # Inverse for multiple features
            (2, 4): (7.0, 2.0),  # Large early for many features
            
            (3, 1): (2.0, 2.0, 3.5),  # Gradual increase
            (3, 2): (2.0, 3.5, 2.0),  # Peak in middle
            (3, 3): (3.5, 2.0, 2.0),  # Front-loaded
            (3, 4): (2.0, 2.0, 3.5),  # Back-loaded
            
            (4, 1): (2.0, 2.0, 2.0, 1.75),  # Most upsampling early
            (4, 2): (1.75, 2.0, 2.0, 2.0),  # Most upsampling late
            (4, 3): (2.0, 1.75, 2.0, 2.0),  # Slight dip in second
            (4, 4): (1.75, 2.0, 2.0, 2.0),  # Progressive increase
            
            (5, 1): (2.0, 2.0, 1.75, 1.0, 2.0),  # Distributed
            (5, 2): (1.4, 2.0, 2.0, 1.25, 2.0),
            (5, 3): (1.75, 1.5, 1.5, 1.75, 2.0),
            (5, 4): (1.4, 1.4, 2.0, 2.0, 1.75),
            
            (6, 1): (1.5, 1.5, 1.5, 1.5, 1.5, 2.07),
            (6, 2): (1.4, 1.4, 1.4, 2.0, 1.5, 1.75),
            (6, 3): (1.5, 1.5, 1.4, 1.4, 2.0, 1.75),
            (6, 4): (1.2, 1.5, 1.5, 1.5, 2.0, 2.0),
        }
        
        # Check if we have a predefined strategy
        key = (n_blocks, min(n_features, 4))  # Cap n_features at 4 for lookup
        if key in factor_strategies:
            factors = factor_strategies[key]
            # Verify the product is approximately 14
            product = math.prod(factors)
            if abs(product - 14.0) > 0.1:
                # Adjust last factor to ensure exact 14x
                factors = list(factors)
                factors[-1] = factors[-1] * (14.0 / product)
                factors = tuple(factors)
            return factors
        
        # Fallback: Smart distribution for arbitrary n_blocks
        factors = self._distribute_factors_smartly(n_blocks, total_factor)
        return factors
    
    def _distribute_factors_smartly(self, n_blocks: int, total: float) -> Tuple[float, ...]:
        """
        Distribute total factor across n_blocks intelligently.
        
        Strategy: Use powers of 2 where possible, put remainder in one block.
        """
        factors = []
        remaining = total
        
        # Calculate how many 2x blocks we can fit
        num_twos = min(n_blocks - 1, int(math.log2(total)))
        
        # Add 2x factors
        for _ in range(num_twos):
            factors.append(2.0)
            remaining /= 2.0
        
        # Distribute remainder across remaining blocks
        remaining_blocks = n_blocks - num_twos
        if remaining_blocks == 1:
            factors.append(remaining)
        else:
            # Split remainder roughly equally
            base_factor = remaining ** (1.0 / remaining_blocks)
            for i in range(remaining_blocks):
                if i == remaining_blocks - 1:
                    # Last factor absorbs any rounding error
                    current_product = math.prod(factors)
                    factors.append(total / current_product)
                else:
                    factors.append(base_factor)
        
        return tuple(factors)
    
    def _get_channel_schedule(self, initial_channels: int) -> List[int]:
        """
        Create a channel reduction schedule based on strategy.
        """
        schedule = [initial_channels]
        current = initial_channels
        
        for i in range(self.n_blocks):
            if self.channel_reduction == "maintain":
                next_channels = current
            elif self.channel_reduction == "aggressive":
                # Halve every block until minimum
                next_channels = max(64, current // 2)
            else:  # gradual
                # Halve every other block
                if i % 2 == 0 and i > 0:
                    next_channels = max(64, current // 2)
                else:
                    next_channels = current
            
            schedule.append(next_channels)
            current = next_channels
        
        return schedule[1:]  # Skip the initial channel count
    
    def _build_layers(self, device: torch.device):
        """Build all CNN layers with proper stages and projections."""
        # Scale channels based on input features
        initial_channels = max(128, self.embedding_size // 2)
        
        self.initial_conv = nn.Conv2d(
            in_channels=self.embedding_size,
            out_channels=initial_channels,
            kernel_size=3,
            padding=1
        ).to(device)
        
        # Get channel schedule
        channel_schedule = self._get_channel_schedule(initial_channels)
        
        # Build stages and projections based on n_blocks
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()

        current_channels = initial_channels
        for i in range(self.n_blocks):
            next_channels = channel_schedule[i]
            
            # Main stage (conv block)
            # Add more layers in middle blocks for better feature extraction
            if i == self.n_blocks // 2:
                # Middle block gets extra processing
                stage = nn.Sequential(
                    nn.BatchNorm2d(current_channels),
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(current_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
                ).to(device)
            else:
                # Standard block
                stage = nn.Sequential(
                    nn.BatchNorm2d(current_channels),
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
                ).to(device)
            self.stages.append(stage)
            
            # Projection for residual connection if channels change
            if current_channels != next_channels:
                projection = nn.Conv2d(current_channels, next_channels, 
                                     kernel_size=1, padding=0, bias=False).to(device)
            else:
                projection = nn.Identity().to(device)
            self.projections.append(projection)
            
            current_channels = next_channels
   
        
        self.classifier = nn.Conv2d(current_channels, self.num_classes, 
                                   kernel_size=1, padding=0).to(device)
    
    def forward(self, inputs: dict) -> torch.Tensor:
        feats = inputs["features"]
        img = inputs["image"]
        img_res = img.shape[-1]
        patch_sz = img_res // 14
        
        # Handle features
        if isinstance(feats, list):
            # Multi-scale features - concatenate
            feats = torch.cat([f[:, 1:, :] for f in feats], dim=-1)
        else:
            feats = feats[:, 1:, :]
        
        B, S, D = feats.shape
        x = feats.view(B, D, patch_sz, patch_sz)
        
        # Initial conv
        x = self.initial_conv(x)
        
        # Calculate exact target sizes
        target_sizes = []
        current_size = float(patch_sz)
        
        for factor in self.up_factors:
            current_size = current_size * factor
            # Round to nearest integer but ensure we don't exceed image size
            target_size = min(int(round(current_size)), img_res)
            target_sizes.append(target_size)
        
        # Ensure final size exactly matches image resolution
        if target_sizes:
            target_sizes[-1] = img_res
        
        # Progressive upsampling with residual connections
        for i, target_size in enumerate(target_sizes):
            # Upsample
            if x.shape[-1] != target_size:
                x = F.interpolate(x, size=(target_size, target_size), 
                                mode="bilinear", align_corners=False)
            
            # Store for residual
            identity = self.projections[i](x)
            
            # Apply convolution block
            conv_out = self.stages[i](x)
            
            # Residual connection
            x = identity + conv_out
     
            # Dropout on specific layers
            if i % 2 == 1 and i != 0 and self.training:
                x = F.dropout2d(x, p=self.dropout, training=self.training)
            
            # Activation
            activation_fn = self.activation_mixin._get_activation(self.activation)
            x = activation_fn(x)
            
        # Final classification
        logits = self.classifier(x)
        
        # Final size adjustment if needed
        if logits.shape[-2:] != (img_res, img_res):
            logits = F.interpolate(logits, size=(img_res, img_res), 
                                 mode="bilinear", align_corners=False)
        
        return logits