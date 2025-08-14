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
        self.memory_efficient = cfg.memory_efficient
        self.activation_mixin = ActivationMixin()
        
        # Smart calculation of upsampling factors
        self.up_factors = self._calculate_smart_up_factors(
            n_blocks=self.n_blocks,
            n_features=self.n_features
        )
        
    
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
        # FIXED: Ensure all products equal exactly 14
        factor_strategies = {
            # (n_blocks, n_features): factors
            (2, 1): (2.0, 7.0),  # 2 * 7 = 14
            (2, 2): (4.0, 3.5),  # 4 * 3.5 = 14
            (2, 3): (3.5, 4.0),  # 3.5 * 4 = 14
            (2, 4): (7.0, 2.0),  # 7 * 2 = 14
            
            (3, 1): (2.0, 2.0, 3.5),  # 2 * 2 * 3.5 = 14
            (3, 2): (2.0, 3.5, 2.0),  # 2 * 3.5 * 2 = 14
            (3, 3): (3.5, 2.0, 2.0),  # 3.5 * 2 * 2 = 14
            (3, 4): (2.0, 2.0, 3.5),  # 2 * 2 * 3.5 = 14
            
            (4, 1): (2.0, 2.0, 2.0, 1.75),  # 2 * 2 * 2 * 1.75 = 14
            (4, 2): (1.75, 2.0, 2.0, 2.0),  # 1.75 * 2 * 2 * 2 = 14
            (4, 3): (2.0, 1.75, 2.0, 2.0),  # 2 * 1.75 * 2 * 2 = 14
            (4, 4): (2.0, 2.0, 1.75, 2.0),  # 2 * 2 * 1.75 * 2 = 14
            
            (5, 1): (2.0, 1.75, 1.0, 2.0, 2.0),  # 2 * 1.75 * 1 * 2 * 2 = 14
            (5, 2): (1.4, 2.0, 1.25, 2.0, 2.0),  # 1.4 * 2 * 1.25 * 2 * 2 = 14
            (5, 3): (1.75, 2.0, 1.0, 2.0, 2.0),  # 1.75 * 2 * 1 * 2 * 2 = 14
            (5, 4): (1.4, 2.5, 1.0, 2.0, 2.0),  # 1.4 * 2.5 * 1 * 2 * 2 = 14
            
            (6, 1): (1.4, 1.4, 1.4, 1.4, 1.4, 1.84),  # ≈ 14
            (6, 2): (1.5, 1.5, 1.5, 1.5, 1.38, 2.0),  # ≈ 14
            (6, 3): (1.5, 1.5, 1.5, 1.56, 1.5, 1.5),  # ≈ 14
            (6, 4): (1.4, 1.5, 1.5, 1.5, 1.74, 1.5),  # ≈ 14
        }
        
        # Check if we have a predefined strategy
        key = (n_blocks, min(n_features, 4))  # Cap n_features at 4 for lookup
        if key in factor_strategies:
            factors = list(factor_strategies[key])
            # Ensure product is exactly 14
            product = math.prod(factors)
            if abs(product - 14.0) > 0.01:
                # Adjust last factor to ensure exact 14x
                factors[-1] = factors[-1] * (14.0 / product)
            return tuple(factors)
        
        # Fallback: Smart distribution for arbitrary n_blocks
        factors = self._distribute_factors_smartly(n_blocks, total_factor)
        return factors
    
    def _distribute_factors_smartly(self, n_blocks: int, total: float) -> Tuple[float, ...]:
        """
        Distribute total factor across n_blocks intelligently.
        
        Strategy: Use powers of 2 where possible, put remainder in one block.
        """
        if n_blocks <= 0:
            return (total,)
            
        # For 14, we can use at most 3 factors of 2 (2^3 = 8 < 14)
        num_twos = min(n_blocks - 1, 3)
        
        factors = []
        remaining = total
        
        # Add 2x factors
        for _ in range(num_twos):
            factors.append(2.0)
            remaining /= 2.0
        
        # Distribute remainder across remaining blocks
        remaining_blocks = n_blocks - num_twos
        if remaining_blocks == 1:
            factors.append(remaining)
        else:
            # Distribute remainder evenly
            base_factor = remaining ** (1.0 / remaining_blocks)
            for i in range(remaining_blocks - 1):
                factors.append(base_factor)
            # Last factor takes any rounding error
            current_product = math.prod(factors)
            factors.append(total / current_product)
        
        return tuple(factors)
    
    def _get_channel_schedule(self, initial_channels: int) -> List[int]:
        """
        Create a channel reduction schedule based on strategy.
        SMART: Reduce channels aggressively as resolution increases to manage memory.
        """
        schedule = []
        
        # Calculate cumulative upsampling at each stage
        cumulative_factors = []
        cumulative = 1.0
        for factor in self.up_factors:
            cumulative *= factor
            cumulative_factors.append(cumulative)
        
        for i in range(self.n_blocks):
            # Inverse relationship: more upsampling = fewer channels
            upsampling_ratio = cumulative_factors[i]
            
            if self.channel_reduction == "maintain":
                next_channels = initial_channels
            elif self.channel_reduction == "aggressive":
                # Very aggressive reduction: halve each time
                next_channels = max(64, initial_channels // (2 ** (i + 1)))
            else:  # gradual (default) - smart memory-aware reduction
                # Reduce channels based on resolution increase
                # At 2x: keep channels, at 4x: halve, at 8x+: quarter
                if upsampling_ratio >= 8.0:
                    next_channels = max(64, initial_channels // 4)
                elif upsampling_ratio >= 4.0:
                    next_channels = max(96, initial_channels // 2)
                elif upsampling_ratio >= 2.0:
                    next_channels = max(128, int(initial_channels / 1.5))
                else:
                    next_channels = initial_channels
            
            # Round to nearest 32 for efficiency
            next_channels = int(round(next_channels / 32) * 32)
            next_channels = max(64, next_channels)
            
            schedule.append(next_channels)
        
        return schedule
    
    def _build_layers(self, device: torch.device):
        """Build all CNN layers with proper stages and projections."""
        # Scale channels based on input features - SMARTER scaling
        if self.memory_efficient:
            # Memory-efficient mode: aggressive channel reduction
            if self.n_features >= 4:
                divisor = 8  # Very aggressive for many features
                max_initial = 128
            elif self.n_features >= 2:
                divisor = 4
                max_initial = 192
            else:
                divisor = 3
                max_initial = 256
            initial_channels = max(64, min(max_initial, self.embedding_size // divisor))
        else:
            # Original mode (may cause OOM with large n_features)
            divisor = 2
            initial_channels = max(128, self.embedding_size // divisor)
                
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
        
        # Calculate exact target sizes based on cumulative product
        target_sizes = []
        cumulative_factor = 1.0
        
        for factor in self.up_factors:
            cumulative_factor *= factor
            # Calculate target size from original patch_sz
            target_size = int(round(patch_sz * cumulative_factor))
            # Clamp to image resolution
            target_size = min(target_size, img_res)
            target_sizes.append(target_size)
        
        # CRITICAL: Ensure final size exactly matches image resolution
        target_sizes[-1] = img_res
                
        # Progressive upsampling with residual connections
        for i, target_size in enumerate(target_sizes):
            # Upsample
            if x.shape[-1] != target_size:
                x = F.interpolate(x, size=(target_size, target_size), 
                                mode="bilinear", align_corners=False)
            
            # Store for residual
            identity = self.projections[i](x)
            
            # Apply convolution block with optional gradient checkpointing
        
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

        # This should not be needed if target_sizes[-1] == img_res
        if logits.shape[-2:] != (img_res, img_res):
            logits = F.interpolate(logits, size=(img_res, img_res), 
                                 mode="bilinear", align_corners=False)
        
        return logits