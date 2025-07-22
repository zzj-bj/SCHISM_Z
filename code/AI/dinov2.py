# pylint: disable=too-many-instance-attributes
"""
DinoV2 Segmentor Class

This class implements a segmentation model based on the DinoV2 architecture.
It allows for both linear and CNN heads, supports PEFT (Parameter-Efficient Fine-Tuning),
and can be quantized for efficient inference.

@author: Florent.BRONDOLO
"""
from dataclasses import dataclass
import torch
from torch import nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from AI.linear_head import LinearHead
from AI.linear_head import LinearHeadConfig
from AI.cnn_head import CNNHead
from AI.cnn_head import CNNHeadConfig


@dataclass
class DinoV2SegmentorConfig:
    """
    DinoV2SegmentorConfig Class for Configuring DinoV2Segmentor

    This class defines the configuration parameters for the DinoV2Segmentor.
    
    It includes parameters such as channels, number of classes, linear head,
    kernel size, activation function, size of the model, number of features,
    PEFT configuration, quantization settings, and inference mode.
    """
    channels: int = 3
    num_classes: int = 3
    linear_head: bool = True
    k_size: int = 3
    activation: str = 'relu'
    size: str = 'base'
    n_features: int = 1
    peft: bool = True
    quantize: bool = True
    r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    inference_mode: bool = False

class DinoV2Segmentor(nn.Module):
    """
    DinoV2Segmentor: A segmentation model based on the DinoV2 architecture.
    """
    REQUIRED_PARAMS = {
		'num_classes': int
	}

    OPTIONAL_PARAMS = {
		'channels': int,
		'linear_head': bool,
		'k_size': int,
		'activation': str,
		'size': str,
		'n_features': int,
		'peft': bool,
		'quantize': bool,
		'r': int,
		'lora_alpha': int,
		'lora_dropout': float,
		'inference_mode': bool
	}

    emb_size = {
		"small": 384,
		"base": 768,
		"large": 1024,
	}

    def __init__(self,
				dinov2_segmentor_config: DinoV2SegmentorConfig
				):
        super().__init__()
        assert dinov2_segmentor_config.size in self.emb_size, "Invalid size embedding size"
        self.config = {
            "inference_mode": dinov2_segmentor_config.inference_mode,
            "linear_head": dinov2_segmentor_config.linear_head,
            "channels": dinov2_segmentor_config.channels,
            "num_classes": dinov2_segmentor_config.num_classes,
            "k_size": dinov2_segmentor_config.k_size,
            "activation": dinov2_segmentor_config.activation.lower(),
            "size": dinov2_segmentor_config.size,
            "embedding_size": self.emb_size[dinov2_segmentor_config.size],
            "n_features": dinov2_segmentor_config.n_features,
            "peft": dinov2_segmentor_config.peft,
            "quantize": dinov2_segmentor_config.quantize,
            "r": dinov2_segmentor_config.r,
            "lora_alpha": dinov2_segmentor_config.lora_alpha,
            "lora_dropout": dinov2_segmentor_config.lora_dropout,
        }


        if self.config["quantize"] :
            self.quantization_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
				bnb_4bit_compute_dtype=torch.bfloat16,
			)
            self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{self.config["size"]}',
                                                      quantization_config=self.quantization_config)
            self.backbone = prepare_model_for_kbit_training(self.backbone)
        else:
            self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{self.config["size"]}')

        if self.config["peft"]:
            peft_config = LoraConfig(inference_mode=self.config["inference_mode"],
                                     r=self.config["r"],
                                     lora_alpha=self.config["lora_alpha"],
                                     lora_dropout=self.config["lora_dropout"],
                                     target_modules="all-linear",use_rslora=True)

            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        if self.config["linear_head"]:
            self.seg_head = LinearHead(LinearHeadConfig(
                embedding_size=self.config["embedding_size"],
                num_classes=self.config["num_classes"],
                n_features=self.config["n_features"]))
        else:
            self.seg_head = CNNHead(CNNHeadConfig(
                embedding_size=self.config["embedding_size"],
                channels=self.config["channels"],
                num_classes=self.config["num_classes"],
                k_size=self.config["k_size"],
                n_features=self.config["n_features"],
                activation=self.config["activation"]))

        print(f"Number of parameters:{sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )}")

    def forward(self, x):
        """
        Forward pass of the DinoV2Segmentor model.
        
        This method processes the input tensor through the backbone and segmentation head,
		returning the segmentation map.
        
        Args:
			x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
			torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
		"""
        # frozen weights of dino
        with torch.set_grad_enabled(self.config["peft"] and self.training):
            if self.config["n_features"] == 1:
                features = self.backbone(pixel_values=x).last_hidden_state
            else:
                features = tuple(
                    list(self.backbone(pixel_values=x,
                         output_hidden_states=True)
                         ['hidden_states'])[-self.config["n_features"]:]
                )
        inputs = {"features" : features, "image" : x}
        return self.seg_head(inputs)
