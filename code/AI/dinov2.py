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
from transformers import AutoModel
from transformers.utils.quantization_config import BitsAndBytesConfig 
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
    channels: int = 512
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

        self.inference_mode=dinov2_segmentor_config.inference_mode
        self.num_classes = dinov2_segmentor_config.num_classes
        self.linear_head = dinov2_segmentor_config.linear_head
        self.k_size = dinov2_segmentor_config.k_size
        self.activation = dinov2_segmentor_config.activation
        self.size = dinov2_segmentor_config.size
        assert self.size in self.emb_size.keys(), "Invalid size embedding size"
        self.embedding_size = self.emb_size[str(dinov2_segmentor_config.size)]
        self.n_features = dinov2_segmentor_config.n_features
        self.peft = dinov2_segmentor_config.peft
        self.quantize = dinov2_segmentor_config.quantize
        self.r = dinov2_segmentor_config.r
        self.lora_alpha = dinov2_segmentor_config.lora_alpha
        self.lora_dropout = dinov2_segmentor_config.lora_dropout

        if self.quantize :
            self.quantization_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
				bnb_4bit_compute_dtype=torch.bfloat16,
			)
            self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{self.size}',
                                                      quantization_config=self.quantization_config)
            self.backbone = prepare_model_for_kbit_training(self.backbone)
        else:
            self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{self.size}')

        if self.peft:
            peft_config = LoraConfig(inference_mode=self.inference_mode,
                                     r=self.r,
                                     lora_alpha=self.lora_alpha,
                                     lora_dropout=self.lora_dropout,
                                     target_modules="all-linear",use_rslora=True)

            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        if self.linear_head:
            self.seg_head = LinearHead(LinearHeadConfig(
                embedding_size=self.embedding_size,
                num_classes=self.num_classes,
                n_features=self.n_features))
        else:
            self.seg_head = CNNHead(CNNHeadConfig(
                embedding_size=self.embedding_size,
                num_classes=self.num_classes,
                k_size=self.k_size,
                n_features=self.n_features,
                activation=self.activation))

        print(
            f"Number of parameters:{sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, x):
        with torch.set_grad_enabled(self.peft and self.training):
            if self.n_features == 1:
                features = self.backbone(pixel_values=x).last_hidden_state
            else:
                features = list(self.backbone(pixel_values=x, output_hidden_states=True)['hidden_states'])[-self.n_features:]
        inputs = {"features" : features, "image" : x}
        return self.seg_head(inputs)
