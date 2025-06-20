"""
DinoV2 Segmentor Class

This class implements a segmentation model based on the DinoV2 architecture.
It allows for both linear and CNN heads, supports PEFT (Parameter-Efficient Fine-Tuning),
and can be quantized for efficient inference.

@author: Florent.BRONDOLO
"""
import torch
from torch import nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from classes.linear_head import LinearHead
from classes.cnn_head import CNNHead

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
				channels=512,
				num_classes=3,
				linear_head=True,
				k_size=3,
				activation='relu',
				size='base',
				n_features=1,
				peft=True,
				quantize=True,
				r=32,
				lora_alpha=32,
				lora_dropout=0.1,
				inference_mode=False
				):
        super().__init__()
        self.inference_mode=inference_mode
        self.channels = channels
        self.num_classes = num_classes
        self.linear_head = linear_head
        self.k_size = k_size
        self.activation = activation
        assert size in self.emb_size, "Invalid size embedding size"
        self.embedding_size = self.emb_size[str(size)]
        self.n_features = n_features
        self.peft = peft
        self.quantize = quantize
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if self.quantize :
            self.quantization_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
				bnb_4bit_compute_dtype=torch.bfloat16,
			)
            self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{size}',
                                                      quantization_config=self.quantization_config)
            self.backbone = prepare_model_for_kbit_training(self.backbone)
        else:
            self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{size}')

        if self.peft:
            peft_config = LoraConfig(inference_mode=self.inference_mode,r=self.r,
                                     lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout,
                                     target_modules="all-linear",use_rslora=True)
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        if self.linear_head:
            self.seg_head = LinearHead(self.embedding_size, self.num_classes, self.n_features)
        else:
            self.seg_head = CNNHead(self.embedding_size,self.channels,self.num_classes,
                                    self.k_size,self.n_features,self.activation)
        print(f"Number of parameters:{sum(p.numel() for p in self.parameters()
                                      if p.requires_grad)}")

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
        with torch.set_grad_enabled(self.peft and self.training):
            if self.n_features == 1:
                features = self.backbone(pixel_values=x).last_hidden_state
            else:
                features = list(self.backbone(pixel_values=x,
                                              output_hidden_states=True)
                                              ['hidden_states'])[-self.n_features:]
        inputs = {"features" : features, "image" : x}
        return self.seg_head(inputs)
