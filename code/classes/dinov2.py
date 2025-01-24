import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from classes.LinearHead import LinearHead
from classes.CNNHead import CNNHead

class DinoV2Segmentor(nn.Module):

	REQUIRED_PARAMS = {
		'num_classes': int
	}

	OPTIONAL_PARAMS = {
		'k_size': int,  # Kernel size for convolutions
		'activation': str,  # Activation function type
		'n_block': int,
		'channels': int,
		'size' : str,
		'linear_head' : bool,
		'n_features' : int,
		'peft' : bool,
		'quantize' : bool,
		'r' : int,
		'lora_alpha' : int,
		'lora_dropout' : float
	}

	emb_size = {
		"small" : 384,
		"base" : 768,
		"large" : 1024,
	}

	def __init__(
			self, 
			n_block=4, 
			channels=512, 
			num_classes=2, 
			linear_head=True,
			k_size=3, 
			activation='relu', 
			size="base", 
			n_features=1, 
			peft=True, 
			quantize=True, 
		):
		super(DinoV2Segmentor, self).__init__()
		assert size in self.emb_size.keys(), "Invalid size"
		self.num_classes = num_classes
		self.n_features = n_features
		self.peft = peft
		self.embedding_size = self.emb_size[size]
		if quantize :
			self.quantization_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
				bnb_4bit_compute_dtype=torch.bfloat16,
			)
			self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{size}', quantization_config=self.quantization_config)
			self.backbone = prepare_model_for_kbit_training(self.backbone)
		else:
			self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{size}')

		if peft:
			peft_config = LoraConfig(inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear", use_rslora=True)
			self.backbone = get_peft_model(self.backbone, peft_config)
			self.backbone.print_trainable_parameters()

		if linear_head:
			self.seg_head = LinearHead(self.embedding_size, num_classes, n_features)
		else:
			self.seg_head = CNNHead(self.embedding_size, n_block, channels, num_classes, k_size, n_features)
		print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

	def forward(self, x):
		# frozen weights of dino
		with torch.set_grad_enabled(self.peft and self.training):
			if self.n_features == 1:
				features = self.backbone(pixel_values=x).last_hidden_state
			else:
				features = list(self.backbone(pixel_values=x, output_hidden_states=True)['hidden_states'])[-self.n_features:]
		inputs = {"features" : features, "image" : x}
		return self.seg_head(inputs)

