from transformers import AutoImageProcessor, AutoModel, get_scheduler, BitsAndBytesConfig, AutoFeatureExtractor, ResNetForImageClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np

class LinearHead(nn.Module):
	def __init__(self, embedding_size=768, num_classes=3, n_features=1):
		super(LinearHead, self).__init__()
		self.n_features=n_features
		self.embedding_size = embedding_size * n_features
		self.head = nn.Sequential(
			nn.BatchNorm2d(self.embedding_size),
			nn.Conv2d(self.embedding_size, num_classes, kernel_size=1, padding=0, bias=True),
		)

	def forward(self, inputs):
		features = inputs["features"]
		img_shape = inputs["image"].shape[-1]
		patch_feature_size = inputs["image"].shape[-1] // 14
		if self.n_features > 1:
			features = torch.cat(features, dim=-1)
		features = features[:,1:].permute(0,2,1).reshape(-1, self.embedding_size, patch_feature_size,patch_feature_size)
		logits = self.head(features)
		logits = F.interpolate(input=logits, size=(int(img_shape),int(img_shape)), mode="bilinear", align_corners=False)
		return logits


class CNNHead(nn.Module):
	def __init__(
			self,
			embedding_size,
			n_block=4, 
			channels=512, 
			num_classes=2, 
			k_size=3, 
			n_features=1
		):
		super(CNNHead, self).__init__()
		self.n_features=n_features
		self.embedding_size = embedding_size * n_features
		self.n_block = n_block
		self.channels = channels
		self.k_size = int(k_size)
		self.num_classes = num_classes
		self.input_conv = nn.Conv2d(
			in_channels=self.embedding_size,
			out_channels=channels,
			kernel_size=self.k_size,
			padding=1
		)
		self.decoder_convs = nn.ModuleList()

		self.upscale_fn = ["interpolate", "interpolate", "pixel_shuffle", "pixel_shuffle"]
		for i in range(n_block):
			if self.upscale_fn[i] == "interpolate":
				self.decoder_convs.append(self._create_decoder_conv_block(channels=channels, kernel_size=self.k_size, downscale_factor=i))
			else:
				channels = channels//4
				self.decoder_convs.append(self._create_decoder_up_conv_block(channels=channels, kernel_size=self.k_size, downscale_factor=1))

		self.seg_conv = nn.Sequential(
			nn.Conv2d(channels, num_classes, kernel_size=self.k_size, padding=1)
		)

	def _create_decoder_conv_block(self, channels, kernel_size, downscale_factor):
			return nn.Sequential(
				nn.BatchNorm2d(channels),
				nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
			)

	def _create_decoder_up_conv_block(self, channels, kernel_size, downscale_factor):
			return nn.Sequential(
				nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
			)

	def forward(self, inputs):
		features = inputs["features"]
		patch_feature_size = inputs["image"].shape[-1] // 14
		if self.n_features > 1:
			features = torch.cat(features, dim=-1)
		features = features[:,1:].permute(0,2,1).reshape(-1, self.embedding_size, patch_feature_size,patch_feature_size)
		x = self.input_conv(features)
		for i in range(self.n_block):
			if self.upscale_fn[i] == "interpolate":
				resize_shape = x.shape[-1]*2 if i >=1 else x.shape[-1]*1.75
				x = F.interpolate(input=x, size=(int(resize_shape), int(resize_shape)), mode="bicubic")
			else:
				x = F.pixel_shuffle(x, 2)
			x = x + self.decoder_convs[i](x)
			if i%2==1 and i!=0:
				x = F.dropout(x, p=0.2)
				x = F.leaky_relu(x)
		return self.seg_conv(x)

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

	def forward(self, x, is_training=False):
		# frozen weights of dino
		with torch.set_grad_enabled(self.peft and is_training):
			if self.n_features == 1:
				features = self.backbone(pixel_values=x).last_hidden_state
			else:
				features = list(self.backbone(pixel_values=x, output_hidden_states=True)['hidden_states'])[-self.n_features:]
		inputs = {"features" : features, "image" : x}
		return self.seg_head(inputs)

if __name__ == "__main__":
	import torch
	from transformers import AutoFeatureExtractor

	# Test configuration
	n_blocks = 4
	channels = 512
	num_classes = 2
	size = "base"
	batch_size = 1
	image_size = 224  # Assuming input image size is 224x224

	# Instantiate the model
	model = DinoV2Segmentor(
		n_block=n_blocks,
		channels=channels,
		num_classes=num_classes,
		size=size,
		n_features=3,
		linear_head=False,
	)

	# Set the model to evaluation mode
	model.eval()

	# Generate random input data
	input_data = torch.rand(batch_size, 3, image_size, image_size)  # Random RGB image

	# Perform inference
	with torch.no_grad():
		output = model(input_data)

	# Print the output shape
	print("Output shape:", output.shape)

