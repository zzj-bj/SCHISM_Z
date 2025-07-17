"""
This module provides a mapping of model names to their respective classes.
It allows for easy retrieval of model classes based on their names,
facilitating the instantiation of different models in a machine learning pipeline.
"""
from models.unet_vanilla import UnetVanilla
from models.unet_vanilla import UnetVanillaConfig
from models.unet_segmentor import UnetSegmentor
from models.unet_segmentor import UnetSegmentorConfig
from models.dinov2 import DinoV2Segmentor
from models.dinov2 import DinoV2SegmentorConfig

model_mapping = {
    'UnetVanilla': UnetVanilla,
    'UnetSegmentor': UnetSegmentor,
    'DINOv2': DinoV2Segmentor
}

model_config_mapping = {
    'UnetVanilla': UnetVanillaConfig,
    'UnetSegmentor': UnetSegmentorConfig,
    'DINOv2': DinoV2SegmentorConfig
}
