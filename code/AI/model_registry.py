"""
This module provides a mapping of model names to their respective classes.
It allows for easy retrieval of model classes based on their names,
facilitating the instantiation of different models in a machine learning pipeline.
"""
from AI.unet_vanilla import UnetVanilla
from AI.unet_vanilla import UnetVanillaConfig
from AI.unet_segmentor import UnetSegmentor
from AI.unet_segmentor import UnetSegmentorConfig
from AI.dinov2 import DinoV2Segmentor
from AI.dinov2 import DinoV2SegmentorConfig

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
