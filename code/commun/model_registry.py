"""
This module provides a mapping of model names to their respective classes.
It allows for easy retrieval of model classes based on their names,
facilitating the instantiation of different models in a machine learning pipeline.
"""
from classes.unet_vanilla import UnetVanilla
from classes.unet_segmentor import UnetSegmentor
from classes.dinov2 import DinoV2Segmentor

model_mapping = {
    'UnetVanilla': UnetVanilla,
    'UnetSegmentor': UnetSegmentor,
    'DINOv2': DinoV2Segmentor
}
