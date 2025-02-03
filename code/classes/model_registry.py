from classes.UnetVanilla import UnetVanilla
from classes.UnetSegmentor import UnetSegmentor
from classes.dinov2 import DinoV2Segmentor

model_mapping = {
    'UnetVanilla': UnetVanilla,
    'UnetSegmentor': UnetSegmentor,
    'DINOv2': DinoV2Segmentor
}