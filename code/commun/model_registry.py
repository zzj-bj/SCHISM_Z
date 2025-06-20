from classes.unet_vanilla import UnetVanilla
from classes.unet_segmentor import UnetSegmentor
from classes.dinov2 import DinoV2Segmentor

model_mapping = {
    'UnetVanilla': UnetVanilla,
    'UnetSegmentor': UnetSegmentor,
    'DINOv2': DinoV2Segmentor
}
