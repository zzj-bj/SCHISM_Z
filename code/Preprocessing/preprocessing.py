# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:05:11 2025

@author: Pierre.FANCELLI
"""

from Menu import selection as sl
from Preprocessing import normalisation as no
from Preprocessing import json_generation as jg


def launch_json_generation():
    """
    Json Generatio
    """
    print("\n[ Json Generation Mode ]")
    print("[!] Starting Json generation")

    json_generation = jg.JsonGeneration()
    try:
        json_generation.calcul()
        print("[✓]] Json Generation terminée\n")
    except ValueError as e:
        print(e)

def launch_normalisation():
    """
    Normalization of masks in 8-bit grayscale format
    """
    print("\n[ Normalisation Mode ]")
    print("[!] Starting normalisation")

    path = sl.get_path("Enter the data directory ")

    normalizer = no.ImageNormalizer(path)
    try:
        normalizer.normalize_images()
        print("[✓]] Normalization terminée\n")
    except ValueError as e:
        print(e)
