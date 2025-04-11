# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:47:48 2025

@author: Pierre.FANCELLI
"""
import os
import numpy as np
from PIL import Image

from classes import normalisation as Nm


#----------------------------------------------------------------------

def get_path(prompt):
    """Requests a valid path from the user."""
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        print("[X] Invalid path. Try again.")

def get_folder(prompt):

    caracteres_invalides = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    while True :
        folder = input(f"[?] {prompt}: ")

        # Vérifie si le nom du dossier contient des caractères invalides
        if any(char in folder for char in caracteres_invalides):
            print("[X] Invalid path. Try again.")
        elif not folder.strip():
            folder = ''
            return folder
        else:
            return folder

#------------------------------------------------------------------------

# Définir les formats d'image à traiter
IMAGE_FORMATS = ['.tif', '.jpg', '.png', '.jpeg']

if __name__ == "__main__":
    path = get_path("Enter the data directory ")
    out_folder = get_folder("Entre the Folder Output ")
    ext_file = get_folder("Extented name ")

    normalizer = Nm.ImageNormalizer(path, out_folder, ext_file)
    try:
        normalizer.normalize_images()
    except ValueError as e:
        print(e)

