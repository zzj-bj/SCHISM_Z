# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:47:48 2025

@author: Pierre.FANCELLI
"""

import os
from tqdm import tqdm
import numpy as np

from PIL import Image

#============================================================================
class Normalisation:
    """
    A class to normalize image masks to a standard range.
    """
    def __init__(self):
        self.image_brute = None
        self.image_morme = None

    def set_image_path(self, image_path):
        """Set the path of the input image."""
        self.image_brute = image_path

    def mise_en_conformite_image(self):
        """Normalize the image to a standard range."""
        # Charger l'image et conversion en niveau de gris
        image = Image.open(self.image_brute).convert('L')
        masques = np.array(image)

        # Vérifier que les valeurs des masques sont dans la plage 0-255
        if not np.all((masques >= 0) & (masques <= 255)):
            raise ValueError("All masks must be between 0 and 255.")

        # Obtenir les classes uniques dans l'image
        unique_classes = np.unique(masques)
        nombre_classes = len(unique_classes)

        if nombre_classes == 0:
            raise ValueError("There must be at least one class.")

        # Créer un mapping des classes à des valeurs mises à l'échelle
        mise_a_echelle = np.linspace(0, 255, nombre_classes)

        # Créer un dictionnaire de mapping
        mapping_classes = {valeur: int(mise_a_echelle[i])
                           for i, valeur in enumerate(unique_classes)}

        # Appliquer le mapping
        masques_conformes = np.vectorize(mapping_classes.get)(masques)

        # Créer une nouvelle image à partir des masques conformes
        self.image_morme = Image.fromarray(masques_conformes.astype(np.uint8))

        return self.image_morme

    def save_normalized_image(self, output_path):
        """Save the normalized image to the specified path."""
        if self.image_morme is not None:
            self.image_morme.save(output_path)
        else:
            raise ValueError("No normalized image to save.")


class ImageNormalizer:
    """
This class allows normalizing a group of images.

input_path  : Directory where the masks to be processed are located.
output_path : Directory where the generated images will be stored.

    """

    def __init__(self, input_path, output_path, report):

        self.input_path = input_path
        self.output_path = output_path
        self.report = report

        self.traitement = Normalisation()

    def normalize_images(self):
        """Normalizes all images in the input directory."""
        # Recherche des fichiers images (*.tif)
        files = [f for f in os.listdir(self.input_path)
                 if f.endswith(".tif")]

        if len(files) == 0:
            aaa = self.input_path
            fff = aaa.split("\\")

            self.report.ajouter_element(' - No files to process in : ',fff[-2])
            # raise ValueError(" - !!! No files to process !!!")
        else:

            for filename in tqdm(files, unit="file",
                          bar_format=" - Normalization: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                          ncols=80):
                file = os.path.join(self.input_path, filename)

                try:
                    self.traitement.set_image_path(file)
                    self.traitement.mise_en_conformite_image()

                    name, ext = os.path.splitext(os.path.basename(file))
                    output_name = f"{name}{ext}"
                    output_file_path = os.path.join(self.output_path, output_name)

                    self.traitement.save_normalized_image(output_file_path)
                except Exception as e:
                    print(f"\n{e}")
