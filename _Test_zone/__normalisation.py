# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:47:48 2025

@author: Pierre.FANCELLI
"""

import os
from tqdm import tqdm
import numpy as np

from PIL import Image

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
            raise ValueError("Tous les masques doivent être compris entre 0 et 255.")

        # Obtenir les classes uniques dans l'image
        unique_classes = np.unique(masques)
        nombre_classes = len(unique_classes)

        if nombre_classes == 0:
            raise ValueError("Il doit y avoir au moins une classe.")

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
            raise ValueError("Aucune image normalisée à sauvegarder.")


class ImageNormalizer:
    """
    Cette classe permet de normaliser un groupe d'images.
      .
    Obligatoire : .
    input_path : Répertoire où se trouve les masques à traiter.

    dest : Sous répertoire où les images normalisées seront stokées.
    S'il n'est pas renseigné, on ajoutera "_Norm_" au nom du fichier.
    ext : Extention ajouté au nom du masque.
    image_formats : Formats des images à traiter par défaut 'TIF'.
    """
    def __init__(self, input_path, dest = None, ext = None, image_formats = None):
        self.input_path = input_path

        self.dest = dest if dest is not None else ''

        self.output_path = os.path.join(input_path, self.dest)
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(self.output_path, exist_ok=True)

        self.ext = ext if ext is not None else ""

        # Si pas de répertoire de sauvegarde ajout '_N_' au nom du fichier
        if not self.dest and self.ext == "" :
            self.ext = '_Norm_'

        self.image_formats = image_formats if image_formats is not None else ['.tif']

        self.traitement = Normalisation()

    def normalize_images(self):
        """Normalizes all images in the input directory."""
        files = [f for f in os.listdir(self.input_path)
                 if any(f.endswith(ext) for ext in self.image_formats)]

        if len(files) == 0:
            raise ValueError(" - !!! Pas de masques à traiter !!!")

        for filename in tqdm(files, unit="file",
                      bar_format=" - Normalisation: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                      ncols=80):
            file = os.path.join(self.input_path, filename)

            try:
                self.traitement.set_image_path(file)
                self.traitement.mise_en_conformite_image()

                # Créer le chemin de sortie pour l'image normalisée
                name, ext = os.path.splitext(os.path.basename(file))
                output_name = f"{name}{self.ext}{ext}"
                output_file_path = os.path.join(self.output_path, output_name)

                self.traitement.save_normalized_image(output_file_path)
            except Exception as e:
                print(f"\n - Erreur lors du traitement de l'image {filename}:\n  - {e}")

        print("\n[✓] Normalization terminée")
