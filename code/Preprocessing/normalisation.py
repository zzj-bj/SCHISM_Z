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

def name_parent(input_path):
    fff = input_path.split("\\")
    return fff[-2]


class Normalisation:
    """
    A class to normalize image masks to a standard range.
    """
    def __init__(self):
        self.raw_image = None
        self.normalized_image = None

    def set_image_path(self, image_path):
        """Set the path of the input image."""
        self.raw_image = image_path

    def image_compliance(self):
        """Normalize the image to a standard range."""
        # Load the image and convert it to grayscale
        image = Image.open(self.raw_image).convert('L')
        masks = np.array(image)

        # Retrieve the unique classes in the image
        unique_classes = np.unique(masks)
        num_classes = len(unique_classes)

        # Create a mapping of classes to scaled values
        scaling = np.linspace(0, 255, num_classes)

        # Create a mapping dictionary
        class_mapping = {value: int(scaling[i]) for i, value in enumerate(unique_classes)}

        # Apply the mapping
        compliant_masks = np.vectorize(class_mapping.get)(masks)

        # Create a new image from the compliant masks
        self.normalized_image = Image.fromarray(compliant_masks.astype(np.uint8))

        return self.normalized_image

    def save_normalized_image(self, output_path):
        """Save the normalized image to the specified path."""
        if self.normalized_image is not None:
            self.normalized_image.save(output_path)
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

        self.processing = Normalisation()

    def normalize_images(self):
        """Normalizes all images in the input directory."""
        # Searching for image files (*.tif)
        files = [f for f in os.listdir(self.input_path)
                 if f.endswith(".tif")]

        if len(files) == 0:
            # Retrieving the name of the parent folder
            parent = name_parent(self.input_path)
            self.report.add(' - No files to process in : ',parent)
        else:
            for filename in tqdm(files, unit="file",
                          bar_format="  - Normalization: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                          ncols=80):
                file = os.path.join(self.input_path, filename)

                try:
                    self.processing.set_image_path(file)
                    self.processing.image_compliance()

                    name, ext = os.path.splitext(os.path.basename(file))
                    output_name = f"{name}{ext}"
                    output_file_path = os.path.join(self.output_path, output_name)

                    self.processing.save_normalized_image(output_file_path)
                except Exception as e:
                    print(f"\n{e}")
