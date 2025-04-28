# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:47:48 2025

@author: Pierre.FANCELLI
"""

import os
import numpy as np
from tqdm import tqdm

from PIL import Image

import torch

from tools import folder as fo

#============================================================================

class Normalisation:
    """
    A class to normalize image masks to a standard range.
    """
    def __init__(self):
        self.raw_image = None
        self.normalized_image = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        # Convert masks to a PyTorch tensor
        masks_tensor = torch.tensor(masks, device=self.device)

        # Apply the mapping using vectorization
        compliant_masks = torch.zeros_like(masks_tensor, dtype=torch.uint8)
        for value, scaled_value in class_mapping.items():
            compliant_masks[masks_tensor == value] = scaled_value

        # Move the compliant masks back to CPU and convert to NumPy array
        compliant_masks_cpu = compliant_masks.cpu().numpy()

        # Create a new image from the compliant masks
        self.normalized_image = Image.fromarray(compliant_masks_cpu.astype(np.uint8))

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
            name = fo.get_name_at_index(self.input_path, -2)
            self.report.add(' - No files to process in : ',name)
        else:
            for filename in tqdm(
                    files,
                    unit="file",
                    bar_format=" - Normalization: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
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
