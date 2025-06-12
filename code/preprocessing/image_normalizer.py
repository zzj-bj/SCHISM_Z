# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 22:12:18 2025

@author: Pierre.FANCELLI
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from tools import folder as fo

class ImageNormalizer:
    """
    This class allows normalizing a group of images.
    """
    def __init__(self, input_path, output_path, report):
        self.input_path = input_path
        self.output_path = output_path
        self.report = report

    def normalize_images(self):
        """Normalizes all images in the input directory."""
        # Searching for image files (*.tif)
        files = [f for f in os.listdir(self.input_path) if f.endswith(".tif")]

        for filename in tqdm(
               files,
                unit="file",
                bar_format="     Normalization: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                ncols=80):
            file = os.path.join(self.input_path, filename)

            try:
                # Load the image and convert it to grayscale
                image = Image.open(file).convert('L')
                masks = np.array(image)

                # Retrieve the unique classes in the image
                unique_classes = np.unique(masks)
                num_classes = len(unique_classes)

                # Create a mapping of classes to scaled values
                scaling = np.linspace(0, 255, num_classes)

                # Create a mapping dictionary
                class_mapping = {value: int(scaling[i]) for i,
                                 value in enumerate(unique_classes)}

                # Convert masks to a PyTorch tensor
                masks_tensor = torch.tensor(masks,
                                    device='cuda' if torch.cuda.is_available() else 'cpu')

                # Apply the mapping using vectorization
                compliant_masks = torch.zeros_like(masks_tensor, dtype=torch.uint8)
                for value, scaled_value in class_mapping.items():
                    compliant_masks[masks_tensor == value] = scaled_value

                # Move the compliant masks back to CPU and convert to NumPy array
                compliant_masks_cpu = compliant_masks.cpu().numpy()

                # Create a new image from the compliant masks
                normalized_image = Image.fromarray(compliant_masks_cpu.astype(np.uint8))

                # Save the normalized image
                name, ext = os.path.splitext(os.path.basename(file))
                output_name = f"{name}{ext}"
                output_file_path = os.path.join(self.output_path, output_name)
                normalized_image.save(output_file_path)

            except Exception as e:
                print(f"\n{e}")
