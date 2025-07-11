# -*- coding: utf-8 -*-
"""
 This script normalizes images in a specified directory by scaling the unique pixel values
 to a range from 0 to 255. It processes each image, applies the normalization
 and saves the normalized images to an output directory.

 @author : Pierre.FANCELLI
"""
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

#=============================================================================
# pylint: disable=too-few-public-methods
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

        for filename in tqdm(files, ncols=100,
                bar_format="   Normalization: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                ):
            file = os.path.join(self.input_path, filename)

            try:
                # Load the image and convert it to grayscale
                masks = np.array(Image.open(file).convert('L'))

                # Retrieve the unique classes in the image
                unique_classes = np.unique(masks)

                # Create a mapping dictionary
                class_mapping = {value: int(np.linspace(0, 255, len(unique_classes))[i]) for i,
                                 value in enumerate(unique_classes)}

                # Convert masks to a PyTorch tensor
                masks_tensor = torch.tensor(masks,
                                    device='cuda' if torch.cuda.is_available() else 'cpu')

                # Apply the mapping using vectorization
                compliant_masks = torch.zeros_like(masks_tensor, dtype=torch.uint8)
                for value, scaled_value in class_mapping.items():
                    compliant_masks[masks_tensor == value] = scaled_value

                # Create a new image from the compliant masks
                normalized_image = Image.fromarray(compliant_masks.cpu().numpy().astype(np.uint8))

                # Save the normalized image
                name, ext = os.path.splitext(os.path.basename(file))
                normalized_image.save(os.path.join(self.output_path,  f"{name}{ext}"))

            except (IOError, ValueError) as e:
                print(f"\n{e}")
