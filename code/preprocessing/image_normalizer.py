# -*- coding: utf-8 -*-
"""
 This script normalizes images in a specified directory by scaling the unique pixel values
 to a range from 0 to 255. It processes each image, applies the normalization
 and saves the normalized images to an output directory.

 @author : Pierre.FANCELLI
"""

# Standard library
import os
from typing import Any, Dict, List, Tuple

# Third-party
import numpy as np
from tqdm import tqdm
from PIL import Image

# Local application imports
from tools import display_color as dc
from tools import utils as ut
import tools.constants as ct
from tools.constants import DISPLAY_COLORS as colors
from tools.constants import IMAGE_EXTENSIONS

#=============================================================================
# pylint: disable=too-few-public-methods
class ImageNormalizer:
    """
    This class allows normalizing a group of images.
    """
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.display = dc.DisplayColor()


    def normalize_images(self) -> None:
        """Normalizes all images in the input directory."""
        files = [f for f in os.listdir(self.input_path)
                 if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
        errors: List[Tuple[str, str]] = []

        folder = self.input_path.split(os.sep)[-2]
 
        pbar = tqdm(
            files,
            ncols=ct.TQDM_NCOLS,
            bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%"
        )
        for filename in pbar:
            pbar.set_description(f"{folder} : {filename}")

            file = os.path.join(self.input_path, filename)

            try:
                # Load the image and convert it to grayscale
                masks = np.array(Image.open(file).convert('L'))

                # Retrieve the unique classes in the image
                unique_classes = np.unique(masks)

                # Create a mapping dictionary
                class_mapping = {value: int(np.linspace(0, 255, len(unique_classes))[i]) for i,
                                 value in enumerate(unique_classes)}

                # Apply the mapping using vectorization
                lut = np.zeros(masks.max()+1, dtype=np.uint8)
                for v, scaled in class_mapping.items():
                    lut[v] = scaled
                compliant_masks = lut[masks]

                # Create a new image from the compliant masks
                normalized_image = Image.fromarray(compliant_masks)

                # Save the normalized image
                name, ext = os.path.splitext(os.path.basename(file))
                res = ut.split_string(name)

                # Insert _normalized_
                if res:
                    new_name = f"{res[0]}_normalized_{res[1]}"
                else:
                    new_name = name
                normalized_image.save(os.path.join(self.output_path,  f"{new_name}{ext}"))

            except (IOError, ValueError) as e:
                # collect the filename and the short reason
                errors.append((filename, str(e)))

        if errors:
            self.display.print(
                f"{len(errors)} files failed to normalize. See details below:", 
                colors["error"]
            )
            for fname, reason in errors:
                self.display.print(f"- {fname}: {reason}", colors["error"])
        else:
            self.display.print("All files normalized successfully.", colors["ok"])
            