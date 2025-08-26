# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
This script processes datasets to calculate the mean
and standard deviation of RGB values for images.

It iterates through specified subfolders, checks for the existence of an 'images' directory,
calculates the mean and standard deviation of the RGB values for the images in that directory,
and saves the results in a JSON file.

@author: Pierre.FANCELLI
"""
# Standard library
from dataclasses import dataclass
import os
import json
from typing import Any, Dict, List, Tuple

# Third-party
import numpy as np
import cv2
from tqdm import tqdm

# Local application imports
import tools.constants as ct
from tools.constants import DISPLAY_COLORS as colors
import tools.display_color as dc
from tools.constants import IMAGE_EXTENSIONS

#---------------------------------------------------------------------------

@dataclass
class JsonConfig:

    parent_dir: str
    subfolders: List[str]
    json_file: str
    percentage_to_process: float = 1.0

class Json:
    """
    This module generates a 'json' file from the provided data.
    You can specify the percentage of data to be included.
    """
    def __init__(self, json_config: JsonConfig) -> None:
        self.parent_dir = json_config.parent_dir
        self.subfolders = json_config.subfolders
        self.json_file = json_config.json_file
        self.percentage_to_process = json_config.percentage_to_process
        self.results = {}
        self.display = dc.DisplayColor()


    def process_datasets(self, add_default: bool = False, append: bool = False) -> None:
        """
        Process each dataset in the specified subfolders.

        This method iterates through each subfolder, checks if the 'images' directory exists.
        Calculates the mean and standard deviation of the RGB values 
        for the images in that directory.
        Results are stored in the `self.results` dictionary,
        and any warnings or errors are reported.

        Args:
            add_default (bool): If True, adds default values for 'mean' and 'std_dev'
              if not present.
            append (bool): If True, appends results to an existing JSON file instead
              of overwriting it.

        Raises:
            Exception: If an error occurs during the processing of a dataset.
        """
        for folder_name in self.subfolders  :

            dataset_path = os.path.join(self.parent_dir, folder_name, 'images')
            rep_name = folder_name.split("\\")[-1]
 
            try:
                std_dev, mean = self.calculate_mean_and_std_rgb(dataset_path, rep_name)
                self.results[folder_name] = [std_dev, mean]
            except (IOError, ValueError) as e:
                self.display.print(f"Error processing {folder_name}:\n {e}", colors["error"])

        if add_default:
            # Add default values for 'mean' and 'std_dev' if not present
            if "default" not in self.results:
                self.results["default"] = ct.DEFAULT_MEAN_STD

        if append:
            # Load existing results if appending
            if os.path.exists(self.json_file):
                with open(self.json_file, "r", encoding="utf-8") as json_file:
                    existing_results = json.load(json_file)
                self.results.update(existing_results)

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.results, json_file, indent=4)


    def calculate_mean_and_std_rgb(self, folder_path: str, rep_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the mean and standard deviation of RGB values for images in a specified folder.

        This method randomly selects a percentage of images from the given folder,
        computes the mean & standard deviation of their pixel values,
        and normalizes the results to the range [0, 1].

        Args:
            folder_path (str): The path to the folder containing the images.

        Returns:
            tuple: A tuple containing the standard deviation & mean of the values, both as lists.

        Raises:
            Exception: If an error occurs while processing the images.
        """
        image_files = [f for f in sorted(os.listdir(folder_path))
                if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]

        num_items_to_process = int(len(image_files) * self.percentage_to_process)

        num_items_to_process = min(num_items_to_process, len(image_files))
        #np.random.seed(57)
        indices_to_process = np.random.choice(len(image_files), num_items_to_process, replace=False)

        pixel_sum = np.zeros(3, dtype=np.float32)
        pixel_sum_squared = np.zeros(3, dtype=np.float32)
        pixel_count = 0

        for idx in tqdm(indices_to_process, ncols=70,
                        bar_format= rep_name + " : {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                        ):

            image_path = os.path.join(folder_path, image_files[idx])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                self.display.print("Unable to load image at path: {image_path}", colors["error"])
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            pixel_sum += np.sum(image, axis=(0, 1))
            pixel_sum_squared += np.sum(image ** 2, axis=(0, 1))
            pixel_count += image.shape[0] * image.shape[1]

        mean = pixel_sum / pixel_count
        mean_squared = pixel_sum_squared / pixel_count

        # Calculate variance
        pixel_variance = mean_squared - mean ** 2
        std_dev = np.sqrt(np.maximum(pixel_variance, 0))

        # Normalize to [0, 1] assuming 8-bit images
        mean = (mean / 255).tolist()
        std_dev = (std_dev / 255).tolist()

        return mean, std_dev
