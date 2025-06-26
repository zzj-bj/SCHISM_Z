    # -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
J_son Generation

@author: Pierre.FANCELLI
"""

import os
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm

from tools import folder as fo

#---------------------------------------------------------------------------

class DatasetProcessor:
    """
    This module generates a 'json' file from the provided data.
    You can specify the percentage of data to be included.
    """
    def __init__(self, parent_dir, subfolders, json_file, report, percentage_to_process):
        self.parent_dir = parent_dir
        self.subfolders = subfolders
        self.percentage_to_process = percentage_to_process
        self.report = report
        self.json_file = json_file
        self.results = {}

    def add_to_report(self, text, who):
        """
            Add a message to a report
        """
        if self.report is not None:
            self.report.add(text, who)


    def process_datasets(self):
        """
        Process each dataset in the specified subfolders.

        This method iterates through each subfolder, checks if the 'images' directory exists.
        Calculates the mean & standard deviation of the images in that directory.
        Results are stored in the `self.results` dictionary,
        and any warnings or errors are reported.

        Raises:
            Exception: If an error occurs during the processing of a dataset.
        """
        for folder_name in self.subfolders :

            dataset_path = os.path.join(self.parent_dir, folder_name, 'images')
            rep_name = fo.get_name_at_index(folder_name, -1)
            print(f" - {rep_name}")

            try:
                std_dev, mean = self.calculate_mean_and_std_rgb(dataset_path)
                self.results[folder_name] = [std_dev, mean]
            except (IOError, ValueError) as e:
                self.add_to_report(f" - Error processing {folder_name}:\n {e}", '')

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.results, json_file, indent=4)

    def calculate_mean_and_std_rgb(self, folder_path):
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
        image_files = self._get_image_files(folder_path)
        num_items_to_process = int(len(image_files) * self.percentage_to_process)

        num_items_to_process = min(num_items_to_process, len(image_files))
        # np.random.seed(42)
        indices_to_process = np.random.choice(len(image_files), num_items_to_process, replace=False)

        pixel_sum = torch.zeros(3, dtype=torch.float32)
        pixel_count = 0
        for idx in tqdm(indices_to_process, ncols=100,
                   bar_format="   Mean     : {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                   ):

            image_path = os.path.join(folder_path, image_files[idx])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                self.add_to_report(f" - Unable to load image at path: {image_path}", "")
                print(f" Unable to load image at path: {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_sum += torch.sum(torch.tensor(image, dtype=torch.float32), dim=(0, 1))
            pixel_count += image.shape[0] * image.shape[1]

        mean = pixel_sum / pixel_count

        pixel_variance = torch.zeros(3, dtype=torch.float32)
        for idx in tqdm(indices_to_process, ncols=100,
                   bar_format="   Variance : {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                   ):

            image_path = os.path.join(folder_path, image_files[idx])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) # pylint: disable=no-member
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # pylint: disable=no-member
            pixel_variance += torch.sum((torch.tensor(image, dtype=torch.float32) - mean) ** 2,
                                        dim=(0, 1))

        std_dev = torch.sqrt(pixel_variance / pixel_count)

        # Normalize to [0, 1] assuming 8-bit images
        mean = (mean / 255).tolist()
        std_dev = (std_dev / 255).tolist()

        return std_dev, mean

    def _get_image_files(self, folder_path):
        """
       Retrieve a sorted list of image files from a specified folder.

       This method filters the contents of the folder to include only image
       files with specific extensions.

       Args:
           folder_path (str): The path to the folder to search for image files.

       Returns:
           list: A sorted list of image file names.
       """
        return [f for f in sorted(os.listdir(folder_path))
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
