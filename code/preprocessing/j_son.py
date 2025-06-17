# -*- coding: utf-8 -*-
"""


@author: Pierre.FANCELLI
"""


import os
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, parent_dir, subfolders, report, percentage_to_process=0.9):
        self.parent_dir = parent_dir
        self.subfolders = subfolders
        self.percentage_to_process = percentage_to_process
        self.report = report
        self.results = {}

    def process_datasets(self):
        for folder_name in tqdm(sorted(self.subfolders),
                           bar_format="   Process : {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}%",
                           ncols=80):
            dataset_path = os.path.join(self.parent_dir, folder_name, 'images')
            if not os.path.isdir(dataset_path):
                self.report.add(f" - Warning: {dataset_path} is not a directory. Skipping", '')
                print(f"Warning: {dataset_path} is not a directory. Skipping.")
                continue

            try:
                std_dev, mean = self.calculate_mean_and_std_rgb(dataset_path)
                self.results[folder_name] = [std_dev, mean]
            except Exception as e:
                self.report.add(f" - Error processing {folder_name}: {e}", '')
                print(f"Error processing {folder_name}: {e}")

    def calculate_mean_and_std_rgb(self, folder_path):
        image_files = self._get_image_files(folder_path)
        num_items_to_process = int(len(image_files) * self.percentage_to_process)
        indices_to_process = np.random.choice(len(image_files), num_items_to_process, replace=False)

        pixel_sum = torch.zeros(3, dtype=torch.float32)
        pixel_count = 0

        for idx in indices_to_process:
            image_path = os.path.join(folder_path, image_files[idx])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Unable to load image at path: {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_sum += torch.sum(torch.tensor(image, dtype=torch.float32), dim=(0, 1))
            pixel_count += image.shape[0] * image.shape[1]

        mean = pixel_sum / pixel_count

        pixel_variance = torch.zeros(3, dtype=torch.float32)
        for idx in indices_to_process:
            image_path = os.path.join(folder_path, image_files[idx])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_variance += torch.sum((torch.tensor(image, dtype=torch.float32) - mean) ** 2,
                                        dim=(0, 1))

        std_dev = torch.sqrt(pixel_variance / pixel_count)

        # Normalize to [0, 1] assuming 8-bit images
        mean = (mean / 255).tolist()
        std_dev = (std_dev / 255).tolist()

        return std_dev, mean

    def _get_image_files(self, folder_path):
        return [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

    def save_results(self, output_file):
        with open(output_file, "w") as json_file:
            json.dump(self.results, json_file, indent=4)
