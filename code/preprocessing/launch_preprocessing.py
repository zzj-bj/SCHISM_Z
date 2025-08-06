# -*- coding: utf-8 -*-
"""
Launches the preprocessing operations for datasets.
This script provides functionalities for preprocessing datasets,
including JSON generation and image normalization.
It prompts the user for the necessary directories and parameters,
validates the input, and then executes the preprocessing tasks.

This module allows for the following operations:
    - Json generation.
    - Normalization of masks in 8-bit grayscale format.

@author: Pierre.FANCELLI
"""

import os
from pathlib import Path
from tools import utils as ut
from tools import display_color as dc
from tools.constants import DISPLAY_COLORS as colors
from tools import menu
from preprocessing import image_normalizer
from preprocessing import json

#=============================================================================

class LaunchPreprocessing:

    def __init__(self):
        """
        Initializes the LaunchPreprocessing class.
            """
        self.display = dc.DisplayColor()

    def menu_preprocessing(self):
        """
        This module allows for the following operations:
            - Json generation
            - Normalization of masks in 8-bit grayscale format.
        """

        preprocessing_menu = menu.Menu('Preprocessing')
        while True:
            preprocessing_menu.display_menu()
            choice = preprocessing_menu.selection()

            # **** Json generation ****
            if choice == 1:
                self.launch_json_generation()

            # **** Normalization ****
            elif choice == 2:
                self.launch_normalisation()

            # **** Return main menu ****
            elif choice == 3:
                return

    def launch_json_generation(self, data_dir=None, file_name_report=None, append=False):
        """
        Generates a JSON file containing statistics about the datasets.
        """

        ut.print_box("JSON generation")

        # 1) Ensure data_dir is a Path
        if data_dir is None:
            data_dir = Path(ut.get_path_color("Enter the data directory"))
        else:
            data_dir = Path(data_dir)

        # 2) Choose percentage
        if ut.answer_yes_or_no("Do you want to use all the data to generate data statistics ?"):
            percentage_to_process = 1.0
        else:
            percentage_to_process = ut.input_percentage("Please enter a percentage between 1 and 100")

        # 3) JSON output path
        if file_name_report is None:
            file_name_report = data_dir / "data_stats.json"
        else:
            file_name_report = Path(file_name_report)

        # 4) Gather subfolder Paths
        subfolders = [p for p in data_dir.iterdir() if p.is_dir()]
        if not subfolders:
            self.display.print("The data directory is empty", colors["error"])
            return

        valid_subfolders = []
        for sub in subfolders:
            images_dir = sub / "images"
            if not images_dir.is_dir():
                self.display.print(f"{sub.name}/images not found", colors["error"])
                continue

            tif_files = list(images_dir.glob("*.tif"))
            if not tif_files:
                self.display.print(f"No tif files in {sub.name}/images", colors["error"])
                continue

            valid_subfolders.append(sub.name)

        if not valid_subfolders:
            self.display.print("No valid subfolders found for JSON generation", colors["error"])
            return

        # 5) Launch generation
        self.display.print("Starting JSON generation", colors["warning"])
        json_generation = json.Json(
            json.JsonConfig(
                parent_dir=data_dir,
                subfolders=valid_subfolders,
                json_file=file_name_report,
                percentage_to_process=percentage_to_process
            )
        )
        json_generation.process_datasets(append=append)

    def launch_normalisation(self):
        """
        Normalizes TIFF masks (8-bit grayscale) by first moving them to
        raw_masks/ and then rewriting masks/ with the normalized output.
        """
        ut.print_box("Data Normalisation")

        # 1) Get and validate data_dir
        data_dir = Path(ut.get_path_color("Enter the data directory"))
        if not data_dir.is_dir():
            self.display.print(f"Invalid data directory: {data_dir}", colors["error"])
            return

        # 2) Find all subfolders
        subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
        if not subdirs:
            self.display.print("The data directory is empty", colors["error"])
            return

        # 3) Identify which subfolders actually contain .tif masks
        valid_subfolders: list[Path] = []
        for sub in subdirs:
            masks_dir = sub / "masks"
            if not masks_dir.is_dir():
                self.display.print(f"{sub.name}/masks not found", colors["error"])
                return

            tif_files = list(masks_dir.glob("*.tif"))
            if not tif_files:
                self.display.print(f"No .tif files in {sub.name}/masks", colors["error"])
                return

            valid_subfolders.append(sub)

        if not valid_subfolders:
            self.display.print(
                "No valid subfolders found for normalization", colors["error"]
            )
            return

        # 4) Proceed with normalization
        self.display.print("Starting data normalisation", colors["warning"])
        for sub in valid_subfolders:
            masks_dir   = sub / "masks"
            raw_masks   = sub / "raw_masks"
            new_masks   = sub / "masks"  # will be recreated

            # a) Rename masks → raw_masks (only if raw_masks doesn’t already exist)
            if raw_masks.exists():
                self.display.print(
                    f"{sub.name}/raw_masks already exists, skipping rename", colors["warning"]
                )
            else:
                os.rename(masks_dir, raw_masks)

            # b) Recreate masks/ directory
            new_masks.mkdir(exist_ok=True)

            # c) Normalize all TIFFs
            normalizer = image_normalizer.ImageNormalizer(
                str(raw_masks), 
                str(new_masks)
            )
            normalizer.normalize_images()

        self.display.print("Data normalisation complete", colors["ok"])
