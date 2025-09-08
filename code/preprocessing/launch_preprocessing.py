# -*- coding: utf-8 -*-

"""
Launches the preprocessing operations for datasets.
This script provides functionalities for preprocessing datasets,
including auto brightness/contrast adjustment, JSON generation and image normalization.
It prompts the user for the necessary directories and parameters,
validates the input, and then executes the preprocessing tasks.

This module allows for the following operations:
    - Auto brightness/contrast adjustment.
    - Json generation.
    - Normalization of masks in 8-bit grayscale format.

@author: Pierre.FANCELLI
"""

# Standard library
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Local application imports
from tools import menu, utils as vf
from tools import display_color as dc
from tools.constants import DISPLAY_COLORS as colors
from tools.constants import IMAGE_EXTENSIONS
from preprocessing import brightness_contrast_adjuster
from preprocessing import image_normalizer
from preprocessing import json


#=============================================================================

class LaunchPreprocessing:

    def __init__(self) -> None:

        self.display = dc.DisplayColor()

    def menu_preprocessing(self) -> None:
        """
        This module allows for the following operations:
            - Auto brightness/contrast adjustment.
            - Json generation
            - Normalization of masks in 8-bit grayscale format.
        """

        preprocessing_menu = menu.Menu('Preprocessing')
        while True:
            preprocessing_menu.display_menu()
            choice = preprocessing_menu.selection()

            # **** Auto brightness/contrast adjustment ****
            if choice == 1:
                self.launch_auto_adjust_brightness_contrast()

            # **** Json generation ****
            if choice == 2:
                self.launch_json_generation()

            # **** Normalization ****
            elif choice == 3:
                self.launch_normalisation()

            # **** Return main menu ****
            elif choice == 4:
                return

    def launch_auto_adjust_brightness_contrast(self) -> None:
        """
        Perform automatic brightness/contrast adjustment similar to Fiji's autoAdjust()
        for TIFF images (8-bit grayscale).
        """

        vf.print_box("Auto brightness/contrast adjustment")

        # 1) Get and validate data_dir
        sequence_dir = Path(vf.get_path_color("Enter the image sequence directory"))
        if not sequence_dir.is_dir():
            self.display.print(f"Invalid sequence directory: {sequence_dir}", colors["error"])
            return

        # 2) Select the hmin/hmax calculation mode used in auto brightness/contrast adjustment:
        # - "ref image": the reference image is used for reference to defines hmin/hmax for all images
        # - "per image": each image calculates its own hmin/hmax
        prompt = "Select the calculation mode of hmin/hmax\n" \
                " (r)ef image / (p)er image "
        hmin_hmax_calc_mode = vf.get_hmin_hmax_calc_mode(prompt)

        # 3) Identify all .tif files in the input directory
        tif_files = []
        for ext in IMAGE_EXTENSIONS:
            tif_files.extend(sequence_dir.glob(f"*{ext}"))
        tif_files = sorted(tif_files)

        if not tif_files:
            self.display.print(f"No .tif files in {sequence_dir}", colors["error"])
            return

        # 4) Select the reference image if using "ref image" mode
        ref_img_name = None
        ref_img_path = None
        if hmin_hmax_calc_mode == "ref_image":
            ref_img_name = Path(vf.get_file_name_color("Enter the reference image name"))
            ref_img_path = Path.joinpath(sequence_dir, ref_img_name)

            if not ref_img_path.is_file():
                self.display.print(f"Invalid reference image: {ref_img_name}", colors["error"])
                return

        # 5) Proceed with auto brightness/contrast adjustment
        self.display.print("Starting auto brightness/contrast adjustment", colors["warning"])
        raw_folder_name = "raw_" + os.path.basename(sequence_dir)
        raw_images = os.path.join(os.path.dirname(sequence_dir), raw_folder_name)

        # a) Rename images → raw_images (only if raw_images doesn’t already exist)
        if os.path.exists(raw_images):
            self.display.print(
                f"{raw_images} already exists, skipping rename", colors["warning"]
            )
        else:
            os.rename(sequence_dir, raw_images)

        # b) Adapt reference image path if needed
        if not ref_img_path is None:
            ref_img_path = os.path.join(raw_images, ref_img_name)    

        # c) Recreate images directory
        sequence_dir.mkdir(exist_ok=True)

        # 6) Auto brightness/contrast adjustment
        auto_adjuster = brightness_contrast_adjuster.BrightnessContrastAdjuster(
            input_path=raw_images,
            output_path=sequence_dir
        )
        auto_adjuster.auto_adjust_brightness_contrast(hmin_hmax_mode=hmin_hmax_calc_mode, ref_img_path=ref_img_path)

    def launch_json_generation(self,
        data_dir: str | None = None,
        file_name_report: str | None = None,
        append: bool = False
    ) -> None:
        """
        Generates a JSON file containing statistics about the datasets.
        """

        vf.print_box("JSON generation")

        # 1) Ensure data_dir is a Path
        if data_dir is None:
            data_dir = Path(vf.get_path_color("Enter the data directory"))
        else:
            data_dir = Path(data_dir)

        # 2) Choose percentage
        if vf.answer_yes_or_no("Do you want to use all the data to generate data statistics ?"):
            percentage_to_process = 1.0
        else:
            prompt = "Please enter a percentage between 1 and 100"
            percentage_to_process = vf.input_percentage(prompt)

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

            tif_files = []
            for ext in IMAGE_EXTENSIONS:
                tif_files.extend(images_dir.glob(f"*{ext}"))
            tif_files = sorted(tif_files)

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

        self.display.print("JSON generation complete", colors["ok"])


    def launch_normalisation(self)-> None:
        """
        Normalizes TIFF masks (8-bit grayscale) by first moving them to
        raw_masks/ and then rewriting masks/ with the normalized output.
        """

        vf.print_box("Data Normalisation")

        # 1) Get and validate data_dir
        data_dir = Path(vf.get_path_color("Enter the data directory"))
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

            tif_files = []
            for ext in IMAGE_EXTENSIONS:
                tif_files.extend(masks_dir.glob(f"*{ext}"))
            tif_files = sorted(tif_files)

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
