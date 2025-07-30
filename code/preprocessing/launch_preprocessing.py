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

from tools import error_report as re
from tools import utils as ut
from tools import display_color as dc
from tools import constants as ct
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

    def launch_json_generation(self,data_dir=None,file_name_report=None,append=False):
        """
        Generates a JSON file containing statistics about the datasets.
        """

        ut.print_box("Json generation")

        report_json = re.ErrorReport()
        if data_dir is None:
            data_dir = ut.get_path_color("Enter the data directory")
        select = ut.answer_yes_or_no("Use all the data (100%) ?")
        if select :
            percentage_to_process = 1.0
        else :
            percentage_to_process = ut.input_percentage("Please enter a number between 1 and 100")


        if file_name_report is None:
            file_name_report = os.path.join(data_dir, '', 'data_stats.json')

        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

        valid_subfolders = []
        if len(subfolders) == 0 :
            report_json.add(" - The data directory is Empty ", '')
        else:
            for f in subfolders :
                path = os.path.join(data_dir, f, 'images')
                ut.validate_subfolders(path, f, valid_subfolders, report_json,
                                    folder_type='images')

        if report_json.is_report() == 0 :
            print("[!] Starting Json generation")
            json_generation = json.Json(
                                    json.JsonConfig(
                                        parent_dir = data_dir,
                                        subfolders=valid_subfolders,
                                        json_file=file_name_report,
                                        report=report_json,
                                        percentage_to_process=percentage_to_process
                                    ))
            json_generation.process_datasets(append=append)

        report_json.status("Json generation")

    def launch_normalisation(self):
        """
        Normalization of masks in 8-bit grayscale format
        """
        ut.print_box("Data Normalization")

        report_normal = re.ErrorReport()
        data_dir = ut.get_path_color("Enter the data directory")
        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

        valid_subfolders = []
        if len(subfolders) == 0 :
            report_normal.add(" - The data directory is Empty ", '')
        else:
            for f in subfolders :
                path = os.path.join(data_dir, f, 'masks')
                ut.validate_subfolders(path, f, valid_subfolders, report_normal,
                                    folder_type='masks')
            # # rename masks to raw_masks
            for f in valid_subfolders:
                old_directory = os.path.join(data_dir, f, 'masks')
                new_directory = os.path.join(data_dir, f, 'raw_masks')
                if not os.path.exists(new_directory):
                    os.rename(old_directory, new_directory)

        if report_normal.is_report() == 0 :
            print("[!] Starting Data normalization")
            for f in valid_subfolders:
                print(f" - {f} :")
                in_path = os.path.join(data_dir, f, 'raw_masks')
                out_path = os.path.join(data_dir, f, 'masks')
                os.makedirs(out_path, exist_ok=True)

                normalizer = image_normalizer.ImageNormalizer(in_path, out_path, report_normal)
                normalizer.normalize_images()
                
        report_normal.status("Data Normalization")
