# -*- coding: utf-8 -*-
"""
This module allows for the following operations:
    - Json generation.
    - Normalization of masks in 8-bit grayscale format.

@author: Pierre.FANCELLI
"""

import os

from tools import folder as fo
from tools import report as re
from tools import various_functions as vf

from preprocessing import image_normalizer as no
from preprocessing import j_son as js

#=============================================================================

def launch_json_generation():
    """
    Json Generation

    """
    print("\n[ Json Generation Mode ]")

    report_json = re.ErrorReport()

    data_dir = fo.get_path("Enter the data directory")
    select = vf.answer_yes_or_no("Use all the data (100%) ")
    if select :
        percentage_to_process = 1.0
    else :
        percentage_to_process = vf.input_percentage("Enter a percentage between 1 & 100")

    file_name_report = fo.create_name_path(data_dir, '', 'data_stats.json')

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    valid_subfolders = []
    if len(subfolders) == 0 :
        report_json.add(" - The data directory is Empty ", '')
    else:
        for f in subfolders :
            path = fo.create_name_path(data_dir, f, 'images')
            vf.validate_subfolders_0_(path, f, valid_subfolders, report_json,
                                folder_type='images')
    vf.warning_message(subfolders, valid_subfolders)

    if len(valid_subfolders) != 0 :
        print("[!] Starting Json generation")
        json_generation = js.DatasetProcessor(data_dir,
                                              valid_subfolders,
                                              file_name_report,
                                              report_json,
                                              percentage_to_process)
        try:
            json_generation.process_datasets()
        except (IOError, ValueError) as e:
            print(e)

    report_json.status("Json Generation Mode")

def launch_normalisation():
    """
    Normalization of masks in 8-bit grayscale format
    """
    print("\n[ Normalization Mode ]")

    report_normal = re.ErrorReport()

    data_dir = fo.get_path("Enter the data directory")

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    valid_subfolders = []
    if len(subfolders) == 0 :
        report_normal.add(" - The data directory is Empty ", '')
    else:
        for f in subfolders :
            path = fo.create_name_path(data_dir, f, 'masks')
            vf.validate_subfolders_0_(path, f, valid_subfolders, report_normal,
                                folder_type='masks')
    vf.warning_message(subfolders, valid_subfolders)

    if len(valid_subfolders) != 0 :
        print("[!] Starting normalization")
        for f in valid_subfolders:
            print(f" - {f} :")
            in_path = fo.create_name_path(data_dir, f, 'masks')
            out_path = fo.create_name_path(data_dir, f, 'normalized')
            os.makedirs(out_path, exist_ok=True)

            normalizer = no.ImageNormalizer(in_path, out_path, report_normal)

            try:
                normalizer.normalize_images()
            except (IOError, ValueError) as e:
                print(e)

    report_normal.status("Normalization")
