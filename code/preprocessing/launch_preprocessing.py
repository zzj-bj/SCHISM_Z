# -*- coding: utf-8 -*-
"""
This module allows for the following operations:
    - Json generation   *** In development ***
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
    select = vf.answer_yes_or_non("Used all the data")
    if select :
        vv = 1
    else :
        vv = vf.input_percentage("Enter a percentage between 1 & 100")

    file_name_report = fo.create_name_path(data_dir, '', 'data_stats.json')

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    valid_subfolders = []
    if len(subfolders) == 0 :
        report_json.add(" - No folder found in ", data_dir)
    else:
        for f in subfolders:
            images_path = fo.create_name_path(data_dir, f, 'images')

            if not os.path.isdir(images_path):
                report_json.add(" - Folder 'images' missing in :", f)
            else:
                nb_f_images = fo.count_tif_files(images_path)
                if nb_f_images == 0:
                    report_json.add(" - No file (*.tif') in folder 'images'  :", f)
                else:
                    valid_subfolders.append(f)


    if len(valid_subfolders) != 0 :
        print("[!] Starting Json generation")
        json_generation = js.DatasetProcessor(data_dir,
                                              valid_subfolders,
                                              file_name_report,
                                              report_json,
                                              vv)
        try:
            json_generation.process_datasets()
        except ValueError as e:
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
        report_normal.add(" - No folder found in ", data_dir)
    else:
        for f in subfolders:
            masks_path = fo.create_name_path(data_dir, f, 'masks')

            if not os.path.isdir(masks_path):
                report_normal.add(" - Folder 'masks' missing in :", f)
            else:
                nb_f_masks = fo.count_tif_files(masks_path)
                if nb_f_masks == 0:
                    report_normal.add(" - No file (*.tif') in folder 'masks'  :", f)
                else:
                    valid_subfolders.append(f)

    if len(valid_subfolders) != 0 :
        print("[!] Starting normalisation")
        for f in valid_subfolders:
            print(f" - {f} :")
            in_path = fo.create_name_path(data_dir, f, 'masks')
            out_path = fo.create_name_path(data_dir, f, 'normalized')
            os.makedirs(out_path, exist_ok=True)

            normalizer = no.ImageNormalizer(in_path, out_path, report_normal)

            try:
                normalizer.normalize_images()
            except ValueError as e:
                print(e)

    report_normal.status("Normalization")
