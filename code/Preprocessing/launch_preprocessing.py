# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:05:11 2025

@author: Pierre.FANCELLI
"""
import os

from tools import folder as fo
from tools import report as re

from preprocessing import normalisation as no
from preprocessing import json_generation as jg

#=============================================================================
def launch_json_generation():
    """
    Json Generation

             *** Development in progress ***
    """
    print("\n[ Json Generation Mode ]")

    report_json = re.ErrorReport()
    data_dir = fo.get_path("Enter the data directory")
    report_dir = fo.get_path("Enter the directory to save report")
    file_name_report = fo.create_name_path(report_dir, '', 'Json')

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    valid_subfolders = []
    if len(subfolders) == 0 :
        report_json.add(" - No folder found in ", data_dir)
    else:
        for f in subfolders:
            #TODO to be defined
            pass

    #TODO waiting work : "or True" to be deleter
    if len(valid_subfolders) != 0 or True :
        print("[!] Starting Json generation")
        json_generation = jg.JsonGeneration(report_json)
        try:
            json_generation.process()
        except ValueError as e:
            print(e)

    report_json.status("Json Generation Mode", file_name_report)

def launch_normalisation():
    """
    Normalization of masks in 8-bit grayscale format
    """
    print("\n[ Normalization Mode ]")

    report_mormal = re.ErrorReport()
    data_dir = fo.get_path("Enter the data directory")
    report_dir = fo.get_path("Enter the directory to save report")
    file_name_report = fo.create_name_path(report_dir, '', 'Normalization')

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    valid_subfolders = []
    if len(subfolders) == 0 :
        report_mormal.add(" - No folder found in ", data_dir)
    else:
        for f in subfolders:
            masks_path = fo.create_name_path(data_dir, f, 'masks')

            if not os.path.isdir(masks_path):
                report_mormal.add(" - Folder 'masks' missing in :", f)
            else:
                nb_f_masks = fo.compter_files(masks_path)
                if nb_f_masks == 0:
                    report_mormal.add(" - No file in folder 'masks'  :", f)
                else:
                    valid_subfolders.append(f)

    if len(valid_subfolders) != 0 :
        print("[!] Starting normalisation")
        for f in valid_subfolders:
            print(f" - {f} :")
            in_path = fo.create_name_path(data_dir, f, 'masks')
            out_path = fo.create_name_path(data_dir, f, 'normalized')
            os.makedirs(out_path, exist_ok=True)

            normalizer = no.ImageNormalizer(in_path, out_path, report_mormal)

            try:
                normalizer.normalize_images()
            except ValueError as e:
                print(e)

    report_mormal.status("Normalization", file_name_report)
