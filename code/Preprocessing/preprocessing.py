# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:05:11 2025

@author: Pierre.FANCELLI
"""
import os

from tools import folder as fo
from tools import repport as re
from preprocessing import normalisation as no
from preprocessing import json_generation as jg


def launch_json_generation():
    """
    Json Generatio
    """
    print("\n[ Json Generation Mode ]")


    print("[!] Starting Json generation")
    json_generation = jg.JsonGeneration()
    try:
        json_generation.calcul()
        print("[√] Json Generation terminée\n")
    except ValueError as e:
        print(e)


def launch_normalisation():
    """
    Normalization of masks in 8-bit grayscale format
    """
    print("\n[ Normalisation Mode ]")

    data_dir = fo.get_path("Enter the data directory")
    report_dir = fo.get_path("Enter the directory to save report")

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    # Reataty a repport for errors
    report_prepo = re.Rapport()

    print("[!] Starting normalisation")
    valid_subfolders = []
    if len(subfolders) == 0 :
        report_prepo.ajouter_element(" - No folder found in ", data_dir)
    else:
        for f in subfolders:
            masks_path = fo.create_name_path(data_dir, f, 'masks')

            if not os.path.isdir(masks_path):
                report_prepo.ajouter_element(" - No folder 'Mask' found :", f)
            else:
                nb_f_masks = fo.compter_files(masks_path)
                if nb_f_masks == 0:
                    report_prepo.ajouter_element(" - No file in folder 'masks'  :", f)
                else:
                    valid_subfolders.append(f)

    if len(valid_subfolders) != 0 :
        for f in valid_subfolders:
            print(f" - {f} :")
            in_path = fo.create_name_path(data_dir, f, 'masks')
            out_path = os.path.join(data_dir, f, 'normalized')
            os.makedirs(out_path, exist_ok=True)

            normalizer = no.ImageNormalizer(in_path, out_path, report_prepo)
            try:
                normalizer.normalize_images()
            except ValueError as e:
                print(e)

    if report_prepo.exist_reppot():
        print("[X] Normalization terminée with error")
        report_prepo.afficher_rapport()
    else:
        print("[√] Normalization terminée without error\n")
