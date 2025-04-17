# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os

from tools import folder as fo
from tools import repport as re

from classes.Training import Training
from commun.hyperparameters import Hyperparameters

#===========================================================================
def train_model():
    """Executes the training process in CLI."""

    print("\n[ Training Mode ]")


    data_dir = fo.get_path("Enter the data directory")
    run_dir = fo.get_path("Enter the directory to save runs")
    hyperparameters_path = fo.get_path("Enter the path to the hyperparameters INI file")

    hyperparameters_path = os.path.join(hyperparameters_path, "hyperparameters.ini")
    hyperparameters = Hyperparameters(hyperparameters_path)

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    report_training = re.Rapport()

    # subfolders_t = subfolders.copy()
    # if len(subfolders_t) == 0 :
    #         report_training.ajouter_element(" - No folder found in ", data_dir)
    # else:
    #     for f in subfolders_t:
    #         images_path = fo.create_name_path(data_dir, f, 'images')
    #         masks_path = fo.create_name_path(data_dir, f, 'masks')
    #         if not os.path.isdir(images_path):
    #             report_training.ajouter_element(" - No folder 'images' found :", f)
    #             subfolders.remove(f)
    #         else :
    #             if not os.path.isdir(masks_path):
    #                 report_training.ajouter_element(" - No folder 'mask' found :", f)
    #                 subfolders.remove(f)
    #             else:
    #                 nb_f_image = fo.compter_files(images_path)
    #                 nb_f_masks = fo.compter_files(masks_path)
    #                 if nb_f_image == 0:
    #                     report_training.ajouter_element(" - No file in folder 'image'  :", f)
    #                     subfolders.remove(f)
    #                 else:
    #                     if nb_f_masks == 0:
    #                         report_training.ajouter_element(" - No file in folder 'masks'  :", f)
    #                         subfolders.remove(f)
    #                     else:
    #                         if nb_f_image != nb_f_masks :
    #                             report_training.ajouter_element(" - 'images '  &'masks' : Size not equal :", f)
    #                             subfolders.remove(f)


    print("[!] Starting training. ")

    train_object = Training(
        data_dir=data_dir,
        subfolders=subfolders,
        run_dir=run_dir,
        hyperparameters=hyperparameters
        )
    try:
        train_object.load_segmentation_data()
        train_object.train()
        print("[√] Training completed successfully!\n")
    except ValueError as e:
        print(e)
