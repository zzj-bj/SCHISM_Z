# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os

from tools import folder as fo
from tools import report as re

from classes.Training import Training
from commun.hyperparameters import Hyperparameters

#===========================================================================
def train_model():
    """Executes the training process in CLI."""

    print("\n[ Training Mode ]")

    report_training = re.Error_Report()
    data_dir = fo.get_path("Enter the data directory")
    run_dir = fo.get_path("Enter the directory to save runs")
    hyperparameters_path = fo.get_path("Enter the path to the hyperparameters INI file")

    hyperparameters_path = os.path.join(hyperparameters_path, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        report_training.add("hyperparameters.ini file was not found", "")
    else:
        hyperparameters = Hyperparameters(hyperparameters_path)

        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

        valid_subfolders = []
        if len(subfolders) == 0 :
                report_training.add(" - No folder found in ", data_dir)
        else:
            for f in subfolders:
                images_path = fo.create_name_path(data_dir, f, 'images')
                masks_path = fo.create_name_path(data_dir, f, 'masks')

                if not os.path.isdir(images_path):
                    report_training.add(" - No folder 'images' found :", f)
                else :
                    if not os.path.isdir(masks_path):
                        report_training.add(" - No folder 'mask' found :", f)
                    else:
                        nb_f_image = fo.compter_tif_files(images_path)   # compter_files
                        nb_f_masks = fo.compter_tif_files(masks_path)    # compter_files



                        if nb_f_image == 0:
                            report_training.add(" - No file in folder 'image'  :", f)
                        else:
                            if nb_f_masks == 0:
                                report_training.add(" - No file in folder 'masks'  :", f)
                            else:
                                if nb_f_image != nb_f_masks :
                                    report_training.add(" - 'images ' &'masks' : Size not equal :", f)
                                else:
                                    valid_subfolders.append(f)

        if len(valid_subfolders) != 0 :
            print(f"[!] Starting training.")

            train_object = Training(
                data_dir=data_dir,
                subfolders=valid_subfolders,
                run_dir=run_dir,
                hyperparameters=hyperparameters
                )

            try:
                train_object.load_segmentation_data()
                train_object.train()
            except ValueError as e:
                print(e)

    if report_training.is_report():
        print("[X] Training finished with error")
        report_training.display_report()
    else:
        print("[√] Training completed successfully!\n")
