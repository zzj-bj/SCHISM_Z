# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os

from tools import folder as fo
from tools import report as re
from tools import selection as sl

from commun.hyperparameters import Hyperparameters

from training import Training as tr

#===========================================================================
def check_folder(folder, root, report):

    valid_subfolders =[]
    for f in folder:
        images_path = fo.create_name_path(root, f, 'images')
        masks_path = fo.create_name_path(root, f, 'masks')

        if not os.path.isdir(images_path):
            report.add(" - Folder 'images' missing in :", f)
        else :
            if not os.path.isdir(masks_path):
                report.add(" - Folder 'masks' missing in :", f)
            else:
                nb_f_image = fo.compter_tif_files(images_path)
                nb_f_masks = fo.compter_tif_files(masks_path)
                if nb_f_image == 0:
                    report.add(" - No file in folder 'image' :", f)
                else:
                    if nb_f_masks == 0:
                        report.add(" - No file in folder 'masks' :", f)
                    else:
                        if nb_f_image != nb_f_masks :
                            report.add(" - 'images/masks' : Size not equal :", f)
                        else:
                            valid_subfolders.append(f)
    return  valid_subfolders

def train_model():
    """Executes the training process in CLI."""

    print("\n[ Training Mode ]")

    report_training = re.ErrorReport()
    data_dir = fo.get_path("Enter the data directory")
    run_dir = fo.get_path("Enter the directory to save runs")
    hyperparameters_path = fo.get_path("Enter the path to the hyperparameters INI file")

    report_dir = fo.get_path("Enter the directory to save report")
    file_name_report = fo.create_name_path(report_dir, '', 'Training')

    hyperparameters_path = os.path.join(hyperparameters_path, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        report_training.add("hyperparameters.ini file was not found", "")
    else:
        hyperparameters = Hyperparameters(hyperparameters_path)

        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

        valid_subfolders = []
        if len(subfolders) == 0 :
            report_training.add(" - No folder found in : ", data_dir)
        else:
            valid_subfolders = check_folder(subfolders, data_dir, report_training)

        if len(valid_subfolders) != 0 :
            print("[!] Starting training.")

            train_object = tr.Training(
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
        sl.display_color("[X] Training finished with error", "red")
        print("Some directories have been removed from processing")
        report_training.display_report()
    else:
        sl.display_color("[√] Training completed successfully!\n", "green")
    report_training.print_report(file_name_report)
