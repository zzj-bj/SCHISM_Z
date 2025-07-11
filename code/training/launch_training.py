# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os

from tools import folder as fo
from tools import report as re

from commun.hyperparameters import Hyperparameters
from training import training as tr


#===========================================================================
def check_folder(folder, root, report):
    """
    This function checks that all the directories passed as parameters comply
    with the expected requirements.

    Parameters
    ----------
    folder : liste
        List of directories to analyze.
    root : str
        Reference path for analysis.
    report : str
        File for report.

    Returns
    -------
    valid_subfolders : liste
        List of valid directories.
    num_file : int
        Total number of files considered.

    """
    valid_subfolders =[]
    num_file = 0
    for f in folder:
        images_path = fo.create_name_path(root, f, 'images')
        masks_path = fo.create_name_path(root, f, 'masks')

        if not os.path.isdir(images_path):
            report.add(" - Folder 'images' missing in :", f)
        else :
            if not os.path.isdir(masks_path):
                report.add(" - Folder 'masks' missing in :", f)
            else:
                nb_f_image = fo.count_tif_files(images_path)
                nb_f_masks = fo.count_tif_files(masks_path)
                if nb_f_image == 0:
                    report.add(" - No file (*.tif') in folder 'image'  :", f)
                else:
                    if nb_f_masks == 0:
                        report.add(" - No file (*.tif') in folder 'masks'  :", f)
                    else:
                        if nb_f_image != nb_f_masks :
                            report.add(" - 'images/masks' : Size not equal :", f)
                        else:
                            num_file += nb_f_image
                            valid_subfolders.append(f)
    return  valid_subfolders, num_file

def train_model():
    """Executes the training process in CLI."""

    print("\n[ Training Mode ]")

    report_training = re.ErrorReport()

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
        num_file = 0
        if len(subfolders) == 0 :
            report_training.add(" - The data directory is Empty ", '')
        else:
            valid_subfolders, num_file = check_folder(subfolders, data_dir, report_training)

        if len(valid_subfolders) != 0 :
            print("[!] Starting training.")

            try:
                train_object = tr.Training(
                    data_dir=data_dir,
                    subfolders=valid_subfolders,
                    run_dir=run_dir,
                    hyperparameters=hyperparameters,
                    report = report_training,
                    num_file = num_file
                    )

                train_object.load_segmentation_data()
                train_object.train()
            except (IOError, ValueError) as e:
                print(e)

    report_training.status("Training")
