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
    subfolders_t = subfolders.copy()


    print("[!] Starting training. ")
    report_training = re.Rapport()

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
