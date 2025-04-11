# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:51:50 2025

@author: Pierre.FANCELLI
"""

import os

from Menu.menu import selection as sl

# from classes.Training import Training
import training as tr

from Commun import hyperparameters as hy
# from classes.Hyperparameters import Hyperparameters

#==========================================================================
def train_model():
    """Executes the training process in CLI."""
    data_dir = sl.get_path("Enter the data directory")
    run_dir = sl.get_path("Enter the directory to save runs")
    hyperparameters_path = sl.get_path("Enter the path to the hyperparameters INI file")
    hyperparameters_path = os.path.join(hyperparameters_path, "hyperparameters.ini")

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    hyperparameters = hy.yperparameters(hyperparameters_path)

    print("\n[ Training Mode ]")
    print("[!] Starting training. ")

    try:
        train_object = tr.Training(
            data_dir=data_dir,
            subfolders=subfolders,
            run_dir=run_dir,
            hyperparameters=hyperparameters
        )
    except ValueError as e:
        print(e)

    train_object.load_segmentation_data()
    train_object.train()
    print("[√] Training completed successfully!\n")

