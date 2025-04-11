# -*- coding: utf-8 -*-
""" SCHISM """

import os
import sys

from Menu import selection as sl
from Preprocessing import preprocessing  as pr


from Training import lauch_training as tr
# from classes.Training import Training


from classes.Inference import Inference

from Commun import hyperparameters as hy
# from classes.Hyperparameters import Hyperparameters

#=========================================================================

# def train_model():
#     """Executes the training process in CLI."""
#     data_dir = sl.get_path("Enter the data directory")
#     run_dir = sl.get_path("Enter the directory to save runs")
#     hyperparameters_path = sl.get_path("Enter the path to the hyperparameters INI file")
#     hyperparameters_path = os.path.join(hyperparameters_path, "hyperparameters.ini")

#     subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
#     hyperparameters = hy.Hyperparameters(hyperparameters_path)

#     print("\n[ Training Mode ]")
#     print("[!] Starting training. ")

#     try:
#         train_object = Training(
#             data_dir=data_dir,
#             subfolders=subfolders,
#             run_dir=run_dir,
#             hyperparameters=hyperparameters
#         )
#     except ValueError as e:
#         print(e)

#     train_object.load_segmentation_data()
#     train_object.train()
#     print("[√] Training completed successfully!\n")

#=======================================================================

def run_inference():
    """Executes the inference process in CLI."""
    # print("\n[ Inference Mode ]")
    data_dir = sl.get_path("Enter the directory containing data to predict")
    run_dir = sl.get_path("Enter the directory containing model weights")

    hyperparameters_path = os.path.join(run_dir, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        print("[X] The hyperparameters.ini file was not found in the weights directory.")
        return

    hyperparameters = hy.hyperparameters(hyperparameters_path)
    params = hyperparameters.get_parameters().get("Training", {})
    metrics = [metric.strip()
                for metric in params.get("metrics", "Jaccard").split(",") if metric.strip()]

    if not metrics:
        print("[X] No metrics found in the hyperparameters.")
        return

    # Filter out 'ConfusionMatrix' if it's part of the metrics
    available_metrics = [metric for metric in metrics if metric != "ConfusionMatrix"]

    # Display the available metrics for inference
    for i, metric in enumerate(available_metrics, start=1):
        print(f" {i} --> {metric}")

    while True:
        choice = input("\n[?] Enter the metric number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(metrics):
            selected_metric = metrics[int(choice) - 1]
            break
        sl.InvalidInput('Invalid selection.').invalid_input()


    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    pred_object = Inference(
        data_dir=data_dir,
        subfolders=subfolders,
        run_dir=run_dir,
        selected_metric=selected_metric,
        hyperparameters=hyperparameters
    )

    print("\n[!] Starting inference...")

    pred_object.predict()

    print("\n[√] Inference completed successfully!")



def menu_preprocessing():
    """
    "This module allows for the following operations:
        - Json generation   *** In development ***
        - Normalization of masks in 8-bit grayscale format."
    """
    preprocessing_menu = sl.Menu('Preprocessing')
    while True:
        preprocessing_menu.display_menu()
        choice = preprocessing_menu.selection()

        #TODO In development
        # **** Json generation ****
        if choice == 1:
            pr.launch_json_generation()

        # **** Normalisation ****
        elif choice == 2:
            pr.launch_normalisation()

        # **** Return main menu ****
        elif choice == 3:
            return

def main():
    """Displays the CLI menu and handles user choices."""
    main_menu = sl.Menu('MAIN')
    while True:
        print(sl.LOGO) # "Display the logo SCHISM

        main_menu.display_menu()
        choice = main_menu.selection()

        # Menu Preprocessing
        if choice == 1:
            menu_preprocessing()

        # Training
        elif choice == 2:
            tr.train_model()
            pass

         # Inference
        elif choice == 3:
            run_inference()
            # pass

        # Fin de Programme
        elif choice == 4:
            print("[<3] Goodbye! o/")
            sys.exit()

if __name__ == "__main__":
    main()
