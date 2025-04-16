# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os

from tools import selection as sl
from tools import folder as fo

from classes.Inference import Inference
from commun.hyperparameters import Hyperparameters


#===========================================================================

def run_inference():
    """Executes the inference process in CLI."""
    print("\n[ Inference Mode ]")

    data_dir = fo.get_path("Enter the directory containing data to predict")
    run_dir = fo.get_path("Enter the directory containing model weights")

    hyperparameters_path = os.path.join(run_dir, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        print("[X] The hyperparameters.ini file was not found in the weights directory.")
        return

    hyperparameters = Hyperparameters(hyperparameters_path)
    params = hyperparameters.get_parameters().get("Training", {})
    metrics = [metric.strip()
                for metric in params.get("metrics", "Jaccard").split(",") if metric.strip()]

    if not metrics:
        print("[X] No metrics found in the hyperparameters.")
        return

    # Filter out 'ConfusionMatrix' if it's part of the metrics
    available_metrics = [metric for metric in metrics if metric != "ConfusionMatrix"]


    menu_metric = ['Metric'] + available_metrics
    main_menu = sl.Menu('Dynamic',menu_metric)
    main_menu.display_menu()
    choice = main_menu.selection()
    selected_metric = metrics[int(choice) - 1]
    print(f' - Metric selected = {selected_metric}')

    # # Display the available metrics for inference
    # for i, metric in enumerate(available_metrics, start=1):
    #     print(f" {i} --> {metric}")

    # while True:
    #     choice = input("\n[?] Enter the metric number: ").strip()
    #     if choice.isdigit() and 1 <= int(choice) <= len(metrics):
    #         selected_metric = metrics[int(choice) - 1]
    #         break
    #     fo.InvalidInput('Invalid selection.').invalid_input()


    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    pred_object = Inference(
        data_dir=data_dir,
        subfolders=subfolders,
        run_dir=run_dir,
        selected_metric=selected_metric,
        hyperparameters=hyperparameters
    )

    print("[!] Starting inference...")
    try:
        pred_object.predict()
        print("[√] Inference completed successfully!")
    except ValueError as e:
        print(e)
