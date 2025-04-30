# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os

from tools import selection as sl
from tools import folder as fo
from tools import report as re

from classes.Inference import Inference
from commun.hyperparameters import Hyperparameters


#===========================================================================

def run_inference():
    """Executes the inference process in CLI."""
    print("\n[ Inference Mode ]")

    report_inference = re.ErrorReport()
    valid_subfolders = []

    data_dir = fo.get_path("Enter the directory containing data to predict")
    run_dir = fo.get_path("Enter the directory containing model weights")
    report_dir = fo.get_path("Enter the directory to save report")
    file_name_report = fo.create_name_path(report_dir, '', 'Inference')

    hyperparameters_path = os.path.join(run_dir, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        report_inference.add("hyperparameters.ini file was not found", "")
    else:
        hyperparameters = Hyperparameters(hyperparameters_path)
        params = hyperparameters.get_parameters().get("Training", {})
        metrics = [metric.strip()
                    for metric in params.get("metrics", "Jaccard").split(",") if metric.strip()]

        if not metrics:
            report_inference.add("No metrics found in the hyperparameters", "")
            return

        # Filter out 'ConfusionMatrix' if it's part of the metrics
        available_metrics = [metric for metric in metrics if metric != "ConfusionMatrix"]

        # Display the Metric available Menu
        menu_metric = ['Metric'] + available_metrics
        inference_menu = sl.Menu('Dynamic',menu_metric)
        inference_menu.display_menu()
        choice = inference_menu.selection()
        selected_metric = metrics[int(choice) - 1]
        print(f' - Metric selected = {selected_metric}')

        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

        if len(subfolders) == 0 :
            report_inference.add(" - No folder found in ", data_dir)
        else:
            for f in subfolders:
                images_path = fo.create_name_path(data_dir, f, 'images')

                if not os.path.isdir(images_path):
                    report_inference.add(" - Folder 'images' missing in :", f)
                else:
                    valid_subfolders.append(f)

    if len(valid_subfolders) != 0 :
        print("[!] Starting inference...")

        pred_object = Inference(
            data_dir=data_dir,
            subfolders=valid_subfolders,
            run_dir=run_dir,
            selected_metric=selected_metric,
            hyperparameters=hyperparameters
        )

        try:
            pred_object.predict()
        except ValueError as e:
            print(e)

    if report_inference.is_report():
        sl.display_color("[X] Inference finished with error", "red")
        print("Some directories have been removed from processing")
        report_inference.display_report()
    else:
        sl.display_color(f"[√] Inference ({selected_metric}) completed successfully!\n", "green")
    report_inference.print_report(file_name_report)
