# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os

from tools import selection as sl
from tools import folder as fo
from tools import report as re

from inference import inference as nf
from commun.hyperparameters import Hyperparameters


#===========================================================================
def run_inference():
    """Executes the inference process in CLI."""
    print("\n[ Inference Mode ]")

    report_inference = re.ErrorReport()

    valid_subfolders = []
    selected_metric =""
    initial_condition = True

    data_dir = fo.get_path("Enter the directory containing data to predict")
    run_dir = fo.get_path("Enter the directory containing model weights")

    hyperparameters_path = os.path.join(run_dir, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        report_inference.add("hyperparameters.ini file was not found", "")
        initial_condition = False
    else:
        hyperparameters = Hyperparameters(hyperparameters_path)
        params = hyperparameters.get_parameters().get("Training", {})
        metrics = [metric.strip()
                    for metric in params.get("metrics", "Jaccard").split(",") if metric.strip()]

        # Check for Metrics
        if not metrics:
            report_inference.add("No metrics found in the hyperparameters", "")
            initial_condition = False
        else :
            # Filter out 'ConfusionMatrix' if it's part of the metrics
            available_metrics = [metric for metric in metrics if metric != "ConfusionMatrix"]

            # Display the Metric available Menu
            menu_metric = ['Metric'] + available_metrics
            metric_menu = sl.Menu('Dynamic',menu_metric, style = 'rounds')
            metric_menu.display_menu()
            choice = metric_menu.selection()
            selected_metric = metrics[choice - 1]
            print(f' - Metric selected : {selected_metric}')

            # Check for the existence of the *.pth files
            files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
            if len(files) == 0:
                report_inference.add(" File (*.pth) for Metrics not found", "")
                initial_condition = False

        # Check for the existence of the *.Json file
        json_file_path = os.path.join(run_dir, 'data_stats.json')
        if not os.path.exists(json_file_path):
            report_inference.add(f" File {json_file_path} not found", "")
            initial_condition = False

        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        if len(subfolders) == 0 :
            report_inference.add(" - No folder found in ", data_dir)
            initial_condition = False
        else:
            for f in subfolders:
                images_path = fo.create_name_path(data_dir, f, 'images')

                if not os.path.isdir(images_path):
                    report_inference.add(" - Folder 'images' missing in :", f)
                else:
                    valid_subfolders.append(f)


    if initial_condition and len(valid_subfolders) != 0 :
        print("[!] Starting inference...")

        try:
            pred_object = nf.Inference(
                data_dir=data_dir,
                subfolders=valid_subfolders,
                run_dir=run_dir,
                selected_metric=selected_metric,
                hyperparameters=hyperparameters,
                report = report_inference,
            )

            try:
                pred_object.predict()
            except ValueError as e:
                print(f" ValueError during prediction:\n {e}")

        except FileNotFoundError as e:
            print(f" FileNotFoundError during model initialization:\n {e}")
        except Exception as e:
            print(f" An unexpected error occurred:\n {e}")


    text = f"Inference with Metric '{selected_metric}'"
    report_inference.status(text)
