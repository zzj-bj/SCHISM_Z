# -*- coding: utf-8 -*-
"""
Launches the inference process for a trained model.

This script prompts the user for the necessary directories and hyperparameters,
validates the input, and then runs the inference process using the specified model.

@author : Pierre.FANCELLI
"""
import os
from commun.hyperparameters import Hyperparameters

from tools import selection as sl
from tools import report as re
from tools import various_functions as vf

from inference.inference import Inference

#===========================================================================
def run_inference():
    """
    Launches the inference process for a trained model.
    """
    print("\n[ Inference Mode ]")

    hyperparameters = None

    report_inference = re.ErrorReport()

    valid_subfolders = []
    selected_metric =""
    initial_condition = True

    data_dir = vf.get_path("Enter the directory containing data to predict")
    run_dir = vf.get_path("Enter the directory containing model weights")

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    if len(subfolders) == 0 :
        report_inference.add(" - The directory containing data to predict is Empty ", '')
        initial_condition = False
    else:
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

            for f in subfolders :
                path = vf.create_name_path(data_dir, f, 'images')
                vf.validate_subfolders(path, f, valid_subfolders, report_inference,
                                    folder_type='images')

    if initial_condition and len(valid_subfolders) != 0 :
        print(f"[!] Starting inference with Metric : {selected_metric}.")

        try:
            pred_object = Inference(
                data_dir=data_dir,
                subfolders=valid_subfolders,
                run_dir=run_dir,
                selected_metric=selected_metric,
                hyperparameters=hyperparameters,
                report = report_inference,
            )

            try:
                pred_object.predict()
            except (IOError, ValueError) as e:
                report_inference.add(e, '')
                # print(f" ValueError during prediction:\n {e}")

        except FileNotFoundError as e:
            text = f" FileNotFoundError during model initialization:\n {e}"
            report_inference.add(text, '')
            # print(f" FileNotFoundError during model initialization:\n {e}")
        except (IOError, ValueError) as e:
            text = f" An unexpected error occurred:\n {e}"
            report_inference.add(text, '')
            # print(f" An unexpected error occurred:\n {e}")


    text = f"Inference with Metric '{selected_metric}'"
    report_inference.status(text)
