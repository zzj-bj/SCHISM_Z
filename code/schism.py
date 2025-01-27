import subprocess
import sys
import os
import tkinter as tk
from tkinter import filedialog
from classes.Training import Training
from classes.Inference import Inference
from classes.Hyperparameters import Hyperparameters
import random
import numpy as np
from tqdm import tqdm
#import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from patchify import patchify

current_dir = os.getcwd()
project_dir = os.path.join(current_dir, '..')  
sys.path.append(os.path.join(current_dir))

# Constants
VALID_FILETYPES = [("ini files", "*.ini")]

def show_main_menu():
    """Display the main menu options."""
    ascii_art = r"""
    ███████╗ ██████╗██╗  ██╗██╗███████╗███╗   ███╗
    ██╔════╝██╔════╝██║  ██║██║██╔════╝████╗ ████║
    ███████╗██║     ███████║██║███████╗██╔████╔██║
    ╚════██║██║     ██╔══██║██║╚════██║██║╚██╔╝██║
    ███████║╚██████╗██║  ██║██║███████║██║ ╚═╝ ██║
    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝
    """
    print(ascii_art)
    print("=========================")
    print("Please select an action:")
    print("1. Training")
    print("2. Inference")
    print("3. Quit")
    print("=========================")


def show_training_menu():
    """Display options related to training."""
    print("=========================")
    print("Please select an action:")
    print("1. Run Training")
    print("2. Go Back to Main Menu")
    print("=========================")


def select_directory(title: str) -> str:
    """Prompt user to select a directory and return the selected path."""
    return filedialog.askdirectory(title=title)


def select_file(title: str, filetypes: list) -> str:
    """Prompt user to select a file and return the selected file path."""
    return filedialog.askopenfilename(title=title, filetypes=filetypes)


def option_training():
    """Handle the training option."""
    print("=========================")
    print("The action performed is: training.")
    print("=========================")

    # Select directories and files
    data_dir = select_directory("Select your data folder")
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    run_dir = select_directory("Select your run folder")
    hyperparameters_path = select_file("Select your hyperparameter file", VALID_FILETYPES)
    
    # Load hyperparameters
    hyperparameters = Hyperparameters(hyperparameters_path)
    
    # Initialize the Training object
    train_object = Training(data_dir=data_dir,
                            subfolders=subfolders,
                            run_dir=run_dir,
                            hyperparameters=hyperparameters)

    # Display training options
    while True:
        show_training_menu()
        choice = input("Make a choice between 1 and 2: ")

        if choice == '1':
            train_object.load_segmentation_data()
            train_object.train()
        elif choice == '2':
            print("Returning to the main menu")
            break
        else:
            print("Invalid choice: Please select a valid option.")

def option_inference():
    """Handle the inference option."""
    print("=========================")
    print("The action performed is: inference.")
    print("=========================")
    
    # Select directories and files
    data_dir = select_directory("Select the data to be predited")
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    run_dir = select_directory("Select your weight directory")

    # Load hyperparameters
    hyperparameters = Hyperparameters(os.path.join(run_dir, 'hyperparameters.ini'))
    params = {k: v for k, v in hyperparameters.get_parameters()['Training'].items()}

    # Extract and split metrics, defaulting to 'Jaccard' if not provided or empty
    metrics = [metric.strip() for metric in params.get('metrics', 'Jaccard').split(',') if metric.strip()]

    if len(metrics) == 1:
        selected_metric = metrics[0]
        print(f"Only one metric found: {selected_metric}. Proceeding with it.")
    else:
        print("Multiple metrics found. Please choose one:")
        for i, metric in enumerate(metrics, 1):
            print(f"{i}. {metric}")
        
        # Ask the user to choose one metric
        while True:
            try:
                choice = int(input("Enter the number corresponding to your choice: "))
                if 1 <= choice <= len(metrics):
                    selected_metric = metrics[choice - 1]
                    print(f"Selected metric: {selected_metric}")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(metrics)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")


    # The rest of your code to use the selected_metric
    pred_object = Inference(data_dir=data_dir,
                            subfolders=subfolders,
                            run_dir=run_dir,
                            selected_metric=selected_metric,
                            hyperparameters=hyperparameters)
    pred_object.predict()
    print("Training done ;)")

def main():
    """Main function to run the application."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    while True:
        show_main_menu()
        choice = input("Make a choice between 1 to 3: ")

        if choice == '1':
            option_training()
        elif choice == '2':
            option_inference()
        elif choice == '3':
            print("Exiting SCHISM. Goodbye!")
            break
        else:
            print("Invalid choice! Please select a valid option.")


if __name__ == "__main__":
    main()
