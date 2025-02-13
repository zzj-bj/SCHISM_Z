import os
import sys
from classes.Training import Training
from classes.Inference import Inference
from classes.Hyperparameters import Hyperparameters

ASCII_MENU = """
╔══════════════════════════════════════════════════╗
║  ███████╗ ██████╗██╗  ██╗██╗███████╗███╗   ███╗  ║
║  ██╔════╝██╔════╝██║  ██║██║██╔════╝████╗ ████║  ║
║  ███████╗██║     ███████║██║███████╗██╔████╔██║  ║
║  ╚════██║██║     ██╔══██║██║╚════██║██║╚██╔╝██║  ║
║  ███████║╚██████╗██║  ██║██║███████║██║ ╚═╝ ██║  ║
║  ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝  ║
╠══════════════════════════════════════════════════╣
║        -Let's do some cool segmentation-         ║
╠══════════════════════════════════════════════════╣
║ ▓▓▓▓▓▓ 1 -► Training       ░░░░░░░░░░░░░░░░██████║
║ ▓▓▓▓ 2 -► Inference     ░░░░░░░░░░░░░░░░░████████║
║ ▓▓ 3 -► Quit        ░░░░░░░░░░░░░░░░░░███████████║
╚══════════════════════════════════════════════════╝
"""

def get_path(prompt):
    """Requests a valid path from the user."""
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        print("[X] Invalid path. Try again.")

def train_model():
    """Executes the training process in CLI."""
    print("\n[ Training Mode ]")
    data_dir = get_path("Enter the data directory")
    run_dir = get_path("Enter the directory to save runs")
    hyperparameters_path = get_path("Enter the path to the hyperparameters INI file")
    hyperparameters_path = os.path.join(hyperparameters_path, "hyperparameters.ini")

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    hyperparameters = Hyperparameters(hyperparameters_path)

    train_object = Training(
        data_dir=data_dir,
        subfolders=subfolders,
        run_dir=run_dir,
        hyperparameters=hyperparameters
    )

    print("\n[!] Starting training...")
    train_object.load_segmentation_data()
    train_object.train()
    print("\n[✓] Training completed successfully!")

def run_inference():
    """Executes the inference process in CLI."""
    print("\n[ Inference Mode ]")
    data_dir = get_path("Enter the directory containing data to predict")
    run_dir = get_path("Enter the directory containing model weights")

    hyperparameters_path = os.path.join(run_dir, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        print("[X] The hyperparameters.ini file was not found in the weights directory.")
        return

    hyperparameters = Hyperparameters(hyperparameters_path)
    params = hyperparameters.get_parameters().get("Training", {})
    metrics = [metric.strip() for metric in params.get("metrics", "Jaccard").split(",") if metric.strip()]

    if not metrics:
        print("[X] No metrics found in the hyperparameters.")
        return

    print("\n[?] Choose a metric for inference:")
    for i, metric in enumerate(metrics, start=1):
        print(f" {i} --> {metric}")

    while True:
        choice = input("\n[?] Enter the metric number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(metrics):
            selected_metric = metrics[int(choice) - 1]
            break
        print("[X] Invalid selection. Try again.")

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
    
    print("\n[✓] Inference completed successfully!")

def main():
    """Displays the CLI menu and handles user choices."""
    while True:
        print("\n" + ASCII_MENU)

        choice = input("[?] Make your selection: ").strip()
        if choice == "1":
            train_model()
        elif choice == "2":
            run_inference()
        elif choice == "3":
            print("\n[:)] Goodbye! o/")
            sys.exit()
        else:
            print("[X] Invalid choice. Try again.")

if __name__ == "__main__":
    main()
