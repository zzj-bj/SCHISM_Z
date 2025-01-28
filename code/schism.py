import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from classes.Training import Training
from classes.Inference import Inference
from classes.Hyperparameters import Hyperparameters
import threading

current_dir = os.getcwd()
project_dir = os.path.join(current_dir, '..')
sys.path.append(os.path.join(current_dir))

# Constants
VALID_FILETYPES = [("ini files", "*.ini")]

ASCII_ART = (
    "    ███████╗ ██████╗██╗  ██╗██╗███████╗███╗   ███╗\n"
    "    ██╔════╝██╔════╝██║  ██║██║██╔════╝████╗ ████║\n"
    "    ███████╗██║     ███████║██║███████╗██╔████╔██║\n"
    "    ╚════██║██║     ██╔══██║██║╚════██║██║╚██╔╝██║\n"
    "    ███████║╚██████╗██║  ██║██║███████║██║ ╚═╝ ██║\n"
    "    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝\n"
)

def create_ascii_frame(root):
    """Create a frame for displaying the SCHISM ASCII art logo."""
    ascii_frame = tk.Frame(root)
    ascii_label = tk.Label(ascii_frame, text=ASCII_ART, font=("Courier", 12), justify="center")
    ascii_label.pack(pady=10)
    return ascii_frame

def option_training():
    """Handle the training option."""
    training_window = tk.Toplevel()
    training_window.title("SCHISM - Training")

    create_ascii_frame(training_window).pack()

    label = tk.Label(training_window, text="Select directories and files for training:")
    label.pack(pady=10)

    data_dir_var = tk.StringVar()
    run_dir_var = tk.StringVar()
    hyperparameters_var = tk.StringVar()

    def browse_data_dir():
        data_dir = filedialog.askdirectory(title="Select your data folder")
        data_dir_var.set(data_dir)

    def browse_run_dir():
        run_dir = filedialog.askdirectory(title="Select your run folder")
        run_dir_var.set(run_dir)

    def browse_hyperparameters():
        hyperparameters_path = filedialog.askopenfilename(title="Select your hyperparameter file", filetypes=VALID_FILETYPES)
        hyperparameters_var.set(hyperparameters_path)

    def start_training():
        data_dir = data_dir_var.get()
        run_dir = run_dir_var.get()
        hyperparameters_path = hyperparameters_var.get()

        if not (data_dir and run_dir and hyperparameters_path):
            messagebox.showerror("Error", "Please select all required directories and files.")
            return

        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        hyperparameters = Hyperparameters(hyperparameters_path)
        train_object = Training(
            data_dir=data_dir,
            subfolders=subfolders,
            run_dir=run_dir,
            hyperparameters=hyperparameters
        )


        progress_label = tk.Label(training_window, text="Training Progress:")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(training_window, orient="horizontal", length=400, mode="determinate")
        progress_bar.pack(pady=10)
        progress_bar["value"] = 0

        # Create a frame to hold both the metrics and loss
        metrics_loss_frame = tk.Frame(training_window)
        metrics_loss_frame.pack(pady=10)

        metrics_loss_label = tk.Label(metrics_loss_frame, text="Epoch 0/0 - Loss: 0.0000", font=("Courier", 10))
        metrics_loss_label.pack(side="left")

        def update_progress(epoch, total_epochs, loss, metrics):
            """
            Update the progress bar and labels in a Tkinter window.
            """
            progress = (epoch / total_epochs) * 100
            progress_bar["value"] = progress
            metrics_loss_label.config(
                text=f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f} | " + 
                    " | ".join([f"{metric}: {value:.4f}" for metric, value in metrics.items()])
            )
            training_window.update_idletasks()

        train_object.set_progress_callback(update_progress)

        # Start the training in a separate thread
        def train_thread():
            train_object.load_segmentation_data()
            train_object.train()
            messagebox.showinfo("Success", "Training completed successfully!")

        threading.Thread(target=train_thread, daemon=True).start()


    tk.Button(training_window, text="Select Data Folder", command=browse_data_dir).pack(pady=5)
    tk.Entry(training_window, textvariable=data_dir_var, width=50).pack()

    tk.Button(training_window, text="Select Run Folder", command=browse_run_dir).pack(pady=5)
    tk.Entry(training_window, textvariable=run_dir_var, width=50).pack()

    tk.Button(training_window, text="Select Hyperparameter File", command=browse_hyperparameters).pack(pady=5)
    tk.Entry(training_window, textvariable=hyperparameters_var, width=50).pack()

    tk.Button(training_window, text="Start Training", command=start_training, bg="green", fg="white").pack(pady=20)

def option_inference():
    """Handle the inference option."""
    inference_window = tk.Toplevel()
    inference_window.title("SCHISM - Inference")

    create_ascii_frame(inference_window).pack()

    label = tk.Label(inference_window, text="Select directories and files for inference:")
    label.pack(pady=10)

    data_dir_var = tk.StringVar()
    run_dir_var = tk.StringVar()

    def browse_data_dir():
        data_dir = filedialog.askdirectory(title="Select the data to be predicted")
        data_dir_var.set(data_dir)

    def browse_run_dir():
        run_dir = filedialog.askdirectory(title="Select your weight directory")
        run_dir_var.set(run_dir)

    def start_inference():
        data_dir = data_dir_var.get()
        run_dir = run_dir_var.get()

        if not (data_dir and run_dir):
            messagebox.showerror("Error", "Please select all required directories.")
            return

        hyperparameters = Hyperparameters(os.path.join(run_dir, 'hyperparameters.ini'))
        params = {k: v for k, v in hyperparameters.get_parameters()['Training'].items()}
        metrics = [metric.strip() for metric in params.get('metrics', 'Jaccard').split(',') if metric.strip()]

        if not metrics:
            messagebox.showerror("Error", "No metrics found.")
            return

        # Create a window to select the metric using radio buttons
        metric_window = tk.Toplevel()
        metric_window.title("Select Metric")

        selected_metric = tk.StringVar(value=metrics[0])  # Default to the first metric

        label = tk.Label(metric_window, text="Select a metric:")
        label.pack(pady=10)

        # Create radio buttons for each metric
        for metric in metrics:
            tk.Radiobutton(
                metric_window, text=metric, variable=selected_metric, value=metric
            ).pack(anchor="w", padx=20)

        def confirm_selection():
            nonlocal selected_metric
            if selected_metric.get():
                metric_window.destroy()
            else:
                messagebox.showerror("Error", "No metric selected.")

        # Button to confirm the selection
        tk.Button(metric_window, text="Confirm", command=confirm_selection).pack(pady=10)

        # Wait for the metric window to close before continuing
        metric_window.wait_window()

        # After window is closed, use the selected metric
        selected_metric_value = selected_metric.get()
        if not selected_metric_value:
            messagebox.showerror("Error", "No metric selected.")
            return

        subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        pred_object = Inference(
            data_dir=data_dir,
            subfolders=subfolders,
            run_dir=run_dir,
            selected_metric=selected_metric_value,
            hyperparameters=hyperparameters
        )

        # Add progress bar to the inference window
        progress_label = tk.Label(inference_window, text="Inference Progress:")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(inference_window, orient="horizontal", length=400, mode="determinate")
        progress_bar.pack(pady=10)
        progress_bar["value"] = 0

        def update_progress(current, total):
            """Update the progress bar during inference."""
            progress = (current / total) * 100
            progress_bar["value"] = progress
            inference_window.update_idletasks()

        # Start the inference process, passing the progress callback
        pred_object.predict(progress_callback=update_progress)

        messagebox.showinfo("Success", "Inference completed successfully!")



    tk.Button(inference_window, text="Select Data Folder", command=browse_data_dir).pack(pady=5)
    tk.Entry(inference_window, textvariable=data_dir_var, width=50).pack()

    tk.Button(inference_window, text="Select Run Folder", command=browse_run_dir).pack(pady=5)
    tk.Entry(inference_window, textvariable=run_dir_var, width=50).pack()

    tk.Button(inference_window, text="Start Inference", command=start_inference, bg="blue", fg="white").pack(pady=20)


def main():
    """Main function to run the application."""
    root = tk.Tk()
    root.title("SCHISM - Main Menu")

    create_ascii_frame(root).pack()

    label = tk.Label(root, text="Welcome to SCHISM! Please select an action:")
    label.pack(pady=10)

    tk.Button(root, text="Training", command=option_training, width=20, bg="green", fg="white").pack(pady=5)
    tk.Button(root, text="Inference", command=option_inference, width=20, bg="blue", fg="white").pack(pady=5)
    tk.Button(root, text="Quit", command=root.quit, width=20, bg="red", fg="white").pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
