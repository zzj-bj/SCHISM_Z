# -*- coding: utf-8 -*-
"""
A module for logging training metrics, saving hyperparameters, and visualizing results.
This module provides a `TrainingLogger` class that handles the logging of training metrics,
saving hyperparameters to an INI file, 
and visualizing results such as learning curves and confusion matrices.
"""

# Standard library
from dataclasses import dataclass
import os
import json
import configparser
from typing import Any, Dict, List, Tuple

# Third-party libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tools.constants as ct
from tools.constants import DISPLAY_COLORS as colors
from tools import display_color as dc

@dataclass
# pylint: disable=too-many-instance-attributes
class TrainingLoggerConfig:
    """Configuration for the TrainingLogger."""
    save_directory: str
    num_classes: int
    model_params: Dict[str, Any]
    optimizer_params: Dict[str, Any]
    scheduler_params: Dict[str, Any]
    loss_params: Dict[str, Any]
    training_params: Dict[str, Any]
    data: Dict[str, Any]

class TrainingLogger:
    """    
    A class to log training metrics, save hyperparameters, and visualize results.
    """


    def __init__(self, training_logger_config: TrainingLoggerConfig) -> None:
        self.save_directory = training_logger_config.save_directory
        self.num_classes = training_logger_config.num_classes
        self.model_params = training_logger_config.model_params
        self.optimizer_params = training_logger_config.optimizer_params
        self.scheduler_params = training_logger_config.scheduler_params
        self.loss_params = training_logger_config.loss_params
        self.training_params = training_logger_config.training_params
        self.data = training_logger_config.data
        self.display = dc.DisplayColor()
        os.makedirs(self.save_directory, exist_ok=True)

    def save_indices_to_file(self, indices_list: List[List[Tuple[str, int]]]) -> None:
        """
        Saves the indices of the training, validation, and test sets to text files.

        Args:
            indices_list (list): A list containing the indices for training,
                                 validation, and test sets.
        """
        indices_map = {
            "train": indices_list[0],
            "val": indices_list[1],
            "test": indices_list[2],
        }
        for idx_type, idx_list in indices_map.items():
            file_path = os.path.join(self.save_directory, f'{idx_type}_indices.txt')
            with open(file_path, 'w', encoding="utf-8") as f:
                for subfolder_name, sample_idx in idx_list:
                    f.write(f"{subfolder_name}, {sample_idx}\n")


    def save_data_stats(self, data_stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Saves dataset statistics to a JSON file.

        Args:
            data_stats (dict): A dictionary containing dataset statistics.
        """
        data_stats_serializable = {
            key: [value[0].tolist(), value[1].tolist()]
            for key, value in data_stats.items()
            if key != "default" or len(data_stats) == 1
        }

        json_file_path = os.path.join(self.save_directory, 'data_stats.json')
        with open(json_file_path, 'w', encoding="utf-8") as json_file:
            json.dump(data_stats_serializable, json_file, indent=4)

        self.display.print(f"Data statistics saved to {json_file_path}", colors['ok'])

    def save_hyperparameters(
            self, 
            loss_dict: Dict[str, Dict[int, float]] = None, 
            metrics_dict: Dict[str, Dict[str, List[float]]] = None
            ) -> None:
        """
        Saves hyperparameters to an INI file.
        """

        def save_ini_file(self, config, sections, colors, title):
            """Helper function to write sections to an INI file and display a message."""
            for section, params in sections.items():
                if not config.has_section(section):
                    config.add_section(section)
                for key, value in params.items():
                    config.set(section, key, str(value))

            file_path = os.path.join(self.save_directory, f"{title}.ini")
            with open(file_path, 'w', encoding='utf-8') as f:
                config.write(f)

            self.display.print(f" {title} saved to {file_path}", colors['ok'])
            return file_path

        config = configparser.ConfigParser()

        sections = {
            'Model': self.model_params,
            'Optimizer': self.optimizer_params,
            'Scheduler': self.scheduler_params,
            'Loss': self.loss_params,
            'Training': self.training_params,
            'Data': self.data,
        }

        save_ini_file(self, config, sections, colors, 'hyperparameters')

        if ct.AUGMENTED_HYPERPARAMETERS and loss_dict and metrics_dict:
            results = {
                'Results': {
                    **{f'Best_{metric}': f"{max(values):.4f}" for metric, values in metrics_dict['val'].items()},
                    'Lowest_Loss': f"{min(loss_dict['val'].values()):.4f}"
                }
            }

            save_ini_file(self, config, results, colors, 'augmented_hyperparameters')

    def save_best_metrics(self,
        loss_dict: Dict[str, Dict[int, float]],
        metrics_dict: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """
        Saves validation loss and metrics history.

        Args:
            loss_dict (dict): Dictionary of loss values.
            metrics_dict (dict): Dictionary of metric values.
        """
        file_path = os.path.join(self.save_directory, "val_metrics_history.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Validation Metrics History\n")
            f.write("=" * 30 + "\n\n")

            for epoch in sorted(loss_dict['val'].keys()):
                f.write(f"Epoch {epoch}:\n")
                f.write(f"  - Loss: {loss_dict['val'][epoch]:.4f}\n")
                for metric, values in metrics_dict['val'].items():
                    f.write(f"  - {metric}: {values[epoch - 1]:.4f}\n")
                f.write("\n" + "-" * 30 + "\n\n")

        self.display.print(f"Validation metrics history saved to {file_path}", colors['ok'])

    def plot_learning_curves(self,
        loss_dict: Dict[str, Dict[int, float]],
        metrics_dict: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """
        Plots learning curves for loss and metrics over epochs.

        Args:
            loss_dict (dict): Dictionary of loss values.
            metrics_dict (dict): Dictionary of metric values.
        """
        epochs = list(loss_dict['train'].keys())
        train_loss_values = [loss_dict['train'][epoch] for epoch in epochs]
        val_loss_values = [loss_dict['val'][epoch] for epoch in epochs]

        _, axes = plt.subplots(1, 2, figsize=(15, 5))

        ax0 = axes[0]
        ax0.plot(epochs, train_loss_values, 'b-', label='Train Loss')
        ax0.plot(epochs, val_loss_values, 'r-', label='Val Loss')
        ax0.set_title('Loss')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('Loss Value')
        ax0.legend()
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True))  # labels entiers sur x

        ax1 = axes[1]
        for metric in metrics_dict['train']:
            train_metric_values = [metrics_dict['train'][metric][epoch - 1] for epoch in epochs]
            val_metric_values = [metrics_dict['val'][metric][epoch - 1] for epoch in epochs]
            ax1.plot(epochs, train_metric_values, label=f'Train {metric}')
            ax1.plot(epochs, val_metric_values, label=f'Val {metric}')

        ax1.set_title('Metrics')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metric Values')
        ax1.legend()
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True))  # labels entiers sur x

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_directory, 'learning_curves.png'), dpi=300)
        plt.close()
        self.display.print(f"Learning curves saved to {self.save_directory}\\learning_curves.png",
                            colors['ok'])

    # pylint: disable=too-many-locals
    def save_confusion_matrix(self,
        conf_metric: Any,
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> None:
        """
        Saves a confusion matrix plot.

        Args:
            conf_metric: Initialized confusion matrix metric instance.
            model: Trained model.
            val_dataloader: Validation dataset loader.
            device: Computation device.
        """
        model.eval()
        conf_metric.reset()

        final_preds, final_labels = [], []
        with torch.no_grad():
            for inputs, labels, _ in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if self.num_classes > 1:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (outputs > 0.5).to(torch.uint8)

                final_preds.append(preds.cpu())
                final_labels.append(labels.cpu())

        if final_preds and final_labels:
            preds_tensor = torch.cat(final_preds).to(device).long()
            labels_tensor = torch.cat(final_labels).to(device).long()

            if self.num_classes == 1:
                preds_tensor = preds_tensor.squeeze(1)
                labels_tensor = labels_tensor.squeeze(1)

            cm = conf_metric(preds_tensor, labels_tensor).cpu().numpy()
            cm_percent = (cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)) * 100

            plt.figure(figsize=(8, 6))
            plt.imshow(cm_percent, interpolation='nearest', cmap='Blues')
            plt.colorbar()

            tick_marks = [0, 1] if self.num_classes == 1 else list(
                range(self.num_classes))
            plt.xticks(tick_marks)
            plt.yticks(tick_marks)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            thresh = cm_percent.max() / 2.0
            for i in range(cm_percent.shape[0]):
                for j in range(cm_percent.shape[1]):
                    plt.text(j, i, f"{cm_percent[i, j]:.1f}%", horizontalalignment="center",
                             color="white" if cm_percent[i, j] > thresh else "black")

            plt.savefig(
                os.path.join(self.save_directory, "confusion_matrix.png"),
                dpi=300)
            plt.close()
            prompt = f"Confusion matrix saved to {self.save_directory}\\confusion_matrix.png"
            self.display.print(prompt, colors['ok'])
