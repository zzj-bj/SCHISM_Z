# -*- coding: utf-8 -*-
"""
A module for training a segmentation model using PyTorch.

This module provides a `Training` class that handles the initialization of the model,
data loading, optimizer and scheduler setup, loss function initialization,
and the main training loop.
"""

# Standard library
import os
import sys
import glob
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Third-party
import numpy as np
import torch
import torch.nn.functional as nn_func
import torch.backends.cudnn as cudnn
from torch import nn
from torch.amp import GradScaler
from torch.nn import (
    CrossEntropyLoss, BCEWithLogitsLoss, NLLLoss
)
from torch.optim import (
    Adagrad, Adam, AdamW, NAdam, RMSprop, RAdam, SGD
)
from torch.optim.lr_scheduler import (
    LRScheduler, LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR,
    LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, SequentialLR,
    ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import (
    BinaryJaccardIndex, MulticlassJaccardIndex, MulticlassF1Score, BinaryF1Score,
    BinaryAccuracy, MulticlassAccuracy, BinaryAveragePrecision, MulticlassAveragePrecision,
    BinaryConfusionMatrix, MulticlassConfusionMatrix, BinaryPrecision, MulticlassPrecision,
    BinaryRecall, MulticlassRecall
)
from early_stopping_pytorch import EarlyStopping

# Local application imports
import tools.utils as ut
from tools import display_color as dc
from tools.constants import DISPLAY_COLORS as colors
import tools.constants as ct
from preprocessing import launch_preprocessing as lp
from AI.tiffdatasetloader import TiffDatasetLoader, TiffDatasetLoaderConfig
from tools.paramconverter import ParamConverter
from AI.model_registry import model_mapping, model_config_mapping
from training.training_logger import TrainingLogger, TrainingLoggerConfig


# Ensure project root is on the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Training:
    """
        A module for training a segmentation model using PyTorch.
    """

    def __repr__(self) -> str:
        """
        Returns a string representation of the Training class.

        Returns:
            str: A string indicating the class name.
        """
        return 'Training'

    def __init__(self, **kwargs: Any) -> None:
        self.param_converter = ParamConverter()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.subfolders = kwargs.get('subfolders')
        if not self.subfolders or not isinstance(self.subfolders, list):
            raise ValueError("The 'subfolders' argument must be provided as a non-empty list.")
        self.data_dir = kwargs.get('data_dir')
        self.run_dir = kwargs.get('run_dir')
        self.hyperparameters = kwargs.get('hyperparameters')
        if self.hyperparameters is None:
            raise ValueError("The 'hyperparameters' argument must be provided and not None.")
        self.dataloaders = {}  # Initialize dataloaders attribute
        self.display = dc.DisplayColor()

        self.optimizer_mapping = {
            'Adagrad': Adagrad,
            'Adam': Adam,
            'AdamW': AdamW,
            'NAdam': NAdam,
            'RMSprop': RMSprop,
            'RAdam': RAdam,
            'SGD': SGD
        }

        self.loss_mapping = {
            'CrossEntropyLoss': CrossEntropyLoss,
            'BCEWithLogitsLoss': BCEWithLogitsLoss,
            'NLLLoss': NLLLoss,
        }

        self.scheduler_mapping = {
            'LRScheduler': LRScheduler,
            'LambdaLR': LambdaLR,
            'MultiplicativeLR': MultiplicativeLR,
            'StepLR': StepLR,
            'MultiStepLR': MultiStepLR,
            'ConstantLR': ConstantLR,
            'LinearLR': LinearLR,
            'ExponentialLR': ExponentialLR,
            'PolynomialLR': PolynomialLR,
            'CosineAnnealingLR': CosineAnnealingLR,
            'SequentialLR': SequentialLR,
            'ReduceLROnPlateau': ReduceLROnPlateau,
            'CyclicLR': CyclicLR,
            'OneCycleLR': OneCycleLR,
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts
        }
        
        # Model parameters
        self.model_params = {k: v for k, v in self.hyperparameters.get_parameters()['Model'].items()}
        self.num_classes = self.param_converter._convert_param(self.model_params.get('num_classes', 3))
        self.num_classes = 1 if self.num_classes <= 2 else self.num_classes
        self.model_mapping = model_mapping
        self.model_config_mapping = model_config_mapping
        self.model = self.initialize_model()

        # Optimizer parameters
        self.optimizer_params = {k: v for k, v in self.hyperparameters.get_parameters()['Optimizer'].items()}

        # Scheduler parameters
        self.scheduler_params = {k: v for k, v in self.hyperparameters.get_parameters()['Scheduler'].items()}

        # Loss parameters
        self.loss_params = {k: v for k, v in self.hyperparameters.get_parameters()['Loss'].items()}
        self.weights = self.param_converter._convert_param(self.loss_params.get('weights', "False"))
        self.ignore_background = self.param_converter._convert_param(self.loss_params.get('ignore_background', "False"))
        if self.ignore_background:
            self.ignore_index = -1
        else:
            self.ignore_index = -100

        # Training parameters
        self.training_params = {k: v for k, v in self.hyperparameters.get_parameters()['Training'].items()}
        self.batch_size = self.param_converter._convert_param(self.training_params.get('batch_size', 8))
        self.val_split = self.param_converter._convert_param(self.training_params.get('val_split', 0.8))
        self.epochs = self.param_converter._convert_param(self.training_params.get('epochs', 10))
        self.early_stopping = self.param_converter._convert_param(self.training_params.get('early_stopping', "False"))
        self.metrics_str = self.param_converter._convert_param(self.training_params.get('metrics', ''))
        if self.early_stopping:
            patience = int(self.epochs*0.2)
            if patience > 1:
                self.early_stopping_instance = EarlyStopping(patience=patience, verbose=True, path=Path(self.run_dir))
            else:
                self.early_stopping=False
                display = dc.DisplayColor()
                display.print("Early stopping has been automatically disabled because the patience value is too low.", colors['warning'])
                display.print("Training will begin as normal.", colors['warning'])       

        # Data parameters
        self.data = {k: v for k, v in self.hyperparameters.get_parameters()['Data'].items()}
        self.img_res = self.param_converter._convert_param(self.data.get('img_res', 560))
        self.crop_size = self.param_converter._convert_param(self.data.get('crop_size', 224))
        self.num_samples = self.param_converter._convert_param(self.data.get('num_samples', 500))

        # Extract and parse metrics from the ini file
        self.training_time = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
        
        self.save_directory = self.create_unique_folder()
        self.logger = TrainingLogger(
            TrainingLoggerConfig(save_directory=self.save_directory,
                                     num_classes=self.num_classes,
                                     model_params=self.model_params,
                                     optimizer_params=self.optimizer_params,
                                     scheduler_params=self.scheduler_params,
                                     loss_params=self.loss_params,
                                     training_params=self.training_params,
                                     data=self.data)
            )

        self.metrics = []  # Initialize metrics attribute
        self.metrics_mapping = {}  # Initialize metrics_mapping attribute

    def create_metric(self,
        binary_metric: Any,
        multiclass_metric: Any
    ) -> Any:
        """
        Creates a metric based on the number of classes.

        Args:
            binary_metric: The binary metric to use.
            multiclass_metric: The multiclass metric to use.

        Returns:
            Instance of the appropriate metric.
        """
        return (
            binary_metric(ignore_index=self.ignore_index).to(self.device)
            if self.num_classes == 1
            else multiclass_metric(num_classes=self.num_classes,
                                   ignore_index=self.ignore_index).to(self.device)
        )


    def initialize_metrics(self) -> List[Any]:
        """
        Initializes the specified metrics for evaluation.

        Returns:
            list: A list of metric instances corresponding to the specified names.

        Raises:
            ValueError: If a specified metric is not recognized.
        """
        self.metrics_mapping = {
            "Jaccard": self.create_metric(BinaryJaccardIndex, MulticlassJaccardIndex),
            "F1": self.create_metric(BinaryF1Score, MulticlassF1Score),
            "Accuracy": self.create_metric(BinaryAccuracy, MulticlassAccuracy),
            "AveragePrecision": self.create_metric(BinaryAveragePrecision,
                                                   MulticlassAveragePrecision),
            "ConfusionMatrix": self.create_metric(BinaryConfusionMatrix, MulticlassConfusionMatrix),
            "Precision": self.create_metric(BinaryPrecision, MulticlassPrecision),
            "Recall": self.create_metric(BinaryRecall, MulticlassRecall),
        }

        # Parse metrics from string input or use default
        self.metrics = [metric.strip()
                        for metric
                        in self.metrics_str.split(',')] if self.metrics_str else ["Jaccard"]

        # Retrieve metric instances
        selected_metrics = []
        for metric in self.metrics:
            if metric in self.metrics_mapping:
                selected_metrics.append(self.metrics_mapping[metric])
            else:
                text =f" - Metric '{metric}' not recognized"
                raise ValueError(text)

        return selected_metrics

    def initialize_optimizer(self) -> torch.optim.Optimizer:
        """
        Initializes the optimizer based on the specified parameters.

        Returns:
            Optimizer configured.
        """
        optimizer_name = self.optimizer_params.get('optimizer', 'Adam')
        optimizer_class = self.optimizer_mapping.get(optimizer_name)

        if not optimizer_class:
            text =f" - Optimizer '{optimizer_name}' is not supported"
            raise ValueError(text)


        converted_params = {k: self.param_converter._convert_param(v)
                            for k, v in self.optimizer_params.items() if k != 'optimizer'}

        return optimizer_class(self.model.parameters(), **converted_params)

    def initialize_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """
        Initializes the learning rate scheduler.

        Args:
            optimizer: The optimizer to use with the scheduler.

        Returns:
            Learning rate scheduler configured.
        """
        scheduler_name = self.scheduler_params.get('scheduler', 'ConstantLR')
        scheduler_class = self.scheduler_mapping.get(scheduler_name)

        if not scheduler_class:
            text =f" - Scheduler '{scheduler_name}' is not supported"
            raise ValueError(text)

        converted_params = {k: self.param_converter._convert_param(v)
                            for k, v in self.scheduler_params.items() if k != 'scheduler'}

        if not converted_params:
            return scheduler_class(optimizer)

        return scheduler_class(optimizer, **converted_params)

    def initialize_loss(self, **dynamic_params: Any) -> Any:
        """
        Initializes the loss function based on the specified parameters.

        Args:
            **dynamic_params: Dynamic parameters to use for the loss function.

        Returns:
            Loss function configured.
        """
        loss_name = self.loss_params.get('loss', 'CrossEntropyLoss')
        loss_class = self.loss_mapping.get(loss_name)

        if not loss_class:
            text = f" - Loss '{loss_name}' is not supported"
            raise ValueError(
                f"Loss '{loss_name}' is not supported. Check your 'loss_mapping'.")

        # Convert static parameters from config
        converted_params = {
            k: self.param_converter._convert_param(v)
            for k, v in self.loss_params.items()
            # Exclude unwanted params
            if k not in {'loss', 'ignore_background', 'weights'}
        }

        # Merge with dynamic parameters (e.g., batch-specific weights)
        final_params = {**converted_params, **dynamic_params}

        # Check if ignore_index should be included (for all losses except BCEWithLogitsLoss)
        if loss_name == 'BCEWithLogitsLoss':
            # Remove ignore_index if not needed
            final_params.pop('ignore_index', None)
        else:
            if self.num_classes > 1:
                final_params['ignore_index'] = self.ignore_index
            else:
                # Remove ignore_index if not needed
                final_params.pop('ignore_index', None)

        return loss_class(**final_params)

    def initialize_model(self) -> torch.nn.Module:
        """
        Initializes the model based on the specified parameters.

        Returns:
            nn.Module: Instance of the configured model.
        """
        model_name = self.model_params.get('model_type', 'UnetVanilla')

        if model_name not in self.model_mapping:
            text = f" - Model '{model_name}' is not supported"
            raise ValueError(
                f"Model '{model_name}' is not supported. Check your 'model_mapping'.")

        model_class = self.model_mapping[model_name]
        model_config_class = self.model_config_mapping[model_name]

        self.model_params['num_classes'] = self.num_classes
        
        if model_name == 'DINOv2':
            self.model_params['img_res'] = self.img_res

        required_params = {
            k: self.param_converter._convert_param(v)
            for k, v in self.model_params.items() if k in model_class.REQUIRED_PARAMS
        }
        optional_params = {
            k: self.param_converter._convert_param(v)
            for k, v in self.model_params.items() if k in model_class.OPTIONAL_PARAMS
       }

        required_params.pop('model_type', None)
        optional_params.pop('model_type', None)

        try:
            typed_required_params = {
                k: model_class.REQUIRED_PARAMS[k](v)
                for k, v in required_params.items()
            }

            typed_optional_params = {
                k: model_class.OPTIONAL_PARAMS[k](v)
                for k, v in optional_params.items()
            }
        except ValueError as e:
            text =f" - Error converting parameters for model '{model_name}' : {e}"
            raise ValueError(e)

        try:
            return model_class(
                model_config_class(**typed_required_params, **typed_optional_params)
            ).to(self.device)
        except RuntimeError as e:
            dc.DisplayColor().print(
                f"CUDA error during model initialization: {e}. Falling back to CPU.",
                colors['warning']
            )
            self.device = 'cpu'
            return model_class(
                model_config_class(**typed_required_params, **typed_optional_params)
            ).to(self.device)

    def create_unique_folder(self) -> str:
        """
        Creates a unique folder for saving model weights
        and logs based on the current training parameters.

        Returns:
            str: The path to the created directory.
        """
        filename = f"{self.model_params.get('model_type', 'UnetVanilla')}__" \
            f"{self.training_time}"

        save_directory = os.path.join(str(self.run_dir),
                                      str(filename))

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        return save_directory

    def load_segmentation_data(self) -> None:
        """
        Loads segmentation data from the specified directories,
        prepares datasets, and creates data loaders.

        This method also handles the loading of normalization statistics
        and the splitting of data into training,
        validation, and test sets.
        """

        def generate_random_indices(num_samples, val_split, subfolders, num_sample_subfolder):
            """
            Generates random indices for splitting the dataset into training, validation,
            and test sets.

            Args:
                num_samples (int): Total number of samples in the dataset.
                val_split (float): Proportion of the dataset to use for validation.
                subfolders (list): List of subfolder names containing the dataset.
                num_sample_subfolder (dict): Dictionary mapping subfolder names to
                                            the number of samples in each.

            Returns:
                tuple: Three lists containing the indices for training, validation, and test sets.
            """
            num_train = int(num_samples * val_split)
            num_val = num_samples - num_train
            all_indices = [
                (sub_folder_temp, sample_id)
                for sub_folder_temp in subfolders
                for sample_id in range(num_sample_subfolder[sub_folder_temp])
            ]
            np.random.shuffle(all_indices)
            val_indices_temp = np.random.choice(
                len(all_indices), size=num_val, replace=False)
            val_set = set(val_indices_temp)
            train_indices_temp = [
                idx for idx in range(len(all_indices)) if idx not in val_set
            ][:num_train]
            train_set = set(train_indices_temp)
            all_indices_set = set(range(len(all_indices)))
            test_indices_temp = list(all_indices_set - train_set - val_set)
            train_indices_temp = [all_indices[i] for i in train_indices_temp]
            val_indices_temp = [all_indices[i] for i in val_indices_temp]
            test_indices_temp = [all_indices[i] for i in test_indices_temp]
            return train_indices_temp, val_indices_temp, test_indices_temp

        img_data = {}
        mask_data = {}
        num_sample_per_subfolder = {}
        data_stats = ut.load_data_stats(self.data_dir, self.data_dir)
        
        if not self.subfolders or not isinstance(self.subfolders, list):
            text = "The 'subfolders' attribute must be a non-empty list before loading data."
            raise ValueError(text)

        for subfolder in self.subfolders:
            img_folder = os.path.join(
                str(self.data_dir),
                str(subfolder),
                "images")
            mask_folder = os.path.join(
                str(self.data_dir),
                str(subfolder),
                "masks")
            img_files = sorted(glob.glob(os.path.join(img_folder, "*")))
            mask_files = sorted(glob.glob(os.path.join(mask_folder, "*")))
            assert len(img_files) == len(mask_files), (
                f"Mismatch: {len(img_files)} images, {len(mask_files)} masks in {subfolder}"
            )
            img_data[subfolder] = img_files
            mask_data[subfolder] = mask_files
            num_sample_per_subfolder[subfolder] = len(img_files)

        train_indices, val_indices, test_indices = generate_random_indices(
            num_samples=self.num_samples,
            val_split=self.val_split,
            subfolders=self.subfolders,
            num_sample_subfolder=num_sample_per_subfolder,
        )

        indices = [train_indices, val_indices, test_indices]

        train_dataset = TiffDatasetLoader(
            TiffDatasetLoaderConfig(
                indices=train_indices,
                img_data=img_data,
                mask_data=mask_data,
                num_classes=self.num_classes,
                crop_size=(self.crop_size, self.crop_size),
                data_stats=data_stats,
                img_res=self.img_res,
                ignore_background=self.ignore_background,
                weights=self.weights
            )
        )

        val_dataset = TiffDatasetLoader(
            TiffDatasetLoaderConfig(
                indices=val_indices,
                img_data=img_data,
                mask_data=mask_data,
                num_classes=self.num_classes,
                crop_size=(self.crop_size, self.crop_size),
                data_stats=data_stats,
                img_res=self.img_res,
                ignore_background=self.ignore_background,
                weights=self.weights
            )
        )

        test_dataset = TiffDatasetLoader(
            TiffDatasetLoaderConfig(
                indices=test_indices,
                img_data=img_data,
                mask_data=mask_data,
                num_classes=self.num_classes,
                crop_size=(self.crop_size, self.crop_size),
                data_stats=data_stats,
                img_res=self.img_res,
                ignore_background=self.ignore_background,
                weights=self.weights
            )
        )

        try:
            pin_mem = torch.cuda.is_available()
        except Exception:
            pin_mem = False

        train_loader = DataLoader(train_dataset, 
                                batch_size=self.batch_size,
                                num_workers = ct.NUM_WORKERS, 
                                shuffle = True, 
                                drop_last = True,
                                pin_memory = pin_mem)
        val_loader =  DataLoader(val_dataset, 
                                batch_size = self.batch_size,
                                shuffle = False, 
                                num_workers= ct.NUM_WORKERS, 
                                drop_last = True,
                                pin_memory = pin_mem)
        test_loader =  DataLoader(test_dataset, 
                                batch_size=1, 
                                shuffle = False,
                                num_workers = 2, 
                                drop_last = True,
                                pin_memory = pin_mem)

        self.logger.save_indices_to_file(
            [train_indices, val_indices, test_indices])

        self.dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'indices': indices,
        }

    def training_loop(self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[str, List[float]]], List[Any]]:
        """
        Main training loop that handles the training and validation of the model.

        Args:
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use.

        Returns:
            tuple: Dictionaries containing losses and metrics for each phase.
        """
        scaler = None
        # Choose whether to use the gradient scaler
        if self.device.startswith("cuda"):
            scaler = GradScaler()              # automatically uses CUDA
            cudnn.benchmark = True
        else:
            # disable scaling on CPU
            scaler = GradScaler(enabled=False)
        
        # Initialize metric instances and losses
        # This list includes your ConfusionMatrix instance if enabled
        metrics = self.initialize_metrics()
        loss_dict = {"train": {}, "val": {}}

        # Build a list of display metric names (excluding "ConfusionMatrix")
        display_metrics = [m for m in self.metrics if m != "ConfusionMatrix"]
        metrics_dict = {
            phase: {metric: [] for metric in display_metrics}
            for phase in ["train", "val"]
        }
        best_val_loss = float("inf")
        best_val_metrics = {metric: 0.0 for metric in display_metrics}

        for epoch in range(1, self.epochs + 1):

            ut.print_box(f"Epoch {epoch}/{self.epochs}")
            epoch_val_loss = None if self.early_stopping else None

            for phase in ["train", "val"]:
                is_training = phase == "train"

                if is_training:
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_metrics = {metric: 0.0 for metric in display_metrics}
                total_samples = 0

                with tqdm(
                    total=len(self.dataloaders[phase]),
                    desc=f"{phase.capitalize()}",
                    unit="batch",
                    position=0,
                    leave=False,
                    ncols=70,              
                    dynamic_ncols=False, 
                ) as pbar, tqdm(
                    total=0,
                    desc="",
                    position=1,
                    bar_format="{desc}",
                    leave=False,
                    ncols=70,
                    dynamic_ncols=False
                ) as mbar:

                    for inputs, labels, weights in self.dataloaders[phase]:

                        inputs, labels, weights = (
                            inputs.to(self.device),
                            labels.to(self.device),
                            weights.to(self.device)
                        )
                        optimizer.zero_grad()
                        batch_weights = torch.mean(weights, dim=0)
                        batch_weights = torch.clamp(batch_weights, min=1e-6)  # Avoid exact zero values

                        with torch.set_grad_enabled(is_training):
                            with torch.autocast(
                                    device_type=self.device,
                                    dtype=torch.bfloat16
                                ):
                                outputs = self.model.forward(inputs)
                                if self.loss_params.get('loss') in ['NLLLoss']:
                                    outputs = nn_func.log_softmax(
                                        outputs, dim=1)

                                if self.num_classes == 1:
                                    outputs = outputs.squeeze()
                                    labels = labels.squeeze().float()
                                else:
                                    labels = labels.squeeze().long()

                                # only apply class weights to multiclass segmentation
                                loss_fn = self.initialize_loss(weight=batch_weights if (self.weights and self.num_classes > 1) else None)
                                loss = loss_fn(outputs.float(), labels)

                            if is_training:
                                if scaler:
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    loss.backward()
                                    optimizer.step()

                                if isinstance(
                                        scheduler,
                                        torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    scheduler.step(loss)
                                else:
                                    scheduler.step()

                        running_loss += loss.item()
                        total_samples += labels.size(0)

                        with torch.no_grad():
                            preds = (torch.argmax(outputs, dim=1).long()
                                     if self.num_classes > 1
                                     else (outputs > 0.5).to(torch.uint8))
                            labels = labels.long()

                            # Update display metrics only (skip ConfusionMatrix)
                            for name, fn in zip(self.metrics, metrics):
                                if name != "ConfusionMatrix":
                                    running_metrics[name] += fn(preds, labels).item()

                        pbar.update(1)

                        # build exactly the string you want, with pipes and colons
                        avg_loss = running_loss / pbar.n
                        metrics_str = (
                            f"{phase.capitalize()} Loss: {avg_loss:.4f} | "
                            + " | ".join(
                                f"{m}: {running_metrics[m]/pbar.n:.4f}"
                                for m in display_metrics
                            )
                            + " |"
                        )
                        # overwrite the empty desc of the second bar
                        mbar.set_description(metrics_str)
                        
                epoch_loss = running_loss / len(self.dataloaders[phase])
                epoch_metrics = {metric: running_metrics[metric] /
                                 len(self.dataloaders[phase])
                                 for metric in display_metrics}
                loss_dict[phase][epoch] = epoch_loss

                for metric, value in epoch_metrics.items():
                    metrics_dict[phase][metric].append(value)

                print(f" {phase.title().ljust(5)} Loss: {epoch_loss: .4f}")
                print(f" {phase.title().ljust(5)} ", end='')

                for metric, value in epoch_metrics.items():
                    print(f" {metric}: {value: .4f}", end=" | ")
                print()

                if phase == "val" and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(self.model.state_dict(),
                               os.path.join(self.save_directory,
                               "model_best_loss.pth"))

                if phase == "val":
                    if self.early_stopping:
                        epoch_val_loss = epoch_loss
                    for metric, value in epoch_metrics.items():
                        if value > best_val_metrics[metric]:
                            best_val_metrics[metric] = value
                            torch.save(
                                self.model.state_dict(),
                                os.path.join(self.save_directory,
                                             f"model_best_{metric}.pth")
                            )

            if self.early_stopping and epoch_val_loss is not None:
                self.early_stopping_instance(epoch_val_loss, self.model)
                if self.early_stopping_instance.early_stop:
                    print("Early stopping triggered")
                    break

        formatted_metrics = {metric: f"{value:.4f}" for metric, value in best_val_metrics.items()}
        
        print("Best validation metrics:")
        for metric, value in formatted_metrics.items():
            print(f"  {metric:<10}: {value}")

        return loss_dict, metrics_dict, metrics

    def train(self) -> None:
        """
        Starts the training process for the model.
        """

        optimizer = self.initialize_optimizer()
        scheduler = self.initialize_scheduler(optimizer=optimizer)
        loss_dict, metrics_dict, metrics = self.training_loop(optimizer=optimizer,
                                                              scheduler=scheduler)

        self.logger.save_best_metrics(loss_dict=loss_dict,
                                      metrics_dict=metrics_dict)
        self.logger.plot_learning_curves(loss_dict=loss_dict,
                                         metrics_dict=metrics_dict)
        self.logger.save_hyperparameters()
        if ct.AUGMENTED_HYPERPARAMETERS:
            self.logger.save_hyperparameters(loss_dict=loss_dict, metrics_dict=metrics_dict)
        self.logger.save_data_stats(
            self.dataloaders["train"].dataset.data_stats)
        if "ConfusionMatrix" in self.metrics:
            self.logger.save_confusion_matrix(
                conf_metric=metrics[self.metrics.index("ConfusionMatrix")],
                model=self.model,
                val_dataloader=self.dataloaders["val"],
                device=self.device)
