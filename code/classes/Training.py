import sys
import os
import configparser
import torch
import torch.nn as nn
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from classes.TiffDatasetLoader import TiffDatasetLoader
from classes.UnetVanilla import UnetVanilla
from classes.UnetSegmentor import UnetSegmentor
from classes.Unet_liang import UNet_liang
from classes.dinov2 import DinoV2Segmentor
from datetime import datetime
from torch.optim import Adagrad, Adam, AdamW, NAdam, RMSprop, RAdam, SGD
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU#, DiceScore
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Training:

    def __repr__(self):
        """
        Returns a string representation of the Training class.

        Returns:
            str: A string indicating the class name.
        """
        return 'Training'

    def __init__(self, **kwargs):
        """
        Initializes the Training class with given parameters.

        Args:
            **kwargs: Keyword arguments containing configuration parameters such as:
                - subfolders (list): List of subfolder names containing the dataset.
                - data_dir (str): Directory containing the dataset.
                - run_dir (str): Directory for saving model outputs.
                - hyperparameters (Hyperparameters): An object containing model and training parameters.

        Raises:
            Exception: If pathLogDir is not provided.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.subfolders = kwargs.get('subfolders')
        self.data_dir = kwargs.get('data_dir')
        self.run_dir = kwargs.get('run_dir')
        self.hyperparameters = kwargs.get('hyperparameters')
       
        # Extract category-wise parameters
        self.model_params = {k: v for k, v in self.hyperparameters.get_parameters()['Model'].items()}
        self.optimizer_params = {k: v for k, v in self.hyperparameters.get_parameters()['Optimizer'].items()}
        self.scheduler_params = {k: v for k, v in self.hyperparameters.get_parameters()['Scheduler'].items()}
        self.training_params = {k: v for k, v in self.hyperparameters.get_parameters()['Training'].items()}
        self.data = {k: v for k, v in self.hyperparameters.get_parameters()['Data'].items()}

        self.num_classes = int(self.model_params.get('num_classes'))

        # Training parameters
        self.batch_size = int(self.training_params.get('batch_size', 8))
        self.val_split = float(self.training_params.get('val_split', 0.8))
        self.epochs = int(self.training_params.get('epochs', 10))

        # Data parameters
        self.img_res = int(self.data.get('img_res', 560))
        self.crop_size = int(self.data.get('crop_size', 224))
        self.num_samples = int(self.data.get('num_samples', 500))

        # Extract and parse metrics from the ini file
        self.metrics_str = self.training_params.get('metrics', '')        
        self.training_time = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
        self.num_classes = 1 if self.num_classes <= 2 else self.num_classes

        # Mapping of model names to classes
        self.model_mapping = {
            'UnetVanilla': UnetVanilla,
            'UnetSegmentor': UnetSegmentor,
            'liang': UNet_liang,
            'DINOv2' : DinoV2Segmentor
        }

        # Setup optimizer mapping
        self.optimizer_mapping = {
            'Adagrad' : Adagrad, 
            'Adam' : Adam, 
            'AdamW' : AdamW, 
            'NAdam' : NAdam, 
            'RMSprop' : RMSprop, 
            'RAdam' : RAdam, 
            'SGD' : SGD
        }

        # Setup learning rate scheduler mapping
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

        """self.metrics_mapping = {
            #"DiceScore": DiceScore(num_classes=self.num_classes).to(self.device),
            "GeneralizedDiceScore": GeneralizedDiceScore(num_classes=self.num_classes).to(self.device),
            #TODO : "HausdorffDistance": HausdorffDistance(num_classes=self.num_classes, input_format='index').to(self.device),
            "MeanIoU": MeanIoU(num_classes=self.num_classes, include_background=True).to(self.device),
        }"""

        self.metrics_mapping = {
            "MeanIoU": MulticlassJaccardIndex(num_classes=self.num_classes, ignore_index=-1).to(self.device),
            "GeneralizedDiceScore": GeneralizedDiceScore(num_classes=self.num_classes).to(self.device)
        }

        # Initialize the model dynamically
        self.model = self.initialize_model()
        self.save_directory = self.create_unique_folder()

    def initialize_metrics(self):
        """
        Retrieves the specified metrics for evaluation.

        Args:
            metrics_list (list): A list of metric names to retrieve.

        Returns:
            list: A list of metric instances corresponding to the specified names.

        Raises:
            ValueError: If a specified metric is not recognized.
        """
        if self.metrics_str:
            self.metrics = [metric.strip() for metric in self.metrics_str.split(',')]
        else:
            self.metrics = ["MeanIoU"]

        selected_metrics = []
        for metric in self.metrics:
            metric = metric.strip()
            if metric in self.metrics_mapping:
                selected_metrics.append(self.metrics_mapping[metric])
            else:
                raise ValueError(f"Metric '{metric}' not recognized. Please check the name.")

        return selected_metrics

    def initialize_losses(self):
        """
        Retrieves the appropriate loss function based on the model's output.

        Returns:
            callable: The loss function to be used during training.
        """
        is_binary = self.model.num_classes <= 2

        if not is_binary:
            loss = nn.CrossEntropyLoss
            return loss(ignore_index=-1)
        else:
            loss = nn.BCEWithLogitsLoss
            return loss()

    def initialize_optimizer(self):
        optimizer_name = self.optimizer_params.get('optimizer', 'Adam')
        optimizer_class = self.optimizer_mapping.get(optimizer_name)

        if not optimizer_class:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Check your 'optimizer_mapping'.")

        # Convert parameters to their appropriate types, excluding the 'optimizer' key
        converted_params = {}
        for k, v in self.optimizer_params.items():
            if k == 'optimizer':
                continue  # Skip the 'optimizer' key
            if isinstance(v, str):
                # Attempt to convert string representations of booleans and floats
                if v.lower() == 'true':
                    converted_params[k] = True
                elif v.lower() == 'false':
                    converted_params[k] = False
                else:
                    try:
                        converted_params[k] = float(v)
                    except ValueError:
                        raise ValueError(f"Could not convert parameter '{k}' with value '{v}' to a valid type.")
            else:
                # Use the value as is for non-string types
                converted_params[k] = v

        return optimizer_class(self.model.parameters(), **converted_params)

    def initialize_scheduler(self, optimizer):
        scheduler_name = self.scheduler_params.get('scheduler', 'ConstantLR')
        scheduler_class = self.scheduler_mapping.get(scheduler_name)

        if not scheduler_class:
            raise ValueError(f"Scheduler '{scheduler_name}' is not supported. Check your 'scheduler_mapping'.")

        # Convert parameters to their appropriate types, excluding the 'scheduler' key
        converted_params = {}
        for k, v in self.scheduler_params.items():
            if k == 'scheduler':
                continue  # Skip the 'scheduler' key
            if isinstance(v, str):
                # Strip quotes from strings if they are enclosed
                v = v.strip("'\"")
                # Handle None
                if v.lower() == 'none':
                    converted_params[k] = None
                # Attempt to convert booleans
                elif v.lower() == 'true':
                    converted_params[k] = True
                elif v.lower() == 'false':
                    converted_params[k] = False
                # Attempt to convert to a list of floats
                elif ',' in v:
                    try:
                        converted_params[k] = [float(x.strip()) for x in v.split(',')]
                    except ValueError:
                        raise ValueError(f"Could not convert parameter '{k}' with value '{v}' to a list of floats.")
                else:
                    # Attempt to convert to a float
                    try:
                        converted_params[k] = float(v)
                    except ValueError:
                        # If conversion fails, use as string (as for `mode` parameter)
                        converted_params[k] = v
            else:
                # Use the value as is for non-string types
                converted_params[k] = v

        if not converted_params:
            return scheduler_class(optimizer)
        else:
            return scheduler_class(optimizer, **converted_params)

    def initialize_model(self) -> nn.Module:
        model_name = self.model_params.get('model_type', 'UnetVanilla')

        if model_name not in self.model_mapping:
            raise ValueError(f"Model '{model_name}' is not supported. Check your 'model_mapping'.")

        model_class = self.model_mapping[model_name]

        required_params = {k: v for k, v in self.model_params.items() if k in model_class.REQUIRED_PARAMS}
        optional_params = {k: v for k, v in self.model_params.items() if k not in model_class.REQUIRED_PARAMS}

        required_params.pop('model_type', None)
        optional_params.pop('model_type', None)

        try:
            typed_required_params = {
                k: model_class.REQUIRED_PARAMS[k](v)
                for k, v in required_params.items()
            }
        except ValueError as e:
            raise ValueError(f"Error converting parameters for model '{model_name}': {e}")

        return model_class(**typed_required_params, **optional_params).to(self.device)

    def create_unique_folder(self):
        """
        Creates a unique folder for saving model weights and logs based on the current training parameters.

        Returns:
            str: The path to the created directory.
        """
        filename = f"{self.model_params.get('model_type', 'UnknownModel')}__" \
            f"{self.training_time}"

        save_directory = os.path.join(self.run_dir, filename)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        return save_directory

    def load_segmentation_data(self):
        """
        Loads segmentation data from the specified directories, prepares datasets, and creates data loaders.

        This method also handles the loading of normalization statistics and the splitting of data into training,
        validation, and test sets.
        """

        def load_data_stats(data_dir):
            """
            Loads normalization statistics from a JSON file. Provides default normalization
            stats if the file is missing or improperly formatted.

            Args:
                data_dir (str): Directory containing the data stats JSON file.

            Returns:
                dict: A dictionary containing the loaded data statistics.
            """
            neutral_stats = [np.array([0.5] * 3), np.array([0.5] * 3)]  # Default mean and std
            json_file_path = os.path.join(data_dir, 'data_stats.json')

            if not os.path.exists(json_file_path):
                print(f"File {json_file_path} not found. Using default normalization stats.")
                return {"default": neutral_stats}

            try:
                with open(json_file_path, 'r') as file:
                    raw_data_stats = json.load(file)

                data_stats_loaded = {}
                for key, value in raw_data_stats.items():
                    if not (isinstance(value, list) and len(value) == 2 and
                            all(isinstance(v, list) and len(v) == 3 for v in value)):
                        raise ValueError(f"Invalid format in data_stats.json for key {key}")

                    data_stats_loaded[key] = [
                        np.array(value[0]),
                        np.array(value[1])
                    ]

                return data_stats_loaded

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error loading data stats from {json_file_path}: {e}. Using default normalization stats.")
                return {"default": neutral_stats}


        def save_indices_to_file(indices_list):
            """
            Saves the indices of the training, validation, and test sets to text files.

            Args:
                indices_list (list): A list containing the indices for training, validation, and test sets.
            """
            indices_map = {
                "train": indices_list[0],
                "val": indices_list[1],
                "test": indices_list[2],
            }
            for idx_type, idx_list in indices_map.items():
                file_path = os.path.join(self.save_directory, f'{idx_type}_indices.txt')
                with open(file_path, 'w') as f:
                    for subfolder_name, sample_idx in idx_list:
                        f.write(f"{subfolder_name}, {sample_idx}\n")

        def generate_random_indices(num_samples, val_split, subfolders, num_sample_subfolder):
            """
            Generates random indices for splitting the dataset into training, validation, and test sets.

            Args:
                num_samples (int): Total number of samples in the dataset.
                val_split (float): Proportion of the dataset to use for validation.
                subfolders (list): List of subfolder names containing the dataset.
                num_sample_subfolder (dict): Dictionary mapping subfolder names to the number of samples in each.

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
            val_indices_temp = np.random.choice(len(all_indices), size=num_val, replace=False)
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
        data_stats = load_data_stats(self.data_dir)

        for subfolder in self.subfolders:
            img_folder = os.path.join(self.data_dir, subfolder, "images")
            mask_folder = os.path.join(self.data_dir, subfolder, "masks")
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
            indices=train_indices,
            img_data=img_data,
            mask_data=mask_data,
            num_classes=self.model.num_classes,
            crop_size=(self.crop_size, self.crop_size),
            data_stats=data_stats,
        )
        val_dataset = TiffDatasetLoader(
            indices=val_indices,
            img_data=img_data,
            mask_data=mask_data,
            num_classes=self.model.num_classes,
            crop_size=(self.crop_size, self.crop_size),
            data_stats=data_stats,
        )
        test_dataset = TiffDatasetLoader(
            indices=test_indices,
            img_data=img_data,
            mask_data=mask_data,
            num_classes=self.model.num_classes,
            crop_size=(self.crop_size, self.crop_size),
            data_stats=data_stats,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)

        save_indices_to_file([train_indices, val_indices, test_indices])

        self.dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'indices': indices,
        }

    def save_data_stats(self, data_stats):
        """
        Saves the data statistics to a JSON file.

        Args:
            data_stats (dict): A dictionary containing the data statistics to save.
        """
        # Ensure default stats are not saved unless they are the only stats
        data_stats_serializable = {
            key: [value[0].tolist(), value[1].tolist()]
            for key, value in data_stats.items()
            if key != "default" or len(data_stats) == 1
        }

        json_file_path = os.path.join(self.save_directory, 'data_stats.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(data_stats_serializable, json_file, indent=4)

        print(f"Data stats saved to {json_file_path}")

    def save_hyperparameters(self):
        """
        Saves the hyperparameters to an INI file in the save directory.
        """
        config = configparser.ConfigParser()

        config.add_section('Model')
        for key, value in self.model_params.items():
            config.set('Model', key, str(value))
        
        config.add_section('Optimizer')
        for key, value in self.optimizer_params.items():
            config.set('Optimizer', key, str(value))

        config.add_section('Scheduler')
        for key, value in self.scheduler_params.items():
            config.set('Scheduler', key, str(value))

        config.add_section('Training')
        for key, value in self.training_params.items():
            config.set('Training', key, str(value))

        config.add_section('Data')
        for key, value in self.data.items():
            config.set('Data', key, str(value))

        with open(os.path.join(self.save_directory, 'hyperparameters.ini'), 'w') as configfile:
            config.write(configfile)

    def training_loop(self, optimizer, scheduler):
        """
        Executes the training loop for the model, including training and validation phases.

        Args
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use.
        """

        def plot_learning_curves(_param, metrics_dict_param):
            """
            Plots the learning curves for loss and metrics over epochs.

            Args:
                _param (dict): A dictionary containing loss values for training and validation.
                metrics_dict_param (dict): A dictionary containing metric values for training and validation.
            """
            epochs = list(_param['train'].keys())
            train_loss_values = [_param['train'][epoch_train] for epoch_train in epochs]
            val_loss_values = [_param['val'][epoch_val] for epoch_val in epochs]

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            ax0 = axes[0]
            ax0.plot(epochs, train_loss_values, 'b-', label='Train Loss')
            ax0.plot(epochs, val_loss_values, 'r-', label='Val Loss')
            ax0.set_title('Loss')
            ax0.set_xlabel('Epochs')
            ax0.set_ylabel('Loss Value')
            ax0.legend()

            ax1 = axes[1]
            for metric in metrics_dict_param['train']:
                train_metric_values = [metrics_dict_param['train'][metric][epoch_train - 1] for epoch_train in epochs]
                val_metric_values = [metrics_dict_param['val'][metric][epoch_val - 1] for epoch_val in epochs]
                ax1.plot(epochs, train_metric_values, label=f'Train {metric}')
                ax1.plot(epochs, val_metric_values, label=f'Val {metric}')

            ax1.set_title('Metrics')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Metric Values')
            ax1.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_directory, 'learning_curves.png'), dpi=300)

        scaler = None
        if self.device == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True

        metrics = self.initialize_metrics()
        loss_fn = self.initialize_losses()
        print(loss_fn)

        loss_dict = {"train": {}, "val": {}}
        metrics_dict = {phase: {metric: [] for metric in self.metrics} for phase in ["train", "val"]}

        best_val_loss = float("inf")
        best_val_metrics = {metric: 0 for metric in self.metrics}

        for epoch in range(1, self.epochs + 1):
            print(f"\n{'-' * 18}\n--- Epoch {epoch}/{self.epochs} ---")

            for phase in ["train", "val"]:
                is_training = phase == "train"
                self.model.train() if is_training else self.model.eval()

                running_loss = 0.0
                running_metrics = {metric: 0.0 for metric in self.metrics}
                total_samples = 0

                with tqdm(total=len(self.dataloaders[phase]), unit="batch") as pbar:
                    for inputs, labels, _, _ in self.dataloaders[phase]:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(is_training):
                            with torch.autocast(device_type=self.device, dtype=torch.float16):
                                outputs = self.model(inputs).squeeze()
                                loss = loss_fn(outputs, labels.squeeze())

                            if is_training:
                                if self.device == "cuda":
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    loss.backward()
                                    optimizer.step()

                                if self.scheduler_params.get('scheduler') == "ReduceLROnPlateau":
                                    scheduler.step(loss)
                                else:
                                    scheduler.step()

                        running_loss += loss.item()
                        total_samples += labels.size(0)

                        with torch.no_grad():
                            preds = (outputs > 0.5).int() if self.model.num_classes <= 2 \
                                else torch.argmax(outputs, dim=1).int()
                            labels = labels.int()
                            for metric_name, metric_fn in zip(self.metrics, metrics):
                                running_metrics[metric_name] += metric_fn(preds, labels).item()
                        pbar.set_postfix(
                            loss=running_loss / (pbar.n + 1),
                            **{metric: running_metrics[metric] / (pbar.n + 1) for metric in self.metrics}
                        )
                        pbar.update(1)

                epoch_loss = running_loss / len(self.dataloaders[phase])
                epoch_metrics = {metric: running_metrics[metric] / len(self.dataloaders[phase]) for metric in self.metrics}

                loss_dict[phase][epoch] = epoch_loss
                for metric, value in epoch_metrics.items():
                    metrics_dict[phase][metric].append(value)

                print(f"{phase.title()} Loss: {epoch_loss: .4f}")
                for metric, value in epoch_metrics.items():
                    print(f"{phase.title()} {metric}: {value: .4f}", end=" | ")
                print()

                if phase == "val" and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_directory, "model_best_loss.pth"))

                if phase == "val":
                    for metric, value in epoch_metrics.items():
                        if value > best_val_metrics[metric]:
                            best_val_metrics[metric] = value
                            torch.save(self.model.state_dict(),
                                       os.path.join(self.save_directory, f"model_best_{metric}.pth"))

        print(f"Best Validation Metrics: {best_val_metrics}")
        plot_learning_curves(loss_dict, metrics_dict)
        self.save_hyperparameters()
        self.save_data_stats(self.dataloaders['train'].dataset.data_stats)

    def train(self):
            optimizer = self.initialize_optimizer()
            scheduler = self.initialize_scheduler(optimizer)
            self.training_loop(optimizer, scheduler)
