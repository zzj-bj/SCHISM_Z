"""
Inference module for performing predictions on large images using a trained model.

This module handles the initialization of the model, loading of datasets,
and performing patch-based predictions to reconstruct full-size output images.
"""
import sys
import os
import glob
import json
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as nn_func
from torch.utils.data import DataLoader

from patchify import unpatchify

from AI.tiffdatasetloader import TiffDatasetLoader
from AI.tiffdatasetloader import TiffDatasetLoaderConfig
from AI.paramconverter import ParamConverter
from AI.model_registry import model_mapping
from AI.model_registry import model_config_mapping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#==============================================================================
class Inference:
    """ 
    A class for performing inference using a pre-trained model.

    This class handles the initialization of the model, loading of datasets,
    and performing patch-based predictions to reconstruct full-size output images.

    It supports various model types and configurations, allowing for flexible inference
    on large images by processing them in smaller patches.

    """
    def __repr__(self):
        """
        Returns a string representation of the Inference class.

        Returns:
            str: A string indicating the class name.
        """
        return 'Inference'

    def add_to_report(self, text, who):
        """
            Add a message to a report
        """
        if self.config["report"] is not None:
            self.config["report"].add(text, who)

    def __init__(self, **kwargs):
        """
        Initializes the Inference class with the necessary parameters and model setup.

        Args:
            **kwargs: Keyword arguments containing configuration parameters such as:
                - data_dir (str): Directory containing the input data.
                - run_dir (str): Directory for saving results and loading model weights.
                - hyperparameters (Hyperparameters): An object containing model
                                    and training parameters.
                - selected_metric (str): Metric used for model evaluation.
        """

        self.config = {
            'param_converter': ParamConverter(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'data_dir': kwargs.get('data_dir', ''),
            'run_dir': kwargs.get('run_dir', ''),
            'hyperparameters': kwargs.get('hyperparameters', None),
            'metric': kwargs.get('selected_metric', 'Jaccard'),
            "report":kwargs.get('report'),
            "model_params": {},
            "data_params": {},
            "train_params": {},
            "data_stats": {},
            "img_res": 560,
            "crop_size": 224,
            "num_classes": 1,
            "model_mapping": model_mapping,
            "model_config_mapping": model_config_mapping,
            "model": None,
        }

        if self.config["param_converter"] is None:
            text = "The 'hyperparameters' argument must be provided and not None."
            self.add_to_report(' - Inference', text)
            raise ValueError(text)

        # Extract category-wise parameters
        self.config["model_params"] = self.config["hyperparameters"].get_parameters()['Model']
        self.config["data_params"] = self.config["hyperparameters"].get_parameters()['Data']
        self.config["train_params"] = self.config["hyperparameters"].get_parameters()['Training']

        # Initialize dataset parameters
        self.config["img_res"] = int(self.config["data_params"].get('img_res', 560))
        self.config["crop_size"] = int(self.config["data_params"].get('crop_size', 224))
        self.config["num_classes"] = int(self.config["model_params"].get('num_classes', 1))
        self.config["num_classes"] = (
            1 if self.config["num_classes"] <= 2 else self.config["num_classes"]
        )
        self.config["data_stats"] = self.load_data_stats_from_json()
        self.config["model_mapping"] = model_mapping
        self.config["model_config_mapping"] = model_config_mapping

        # for data in self.config["data_stats"].values():
        #     print(f" Data stats: {data[0]} - {data[1]}")

        self.config["model"] = self.initialize_model()

    def initialize_model(self) -> nn.Module:
        """
        Initializes the model based on the specified model type and loads the pre-trained weights.

        Returns:
            nn.Module: The initialized model ready for inference.

        Raises:
            ValueError: If the specified model type is not supported
            or if there is an error converting parameters.
        """
        model_name = self.config["model_params"].get('model_type', 'UnetVanilla')
        if model_name not in self.config["model_mapping"]:
            text =f" - Model '{model_name}' is not supported"
            self.add_to_report(' - Inference', text)
            raise ValueError(f" Model '{model_name}' is not supported.\n"
                             " Check your 'model_mapping'.")

        model_class = self.config["model_mapping"][model_name]
        model_config_class = self.config["model_config_mapping"][model_name]

        self.config["model_params"]['num_classes'] = self.config["num_classes"]

        required_params = {
            k: self.config["param_converter"].convert_param(v)
            for k, v in self.config["model_params"].items() if k in model_class.REQUIRED_PARAMS
        }
        optional_params = {
            k: self.config["param_converter"].convert_param(v)
            for k, v in self.config["model_params"].items() if k in model_class.OPTIONAL_PARAMS
        }

        # Ensure `model_type` is not included in the parameters
        required_params.pop('model_type', None)
        optional_params.pop('model_type', None)

        # Ensure 'num_classes' is only in the required parameters,
        # remove it from optional if present
        if 'num_classes' in optional_params:
            del optional_params['num_classes']

        try:
            # Convert the required parameters to their correct types as defined by the model class
            typed_required_params = {
                k: model_class.REQUIRED_PARAMS[k](v) for k, v in required_params.items()
            }
        except ValueError as e:
            text =f" - Error converting parameters for model '{model_name}':\n {e}"
            self.add_to_report(' - Inference', text)
            raise ValueError(f" Error converting parameters for model '{model_name}':"
                             "\n {e}") from e

        # Initialize the model
        model = model_class(
            model_config_class(
                **typed_required_params, **optional_params
            )).to(self.config["device"])

        # Load pre-trained weights
        checkpoint_path = os.path.join(
            str(self.config["run_dir"]),
            f"model_best_{self.config['metric']}.pth"
        )
        if not os.path.exists(checkpoint_path):
            text =f" - Checkpoint not found at '{checkpoint_path}'"
            self.add_to_report(' - Inference', text)
            raise FileNotFoundError(f" Checkpoint not found at '{checkpoint_path}'.\n"
                                    " Ensure the path is correct.")

        checkpoint = torch.load(checkpoint_path, map_location=self.config["device"])
        model.load_state_dict(checkpoint)
        model.eval()  # Set the model to evaluation mode

        return model

    def load_dataset(self):
        """
        Initializes the dataset loader in inference mode to process all images.

        Returns:
            DataLoader: A DataLoader object for iterating over the dataset.
        """
        # Prepare dataset structure
        img_data = {}
        indices = []

        for subfolder in os.listdir(self.config["data_dir"]):
            img_folder = os.path.join(
                str(self.config["data_dir"]),
                str(subfolder),
                "images")

            if not os.path.isdir(img_folder):
                continue

            img_data[subfolder] = sorted(
                glob.glob(os.path.join(img_folder, "*.*"))
            )
            for i in range(len(img_data[subfolder])):
                indices.append((subfolder, i))

            preds = f"preds_{self.config['metric']}"
            preds_folder = os.path.join(
                str(self.config["data_dir"]),
                str(subfolder),
                str(preds))
            os.makedirs(preds_folder, exist_ok=True)

        dataset = TiffDatasetLoader(
            TiffDatasetLoaderConfig(
                img_data=img_data,
                indices=indices,
                data_stats=self.config["data_stats"],
                num_classes=self.config["num_classes"],
                img_res=self.config["img_res"],
                crop_size=(self.config["crop_size"], self.config["crop_size"]),
                inference_mode=True,
            )
        )
        return DataLoader(dataset, batch_size=1, shuffle=False,pin_memory=torch.cuda.is_available())

    def load_data_stats_from_json(self):
        """
        Loads normalization statistics from a JSON file and populates self.data_stats.

        Returns:
            dict: A dictionary containing the loaded data statistics.

        Raises:
            Exception: If there is an error loading the JSON file.
        """
        json_file_path = os.path.join(str(self.config["run_dir"]), 'data_stats.json')
        try:
            # Read the JSON file
            with open(json_file_path, 'r', encoding='utf-8') as file:
                raw_data_stats = json.load(file)

            # Convert the JSON content to the desired format
            self.config["data_stats"] = {
                key: [np.array(values[0]), np.array(values[1])]
                for key, values in raw_data_stats.items()
            }

            print(" Data stats loaded successfully.")
            return self.config["data_stats"]  # Return for verification if needed

        except Exception as e:
            print(f" Error loading data stats: {e}")
            text =f" - Error loading data stats: {e}"
            self.add_to_report(' - Inference', text)
            raise

    def predict(self):
        """
        Performs patch-based predictions on the dataset and saves the results.

        This method processes large images in smaller patches, predicts using the model,
        and stitches the patches back together to form the full-size output image.
        """
        dataloader = self.load_dataset()

        with tqdm(total=len(dataloader), ncols=100,
          bar_format="- Progress: {n_fmt}/{total_fmt} |{bar}| {percentage:6.2f}%",
          ) as pbar:

            for _, (img, _, img_path) in enumerate(dataloader):

                with torch.no_grad():
                    # Perform patch-based prediction
                    full_pred = self._patch_based_prediction(img)

                # Save the reconstructed prediction
                name_c = os.path.basename(img_path[0])
                base_name, ext = os.path.splitext(name_c)
                new_name = f"{base_name}_{self.config['metric']}{ext}"

                subfolder = os.path.basename(os.path.dirname(os.path.dirname(img_path[0])))

                preds = f"preds_{self.config['metric']}"
                pred_save_path = os.path.join(
                    str(self.config["data_dir"]),
                    str(subfolder),
                    str(preds),
                    str(f"{new_name}"))
                self._save_mask(full_pred, pred_save_path)

                # Mettez à jour la barre de progression
                pbar.update(1)

    def _patch_based_prediction(self, patches):
        """
        Reassembles patches into a full-size prediction map.

        Args:
            patches (list of torch.Tensor): List of image patches.
            original_dims (tuple): Original dimensions of the full image (height, width).

        Returns:
            torch.Tensor: Final prediction map (class indices) of the full image.
        """

        # Calculate the grid size (number of patches per dimension)
        num_patches = len(patches)
        grid_size = int(num_patches ** 0.5)  # Assuming square grid of patches (e.g., 5x5)
        dimenssions = grid_size * self.config["crop_size"]

        # Initialize an empty tensor to store the final predictions (no overlap handling needed)
        full_pred = torch.zeros((
            self.config["num_classes"],
            dimenssions, dimenssions),
            device=self.config["device"]
        )

        patch_index = 0
        predicted_patches = []

        for _ in range(grid_size):
            for _ in range(grid_size):
                patch = patches[patch_index]
                with torch.no_grad():
                    # Perform inference on the patch
                    # (model expects 4D input: [batch_size, channels, height, width])
                    patch_pred = self.config["model"].forward(patch.to(self.config["device"]))

                    if self.config["num_classes"] > 1:
                        # Multiclass: Apply softmax to get probabilities
                        # and then get the class with the highest probability for each pixel
                        patch_pred = torch.argmax(patch_pred, dim=1).to(torch.uint8)
                    else:
                        # Binary: Apply sigmoid to get probabilities
                        # and then threshold to get binary classification
                        patch_pred = torch.sigmoid(patch_pred).squeeze(0)
                        patch_pred = (patch_pred > 0.5).to(torch.uint8)


                patch_pred = self._scale_mask_to_class_values(patch_pred)
                patch_pred_resized = nn_func.interpolate(patch_pred.unsqueeze(0),
                                                            size=(self.config["crop_size"],
                                                                  self.config["crop_size"]),
                                                            mode='nearest-exact').squeeze(0)

                predicted_patches.append(patch_pred_resized.cpu())
                patch_index += 1

        # Resize the full prediction map to the original image size
        predicted_patches_reshaped = np.reshape(predicted_patches,
                                       (grid_size,
                                        grid_size,
                                        self.config["crop_size"],
                                        self.config["crop_size"])
                                    )
        reconstructed_image = unpatchify(predicted_patches_reshaped, (dimenssions, dimenssions))
        full_pred = torch.tensor(reconstructed_image).float()

        return full_pred

    def _scale_mask_to_class_values(self, mask_tensor):
        """
        Scales the predicted mask tensor to the appropriate class values.

        Args:
            mask_tensor (torch.Tensor): The predicted mask tensor containing class indices.

        Returns:
            torch.Tensor: A tensor containing the scaled mask with class values.
        """
        if self.config["num_classes"] > 1:
            class_values = torch.linspace(0, 255,
                                        self.config["num_classes"],
                                        device=mask_tensor.device).round()
            scaled_mask = class_values[mask_tensor.long()]  # Map class indices to class values
            return scaled_mask
        return mask_tensor * 255  # Converts 0 to 0 and 1 to 255

    def _save_mask(self, mask_tensor, save_path):
        """
        Saves a segmentation mask to a file.

        Args:
            mask_tensor (torch.Tensor): The mask tensor to be saved.
            save_path (str): The file path where the mask will be saved.
        """
        mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask_array)
        mask_image.save(save_path)
