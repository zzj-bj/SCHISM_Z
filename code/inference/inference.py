
# -*- coding: utf-8 -*-
"""
Inference module for performing predictions on large images using a trained model.

This module handles the initialization of the model, loading of datasets,
and performing patch-based predictions to reconstruct full-size output images.
"""

# Standard library
import sys
import os
import glob
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-party libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as nn_func
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from patchify import unpatchify

# Local application imports
from AI.tiffdatasetloader import TiffDatasetLoader, TiffDatasetLoaderConfig
from tools.paramconverter import ParamConverter
from AI.model_registry import model_mapping, model_config_mapping
from tools import utils as ut
from tools import display_color as dc
from tools.constants import DISPLAY_COLORS as colors
from tools.constants import NUM_WORKERS
import tools.constants as ct

# Ensure project root is on the PYTHONPATH
# Z: Add the project root to sys.path so local packages can be imported
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
    def __repr__(self) -> str:
        """
        Returns a string representation of the Inference class.

        Returns:
            str: A string indicating the class name.
        """
        return 'Inference'

    def __init__(self, **kwargs: Any) -> None:
        # Z: ParamConverter to sparse INI/string parameters
        self.param_converter = ParamConverter()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_dir = kwargs.get('data_dir')
        if data_dir is None:
            raise ValueError("The 'data_dir' argument must be provided and not None.")
        self.data_dir = Path(data_dir)
        run_dir = kwargs.get('run_dir')
        if run_dir is None:
            raise ValueError("The 'run_dir' argument must be provided and not None.")
        self.run_dir = Path(run_dir)
        self.hyperparameters = kwargs.get('hyperparameters')
        self.metric = kwargs.get('selected_metric')
        self.subfolders = kwargs.get('subfolders')
        self.display = dc.DisplayColor()

        # Extract category-wise parameters
        if self.hyperparameters is None:
            raise ValueError("The 'hyperparameters' argument must be provided and not None.")
        self.model_params = self.hyperparameters.get_parameters()['Model']
        self.data_params = self.hyperparameters.get_parameters()['Data']
        self.train_params = self.hyperparameters.get_parameters()['Training']

        # Initialize dataset parameters
        self.img_res = int(self.data_params.get('img_res', 560))
        self.crop_size = int(self.data_params.get('crop_size', 224))
        self.num_classes = int(self.model_params.get('num_classes', 1))
        # Z: may cause issues when saving parameters to JSON causes here num_classes is changed non manually
        self.num_classes = 1 if self.num_classes <= 2 else self.num_classes
        self.data_stats = ut.load_data_stats(self.run_dir, self.data_dir)
        self.model_mapping = model_mapping
        self.model_config_mapping = model_config_mapping

    def initialize_model(self) -> torch.nn.Module:
        """
        Initializes the model based on the specified model type and loads the pre-trained weights.

        Returns:
            nn.Module: The initialized model ready for inference.

        Raises:
            ValueError: If the specified model type is not supported
            or if there is an error converting parameters.
        """
        model_name = self.model_params.get('model_type', 'UnetVanilla')
        if model_name not in self.model_mapping:
            msg = f"Model '{model_name}' is not supported. Check your 'model_mapping'."
            self.display.print(msg, colors['error'])
            raise ValueError(msg)

        model_class = self.model_mapping[model_name]
        model_config_class = self.model_config_mapping[model_name]

        self.model_params['num_classes'] = self.num_classes

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
        optional_params.pop('num_classes', None)
        # Z: parameters' second type conversion and verification
        try:
            typed_required_params = {
                k: model_class.REQUIRED_PARAMS[k](v) for k, v in required_params.items()
            }
        except ValueError as e:
            msg = f"Error converting parameters for model '{model_name}': {e}"
            self.display.print(msg, colors['error'])
            raise ValueError(msg) from e
        # Z: Try to initialize model on GPU, fallback to CPU on CUDA error
        try:
            model = model_class(
                model_config_class(**typed_required_params, **optional_params)
            ).to(self.device)
        except RuntimeError as e:
            self.display.print(
                f"CUDA error loading model: {e}. Falling back to CPU.",
                colors['warning']
            )
            self.device = 'cpu'
            model = model_class(
                model_config_class(**typed_required_params, **optional_params)
            ).to(self.device)

        checkpoint_path = self.run_dir / f"model_best_{self.metric}.pth"
        if not checkpoint_path.exists():
            msg = f"Checkpoint not found at '{checkpoint_path}'. Ensure the path is correct."
            self.display.print(msg, colors['error'])
            raise FileNotFoundError(msg)
        # Z: weights_only=True to load only model weights
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
        # Z: strict=False to allow lack or excess of keys in state_dict
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        return model

    def load_dataset(self) -> torch.utils.data.DataLoader:
        """
        Initializes the dataset loader in inference mode to process all images.

        Returns:
            DataLoader: A DataLoader object for iterating over the dataset.
        """
        # Prepare dataset structure
        # Z: img_data is a dict with subfolder names as keys and list of image paths as values
        img_data = {}
        # Z: indices is a list of tuples (subfolder_name, image_index)
        indices = []
        # Z: listdir() returns folder and file names in the data_dir not deeper
        for subfolder in os.listdir(self.data_dir):
            img_folder = os.path.join(
                str(self.data_dir),
                str(subfolder),
                "images")
            # Z: skip if not a directory / folder
            if not os.path.isdir(img_folder):
                continue
            # Z: gather all image paths in the subfolder's images directory
            img_data[subfolder] = sorted(
                glob.glob(os.path.join(img_folder, "*.*"))
            )
            for i in range(len(img_data[subfolder])):
                indices.append((subfolder, i))

            preds = f"preds_{self.metric}"
            preds_folder = os.path.join(
                str(self.data_dir),
                str(subfolder),
                str(preds))
            os.makedirs(preds_folder, exist_ok=True)

        dataset = TiffDatasetLoader(
            TiffDatasetLoaderConfig(
                img_data=img_data,
                indices=indices,
                data_stats=self.data_stats,
                num_classes=self.num_classes,
                img_res=self.img_res,
                crop_size=(self.crop_size, self.crop_size),
                inference_mode=True,
            )
        )

        try:
            pin_mem = torch.cuda.is_available()
        except Exception:
            pin_mem = False

        return DataLoader(dataset,
                          batch_size=1,
                          num_workers=NUM_WORKERS,
                          shuffle=False,
                          pin_memory=pin_mem)

    def predict(self) -> None:
        """
        Performs patch-based predictions on the dataset and saves the results.

        This method processes large images in smaller patches, predicts using the model,
        and stitches the patches back together to form the full-size output image.
        """

        try:
            self.model = self.initialize_model()
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            self.display.print(str(e), colors["error"])
            return

        dataloader = self.load_dataset()

        with tqdm(total=len(dataloader),
                bar_format="Inference : {n_fmt}/{total_fmt} |{bar}| {percentage:6.2f}%",
                unit="Images",
                position=0,
                leave=True,
                ncols=ct.TQDM_NCOLS,
                dynamic_ncols=False,
          ) as pbar:
            # Z: for batch_index, (img, maybe GT or mask, img_path) in enumerate(dataloader):
            for _, (img, _, img_path) in enumerate(dataloader):

                with torch.no_grad():
                    # Perform patch-based prediction
                    # Z: and reconstruct the full image prediction
                    full_pred = self._patch_based_prediction(img)

                # Save the reconstructed prediction
                # Z: get original file name and extension
                name_c = os.path.basename(img_path[0])
                # Z: separate file name and extension
                base_name, ext = os.path.splitext(name_c)
                # Z: create new file name with metric
                new_name = f"{base_name}_{self.metric}{ext}"
                # Z: get subfolder name from image path
                subfolder = os.path.basename(os.path.dirname(os.path.dirname(img_path[0])))

                preds = f"preds_{self.metric}"
                pred_save_path = os.path.join(
                    str(self.data_dir),
                    str(subfolder),
                    str(preds),
                    str(f"{new_name}"))
                self._save_mask(full_pred, pred_save_path)

                # Mettez à jour la barre de progression
                pbar.update(1)

    def _patch_based_prediction(self, patches: List[torch.Tensor]) -> torch.Tensor:
        """
        Reassembles patches into a full-size prediction map.
        Z: forward patches through the model and reconstructs the full image prediction.
        
        Args:
            patches (list of torch.Tensor): List of image patches.
            original_dims (tuple): Original dimensions of the full image (height, width).

        Returns:
            torch.Tensor: Final prediction map (class indices) of the full image.
        """

        # Calculate the grid size (number of patches per dimension)
        num_patches = len(patches)
        grid_size = int(num_patches ** 0.5)  # Assuming square grid of patches (e.g., 5x5)
        dimensions = grid_size * self.crop_size

        # Initialize an empty tensor to store the final predictions (no overlap handling needed)
        full_pred = torch.zeros((
            self.num_classes,
            dimensions, dimensions),
            device=self.device
        )
        # Z: Current treated patch index
        patch_index = 0
        # Z: Store prediction for each patch
        predicted_patches = []
        # Z: Double loop over grid to process patches in order: for row in grid, for col in grid
        for _ in range(grid_size):
            for _ in range(grid_size):
                patch = patches[patch_index]
                with torch.no_grad():
                    # Perform inference on the patch
                    # (model expects 4D input: [batch_size, channels, height, width])
                    # Z: Output logits generally have the shape [batch_size, num_classes, H, W]
                    # Z: For each image in the batch, there are num_classes channels,
                    # Z: and each channel is an H×W score map for the whole image
                    patch_pred = self.model.forward(patch.to(self.device))

                    if self.num_classes > 1:
                        # Multiclass: Apply softmax to get probabilities
                        # and then get the class with the highest probability for each pixel
                        # Z: Apply argmax over the channel dimension for each pixel,
                        # Z: producing a [batch, H, W] tensor of type uint8,
                        # Z: where each value is the predicted class index for that pixel (0 to num_classes−1)
                        patch_pred = torch.argmax(patch_pred, dim=1).to(torch.uint8)
                    else:
                        # Binary: Apply sigmoid to get probabilities
                        # and then threshold to get binary classification
                        patch_pred = torch.sigmoid(patch_pred).squeeze(0)
                        # Z: if >0.5 set to 1 else 0 for each pixel
                        patch_pred = (patch_pred > 0.5).to(torch.uint8)


                patch_pred = self._scale_mask_to_class_values(patch_pred)
                # Z: Resize patch prediction to crop_size if needed and squeeze batch dimension
                patch_pred_resized = nn_func.interpolate(patch_pred.unsqueeze(0),
                                                            size=(self.crop_size,
                                                                  self.crop_size),
                                                            mode='nearest-exact').squeeze(0)

                predicted_patches.append(patch_pred_resized.cpu())
                patch_index += 1

        # Resize the full prediction map to the original image size
        predicted_patches_reshaped = np.reshape(predicted_patches,
                                       (grid_size,
                                        grid_size,
                                        self.crop_size,
                                        self.crop_size)
                                    )
        reconstructed_image = unpatchify(predicted_patches_reshaped, (dimensions, dimensions))
        full_pred = torch.tensor(reconstructed_image).float()

        return full_pred

    def _scale_mask_to_class_values(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """
        Scales the predicted mask tensor to the appropriate class values.
        Z: predicted class indices = 0,1,...,num_classes-1, convert to class values in [0,255]

        Args:
            mask_tensor (torch.Tensor): The predicted mask tensor containing class indices.

        Returns:
            torch.Tensor: A tensor containing the scaled mask with class values.
        """
        if self.num_classes > 1:
            class_values = torch.linspace(0, 255,
                                        self.num_classes,
                                        device=mask_tensor.device).round()
            scaled_mask = class_values[mask_tensor.long()]  # Map class indices to class values
            return scaled_mask
        return mask_tensor * 255  # Converts 0 to 0 and 1 to 255

    def _save_mask(self, mask_tensor: torch.Tensor, save_path: str) -> None:
        """
        Saves a segmentation mask to a file.

        Args:
            mask_tensor (torch.Tensor): The mask tensor to be saved.
            save_path (str): The file path where the mask will be saved.
        """
        # Z: uint8 from 0 to 255
        mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask_array)
        mask_image.save(save_path)
