import sys
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from classes.TiffDatasetLoader import TiffDatasetLoader
from classes.UnetVanilla import UnetVanilla
import torch.nn.functional as nn_func
from patchify import unpatchify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Inference:

    def __repr__(self):
        """
        Returns a string representation of the Inference class.

        Returns:
            str: A string indicating the class name.
        """
        return 'Inference'

    def __init__(self, **kwargs):
        """
        Initializes the Inference class with the necessary parameters and model setup.

        Args:
            **kwargs: Keyword arguments containing configuration parameters such as:
                - data_dir (str): Directory containing the input data.
                - run_dir (str): Directory for saving results and loading model weights.
                - hyperparameters (Hyperparameters): An object containing model and training parameters.
                - selected_metric (str): Metric used for model evaluation.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_dir = kwargs.get('data_dir')
        self.run_dir = kwargs.get('run_dir')
        self.hyperparameters = kwargs.get('hyperparameters')
        self.metric = kwargs.get('selected_metric')

        # Extract category-wise parameters
        self.model_params = self.hyperparameters.get_parameters()['Model']
        self.data_params = self.hyperparameters.get_parameters()['Data']
        self.train_params = self.hyperparameters.get_parameters()['Training']

        # Initialize dataset parameters
        self.img_res = int(self.data_params.get('img_res', 560))
        self.crop_size = int(self.data_params.get('crop_size', 224))
        self.num_classes = int(self.model_params.get('num_classes', 1))
        self.data_stats = self.load_data_stats_from_json()
        print(self.data_stats)

        # Initialize model
        self.model_mapping = {
            'UnetVanilla': UnetVanilla,
        }

        self.model = self.initialize_model()

    def initialize_model(self) -> nn.Module:
        """
        Initializes the model based on the specified model type and loads the pre-trained weights.

        Returns:
            nn.Module: The initialized model ready for inference.

        Raises:
            ValueError: If the specified model type is not supported or if there is an error converting parameters.
        """
        model_name = self.model_params.get('model_type')
        if model_name not in self.model_mapping:
            raise ValueError(f"Model '{model_name}' is not supported. Check your 'model_mapping'.")
        model_class = self.model_mapping[model_name]

        # Extract the required and optional parameters for the model
        required_params = {k: v for k, v in self.model_params.items() if k in model_class.REQUIRED_PARAMS}
        optional_params = {k: v for k, v in self.model_params.items() if k not in model_class.REQUIRED_PARAMS}

        # Ensure `model_type` is not included in the parameters
        required_params.pop('model_type', None)  # Remove 'model_type' if present in required_params
        optional_params.pop('model_type', None)  # Remove 'model_type' if present in optional_params

        try:
            # Convert the required parameters to the correct types for the model class
            typed_required_params = {
                k: model_class.REQUIRED_PARAMS[k](v)  # Ensure the parameters match the expected types
                for k, v in required_params.items()
            }
        except ValueError as e:
            raise ValueError(f"Error converting parameters for model '{model_name}': {e}")
        model = model_class(**typed_required_params, **optional_params).to(self.device)

        # Load model weights
        checkpoint_path = os.path.join(self.run_dir, f"model_best_{self.metric}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
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
        dataset_counter = 0

        for subfolder in os.listdir(self.data_dir):
            img_folder = os.path.join(self.data_dir, subfolder, "images")

            if not os.path.isdir(img_folder):
                continue

            img_data[dataset_counter] = sorted(
                glob.glob(os.path.join(img_folder, "*.*"))
            )
            for i in range(len(img_data[dataset_counter])):
                indices.append((dataset_counter, i))

            # Create the preds folder for saving predicted masks
            preds_folder = os.path.join(self.data_dir, subfolder, "preds")
            os.makedirs(preds_folder, exist_ok=True)

            dataset_counter += 1
        dataset = TiffDatasetLoader(
            img_data=img_data,
            indices=indices,
            data_stats=self.data_stats,
            num_classes=self.num_classes,
            img_res=self.img_res,
            crop_size=(self.crop_size, self.crop_size),
            inference_mode=True
        )
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def load_data_stats_from_json(self):
        """
        Loads normalization statistics from a JSON file and populates self.data_stats.

        Returns:
            dict: A dictionary containing the loaded data statistics.

        Raises:
            Exception: If there is an error loading the JSON file.
        """
        json_file_path = os.path.join(self.run_dir, 'data_stats.json')
        try:
            # Read the JSON file
            with open(json_file_path, 'r') as file:
                raw_data_stats = json.load(file)

            # Convert the JSON content to the desired format
            self.data_stats = {
                key: [np.array(values[0]), np.array(values[1])]
                for key, values in raw_data_stats.items()
            }

            print("Data stats loaded successfully.")
            return self.data_stats  # Return for verification if needed

        except Exception as e:
            print(f"Error loading data stats: {e}")
            raise

    def predict(self):
        """
        Performs patch-based predictions on the dataset and saves the results.

        This method processes large images in smaller patches, predicts using the model,
        and stitches the patches back together to form the full-size output image.
        """
        dataloader = self.load_dataset()
        for i, (img, dataset_id, img_path) in enumerate(tqdm(dataloader, desc="Predicting")):

            with torch.no_grad():
                # Perform patch-based prediction
                full_pred = self._patch_based_prediction(img)

            # Save the reconstructed prediction
            base_name = os.path.basename(img_path[0])
            subfolder = os.path.basename(os.path.dirname(os.path.dirname(img_path[0])))
            pred_save_path = os.path.join(self.data_dir, subfolder, "preds", f"pred_{base_name}")
            self._save_mask(full_pred, pred_save_path)

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
        dimenssions = grid_size * self.crop_size

        # Initialize an empty tensor to store the final predictions (no overlap handling needed)
        full_pred = torch.zeros((self.num_classes, dimenssions, dimenssions), device=self.device)

        # Iterate through patches to perform predictions
        # Loop through rows of patches
        patch_index = 0
        predicted_patches = []

        for i in range(grid_size):  
            for j in range(grid_size):  
                patch = patches[patch_index]
                
                # Perform inference on the patch (model expects 4D input: [batch_size, channels, height, width])
                patch_pred = self.model(patch)  # Batch dimension is already handled
                
                if self.num_classes > 1:
                    # Multiclass: get the class with the highest probability for each pixel
                    patch_pred = torch.argmax(patch_pred, dim=1).squeeze(0)  # [C, H, W] -> [H, W]
                else:
                    # Binary: apply sigmoid and threshold
                    patch_pred = torch.sigmoid(patch_pred).squeeze(0)  # [B, 1, H, W] -> [H, W]
                    patch_pred = (patch_pred > 0.5).float()  # Convert to binary mask [H, W]

                patch_pred_resized = nn_func.interpolate(patch_pred.unsqueeze(0), 
                                                         size=(self.crop_size, self.crop_size), 
                                                         mode='bicubic', 
                                                         align_corners=False).squeeze(0)

                # Add the resized patch prediction to the list of predicted patches
                predicted_patches.append(patch_pred_resized)
                patch_index += 1

        # Resize the full prediction map to the original image size
        predicted_patches_reshaped = np.reshape(predicted_patches, (grid_size, grid_size, self.crop_size, self.crop_size))
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
        class_values = torch.linspace(0, 255, self.num_classes, device=mask_tensor.device).round()
        scaled_mask = class_values[mask_tensor.long()]  # Map class indices to class values
        return scaled_mask

    def _save_mask(self, mask_tensor, save_path):
        """
        Saves a segmentation mask to a file.

        Args:
            mask_tensor (torch.Tensor): The mask tensor to be saved.
            save_path (str): The file path where the mask will be saved.
        """
        mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
        # mask_image = Image.fromarray(mask_array)
        mask_image = Image.fromarray(mask_array)
        mask_image.save(save_path)
