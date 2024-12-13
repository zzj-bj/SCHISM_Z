import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import glob
from datetime import datetime
from tqdm import tqdm
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from PIL import Image
from classes.TiffDatasetLoader import TiffDatasetLoader
from classes.UNet_vanilla import UNet_vanilla
import json

class Inference:

    def __repr__(self):
        return 'Inference'

    def __init__(self, **kwargs):
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
        self.num_classes = int(self.model_params.get('num_classes', 1))
        self.data_stats = self.load_data_stats_from_json()
        print(self.data_stats)
        
        # Initialize model
        self.model_mapping = {
            'UNet_vanilla': UNet_vanilla,
        }
        
        self.model = self.initialize_model()

    def initialize_model(self) -> nn.Module:
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
        model= model_class(**typed_required_params, **optional_params).to(self.device)    

        # Load model weights
        checkpoint_path = os.path.join(self.run_dir, f"model_best_{self.metric}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def load_dataset(self):
        """
        Initializes the dataset loader in inference mode to process all images.
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
            inference_mode=True
        )
        return DataLoader(dataset, batch_size=1, shuffle=False)
    
    
    def load_data_stats_from_json(self):
        """
        Loads normalization statistics from a JSON file and populates self.data_stats.
        
        Args:
            json_file_path (str): Path to the JSON file.
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
        Performs predictions on the dataset and saves the results.
        """
        dataloader = self.load_dataset()

        for i, (img, dataset_id, img_path) in enumerate(tqdm(dataloader, desc="Predicting")):
            img = img.to(self.device)

            with torch.no_grad():
                pred = self.model(img)
                pred = torch.argmax(pred, dim=1).squeeze(0)

            # Convert the predicted mask to the appropriate class values
            scaled_mask = self._scale_mask_to_class_values(pred)

            # Save the prediction
            base_name = os.path.basename(img_path[0])
            subfolder = os.path.basename(os.path.dirname(os.path.dirname(img_path[0])))
            pred_save_path = os.path.join(self.data_dir, subfolder, "preds", f"pred_{base_name}")
            self._save_mask(scaled_mask, pred_save_path)

    def _scale_mask_to_class_values(self, mask_tensor):
        """
        Scales the predicted mask tensor to the appropriate class values.
        """
        class_values = torch.linspace(0, 255, self.num_classes, device=mask_tensor.device).round()
        scaled_mask = class_values[mask_tensor.long()]  # Map class indices to class values
        return scaled_mask

    def _save_mask(self, mask_tensor, save_path):
        """
        Saves a segmentation mask to a file.
        """
        mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask_array)
        mask_image.save(save_path)
