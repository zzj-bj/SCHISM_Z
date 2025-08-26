# pylint: disable=too-many-instance-attributes
"""
TiffDatasetLoader: A PyTorch Dataset for loading TIFF images and their corresponding masks.
This class handles loading, preprocessing, and patch extraction from TIFF images and masks.
"""
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as nn_func
import torchvision
from torchvision.datasets import VisionDataset
from patchify import patchify
from tools import constants

@dataclass
class TiffDatasetLoaderConfig:
    """
    Configuration class for TiffDatasetLoader.

    This class holds the parameters required to initialize the TiffDatasetLoader.

    Attributes:
        img_data (Dict): Dictionary containing image file paths.
        mask_data (Dict): Dictionary containing mask file paths.
        indices (List[Tuple[int, int]]): List of tuples indicating dataset and sample indices.
        data_stats (Dict): Dictionary containing normalization statistics.
        num_classes (int): Number of classes in the dataset.
        img_res (int): Resolution to which images will be resized.
        crop_size (Tuple[int, int]): Size of the crop to be applied to images.
        p (float): Probability of applying random transformations (flips).
        inference_mode (bool): Flag indicating if the dataset is in inference mode.
        ignore_background (bool): Flag to ignore the background class in masks.
    """
    img_data: Optional[Dict] = None
    mask_data: Optional[Dict] = None
    indices: Optional[List] = None
    data_stats: Optional[Dict] = None
    num_classes: Optional[int] = None
    img_res: int = 560
    crop_size: Tuple[int, int] = (224, 224)
    p: float = 0.5
    inference_mode: bool = False
    ignore_background: bool = True


class TiffDatasetLoader(VisionDataset):
    """    
    TiffDatasetLoader: A PyTorch Dataset for loading TIFF images and their corresponding masks.
    This class handles loading, preprocessing, and patch extraction from TIFF images and masks.
    """
    def __init__(
            self,
            tiff_dataset_loader_config: TiffDatasetLoaderConfig
            ):
        """
        Initializes the TiffDatasetLoader with image and mask data.
        Args:
            tiff_dataset_loader_config (TiffDatasetLoaderConfig): Configuration object containing
                all necessary parameters for the dataset loader.
        """
        super().__init__(transforms=None)
        self.data_stats = tiff_dataset_loader_config.data_stats
        self.img_data = tiff_dataset_loader_config.img_data
        self.mask_data = tiff_dataset_loader_config.mask_data
        self.indices = tiff_dataset_loader_config.indices
        self.num_classes = tiff_dataset_loader_config.num_classes
        self.crop_size = tiff_dataset_loader_config.crop_size
        self.img_res = tiff_dataset_loader_config.img_res
        self.inference_mode = tiff_dataset_loader_config.inference_mode
        self.p = tiff_dataset_loader_config.p
        self.ignore_background = tiff_dataset_loader_config.ignore_background

        self.image_dims = self.get_image_dimensions()
        if not self.inference_mode:
            self.class_values = self._compute_class_values()

    def get_image_dimensions(self):
        """
        Reads the dimensions of the first image from the dataset dynamically.
        Assumes all images have the same dimensions.

        Returns:
            tuple: A tuple containing the height and width of the image.
        """
        if self.indices is None or len(self.indices) == 0:
            raise ValueError(
                "self.indices is None or empty. Cannot determine image dimensions.")

        dataset_id, sample_id = self.indices[0]

        if self.img_data is None:
            raise ValueError(
                "self.img_data is None. Cannot determine image dimensions.")

        img_path = self.img_data[dataset_id][sample_id]
        with Image.open(img_path) as img:
            return img.size[::-1]

    def get_random_crop_params(self):
        """
        Generates random cropping parameters for the images.

        Returns:
            tuple: A tuple containing the starting row index, starting column index, height
            and width of the crop.

        Raises:
            ValueError: If the required crop size is larger than the input image size.
        """
        h, w = self.image_dims
        th, tw = self.crop_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)}"
                             " is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def get_valid_crop(self, img, mask, threshold=0.8, max_attempts=10):
        """
        Attempts to find a crop of the image and mask where the fraction of background pixels
        (with value 0 in the mask) is below the specified threshold. 
        Falls back to center crop if no valid crop is found.
    
        Args:
            img (np.array): The input image (H x W x C).
            mask (np.array): The segmentation mask (H x W), where 0 indicates background.
            threshold (float): Maximum allowed fraction of background pixels (default 0.8).
            max_attempts (int): Maximum number of crop attempts before falling back to center crop.
    
        Returns:
            tuple: Cropped image and mask.
        """
        for _ in range(max_attempts):
            # Get random crop parameters
            i, j, h, w = self.get_random_crop_params()
            # Crop the mask
            crop_mask = mask[i:i+h, j:j+w].copy()
            # Calculate the fraction of background pixels
            background_ratio = (crop_mask == 0).sum() / crop_mask.size
            # Check if the background ratio is below the threshold
            if background_ratio < threshold:
                # Valid crop found, return it
                crop_img = img[i:i+h, j:j+w, :].copy()
                return crop_img, crop_mask

        # If max_attempts reached, perform center crop
        center_i = (self.image_dims[0] - self.crop_size[0]) // 2
        center_j = (self.image_dims[1] - self.crop_size[1]) // 2

        crop_img = img[center_i:center_i+self.crop_size[0],
                       center_j:center_j+self.crop_size[1], :].copy()
        crop_mask = mask[center_i:center_i+self.crop_size[0],
                         center_j:center_j+self.crop_size[1]].copy()

        return crop_img, crop_mask

    def extract_patches(self, img_np):
        """
    Extracts patches from an input image represented as a NumPy array.

    This function takes an image in the format [C, H, W] (channels, height, width)
    and extracts non-overlapping patches of a specified size. If the image
    dimensions are not multiples of the patch size, the function pads the image
    with zeros to ensure that the resulting patches are  correctly sized.

    Parameters:
    - img_np (numpy.ndarray): The input image as a NumPy array with shape [C, H, W],
    where C is the number of channels, H is the height, and W is the width.

    Returns:
    - numpy.ndarray: A NumPy array containing the extracted patches with shape
      [num_patches, C, patch_h, patch_w], where num_patches is the total number
      of patches extracted from the image, and patch_h and patch_w are the height
      and width of each patch.

    Notes:
    - The function first transposes the input image to [H, W, C] for padding operations.
    - If the input image is not square or its dimensions are not multiples of the
    specified crop size, it calculates the necessary padding and applies it.
    - After padding, the image is transposed back to its original format before
    extracting patches.

    Example:
    >>> patches = extract_patches(img_np)
    """
        height, width = self.image_dims
        patch_h, patch_w = self.crop_size

        # Transpose to [H, W, C] for padding
        img_np = np.transpose(img_np, (1, 2, 0))

        # Check if padding is necessary (non-square or not a multiple of crop size)
        if height != width or height % patch_h != 0 or width % patch_w != 0:
            # Calculate the padding needed to reach the nearest multiple of crop size (224)
            pad_height = (patch_h - height % patch_h) % patch_h
            pad_width = (patch_w - width % patch_w) % patch_w

            # Correct padding tuple for a 3D image (H, W, C)
            # Pad along height and width, no padding for channels
            padding = [(0, pad_height), (0, pad_width), (0, 0)]

            # Pad the image
            img_np = np.pad(img_np, padding, mode='constant', constant_values=0)

        # Transpose back to [C, H, W] after padding
        img_np = np.transpose(img_np, (2, 0, 1)).squeeze()
        patches = patchify(img_np, (img_np.shape[0], patch_h, patch_h), step=patch_h)
        patches = patches.reshape(-1, img_np.shape[0], patch_h, patch_w)
        return patches

    def _compute_class_values(self):
        """
        Computes unique class values from all available masks.

        Returns:
            list: Sorted list of unique class values in the dataset.
        """
        unique_values = set()

        if self.mask_data is None:
            raise ValueError(
                    "self.mask_data is None. Cannot retrieve mask path by index.")
        if self.mask_data is None:
            raise ValueError(
                    "self.mask_data is None. Cannot retrieve mask path by index.")
       
        for dataset_id, sample_id in self.indices[:10]:

            if self.mask_data is None:
                raise ValueError(
                    "self.mask_data is None. Cannot retrieve mask path by index.")
            if self.mask_data is None:
                raise ValueError(
                    "self.mask_data is None. Cannot retrieve mask path by index.")

            mask_path = self.mask_data[dataset_id][sample_id]
            mask = np.array(Image.open(mask_path).convert("L"))

            # Normalize the mask to [0, 1] only if binary classification (self.num_classes == 1)
            if self.num_classes == 1:
                unique_vals = np.unique(mask)
                # Scale to [0, 1]
                mask = (mask - unique_vals.min()) / (unique_vals.max() - unique_vals.min())

            # Update the set of unique class values
            unique_values.update(np.unique(mask))

        # Convert the set of unique values to a sorted list
        sorted_values = sorted(unique_values)

        # Handle binary classification (ensure only 0 and 1)
        if self.num_classes == 1:
            sorted_values = [0, 1]  # Binary classification, classes should be 0 and 1
        elif self.ignore_background and len(sorted_values) > 1:
            sorted_values.pop(0)  # Remove the background class (usually class 0)

        return sorted_values

    def _weights_calc(self, mask, temperature=50.0):
        """
        Calculate class weights based on the provided mask.

        This function computes the weights for each class in a classification problem,
        particularly useful in scenarios with class imbalance. The weights are calculated
        as the inverse of the class ratios, normalized using the softmax function.

        Parameters:
        mask (np.ndarray): An array of class labels (integers) for which to calculate weights.
        temperature (float): A parameter that controls the softness of the softmax distribution.
                            Higher values result in a more uniform distribution of weights.

        Returns:
        torch.Tensor: A tensor containing the calculated weights for each class.

        Raises:
        RuntimeError: If NaN values are detected in the weights calculation.
        """
        mask = mask.astype(np.int64)
        counts = np.bincount(mask.ravel(), minlength=self.num_classes)[self.class_values]

        class_ratio = counts / (np.sum(counts) + 1e-8)  # Avoid divide by zero
        u_weights = 1 / (class_ratio + 1e-8)  # Avoid division by zero

        weights = nn_func.softmax(torch.from_numpy(u_weights).float() / temperature, dim=-1)

        if torch.any(torch.isnan(weights)):
            print(weights)
            print(class_ratio)
            print(u_weights)
            raise RuntimeError("NaN detected in weights calculation.")

        return weights

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the normalized image tensor,
            mask tensor (if not inference mode),
              dataset ID, and image path.

        Raises:
            AssertionError: If the dimensions of the image and mask do not match.
        """
        dataset_id, sample_id = self.indices[idx]
        img_path = self.img_data[dataset_id][sample_id]

        if self.inference_mode:
            return self._get_inference_item(dataset_id, img_path)

        return self._get_training_item(dataset_id, sample_id, img_path)

    def _get_inference_item(self, dataset_id, img_path):
        """"
        Retrieves an inference item from the dataset,
        including image processing and patch extraction.
        
        Args:
            dataset_id (int): Identifier for the dataset.
            img_path (str): Path to the image file.
        
        Returns:
            tuple: A tuple containing the processed patches, dataset ID, and image path.
        """
        m, s = self._get_mean_std(dataset_id)
        img = np.array(Image.open(img_path).convert("RGB"))
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1).contiguous() / 255
        img_normalized = torchvision.transforms.functional.normalize(img_tensor, mean=m, std=s)
        patches = self.extract_patches(img_normalized)
        processed_patches = []
        for i in range(patches.shape[0]):
            patch = patches[i]
            patch_tensor = torch.tensor(patch).unsqueeze(0)
            patch_resized = nn_func.interpolate(
                patch_tensor,
                size=(self.img_res, self.img_res),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            processed_patches.append(patch_resized)
        return processed_patches, dataset_id, img_path

    def _get_training_item(self, dataset_id, sample_id, img_path):
        """
        Retrieves a training item from the dataset,
        including image and mask processing.
        
        Args:
            dataset_id (int): Identifier for the dataset.
            sample_id (int): Identifier for the sample within the dataset.
            img_path (str): Path to the image file.
        
        Returns:
            tuple: A tuple containing the normalized image tensor,
            mask tensor (if not inference mode),
            class weights, dataset ID, and image path.
        
        Raises:
            AssertionError: If the dimensions of the image and mask do not match.
        """
        m, s = self._get_mean_std(dataset_id)
        mask_path = self.mask_data[dataset_id][sample_id]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        assert img.shape[:2] == mask.shape, (
            f"Mismatch in dimensions: Image {img.shape} vs Mask {mask.shape} for {img_path}"
        )
        img, mask = self.get_valid_crop(img, mask, threshold=0.8, max_attempts=20)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255

        mask_tensor, weights = self._prepare_mask_and_weights(mask)

        img_resized = nn_func.interpolate(img_tensor.unsqueeze(0),
                                        size=(self.img_res, self.img_res),
                                        mode="bicubic",
                                        align_corners=False).squeeze()
        mask_resized = nn_func.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                        size=(self.img_res, self.img_res),
                                        mode="nearest").squeeze()

        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.hflip(img_resized)
            mask_resized = torchvision.transforms.functional.hflip(mask_resized)
        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.vflip(img_resized)
            mask_resized = torchvision.transforms.functional.vflip(mask_resized)

        img_normalized = torchvision.transforms.functional.normalize(
            img_resized, mean=m, std=s).float()

        if self.num_classes > 1:
            if self.ignore_background:
                mask_resized = (mask_resized * self.num_classes).long() - 1
            else:
                mask_resized = (mask_resized * (self.num_classes - 1)).long()

        return img_normalized, mask_resized, weights

    def _prepare_mask_and_weights(self, mask):
        """
        Prepares the mask tensor and class weights based on the number of classes.
        Args:
            mask (np.array): The segmentation mask.
            Returns:
                tuple: A tuple containing the mask tensor and class weights.
        """
        if self.num_classes > 1:
            mask_tensor = torch.from_numpy(mask).contiguous() / 255
            weights = self._weights_calc(mask)
        else:
            unique_vals = np.unique(mask)
            mask = (mask - unique_vals.min()) / (unique_vals.max() - unique_vals.min())
            mask = mask.astype(np.int64)
            mask_tensor = torch.from_numpy(mask).contiguous()
            weights = torch.zeros(self.num_classes)
        return mask_tensor, weights

    def _get_mean_std(self, dataset_id):
        """
        Retrieves the mean and standard deviation for normalization based on the dataset ID.
        
        Args:
            dataset_id (int): Identifier for the dataset.
        
        Returns:
            tuple: A tuple containing the mean and
            standard deviation for normalization.
        """
        if dataset_id in self.data_stats:
            return self.data_stats[dataset_id]

        if "default" in self.data_stats:
            return self.data_stats["default"]
        
        return constants.DEFAULT_MEAN_STD, constants.DEFAULT_MEAN_STD

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.indices is None:
            raise ValueError(
                "self.indices is None. Cannot determine dataset length.")
        return len(self.indices)
