import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as func_torch
from torchvision.datasets import VisionDataset
import torch.nn.functional as nn_func
import os
from patchify import patchify
import torchvision.transforms as T


class TiffDatasetLoader(VisionDataset):

    def __init__(self, img_data=None, mask_data=None, indices=None, data_stats=None,
                 num_classes=None, img_res=560, crop_size=(224, 224), p=0.5, inference_mode=False):
        """
        Initializes the TiffDatasetLoader with image and mask data.

        Args:
            img_data (dict): A dictionary containing image file paths.
            mask_data (dict): A dictionary containing mask file paths.
            indices (list): A list of tuples indicating the dataset and sample indices.
            data_stats (dict): A dictionary containing normalization statistics.
            num_classes (int): The number of classes in the dataset.
            img_res (int): The resolution to which images will be resized.
            crop_size (tuple): The size of the crop to be applied to images.
            p (float): Probability of applying random transformations (flips).
            inference_mode (bool): Flag indicating if the dataset is in inference mode.
        """
        super().__init__(transforms=None)
        self.data_stats = data_stats
        self.img_data = img_data
        self.mask_data = mask_data
        self.indices = indices
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.img_res = img_res
        self.inference_mode = inference_mode
        self.p = p
        self.image_dims = self.get_image_dimensions()

    def get_image_dimensions(self):
        """
        Reads the dimensions of the first image from the dataset dynamically.
        Assumes all images have the same dimensions.

        Returns:
            tuple: A tuple containing the height and width of the image.
        """
        dataset_id, sample_id = self.indices[0]
        img_path = self.img_data[dataset_id][sample_id]
        with Image.open(img_path) as img:
            return img.size[::-1]

    def get_random_crop_params(self):
        """
        Generates random cropping parameters for the images.

        Returns:
            tuple: A tuple containing the starting row index, starting column index, height, and width of the crop.

        Raises:
            ValueError: If the required crop size is larger than the input image size.
        """
        h, w = self.image_dims
        th, tw = self.crop_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def extract_patches(self, img_tensor, dataset_id):
        height, width = self.image_dims
        patch_h, patch_w = self.crop_size  # Typically 224x224

        # Convert the tensor to numpy for patchify
        img_np = img_tensor.numpy()

        # Check if padding is necessary (non-square or not a multiple of crop size)
        if height != width or height % patch_h != 0 or width % patch_w != 0:
            # Calculate the padding needed to reach the nearest multiple of crop size (224)
            pad_height = (patch_h - height % patch_h) % patch_h
            pad_width = (patch_w - width % patch_w) % patch_w
            padding = [(0, 0), (0, pad_height), (0, pad_width)]  # Pad along height and width

            # Pad the image
            img_np = np.pad(img_np, padding, mode='constant', constant_values=0)

        patches = patchify(img_np, (3, patch_h, patch_w), step=patch_h)

        m, s = self.data_stats.get(dataset_id, self.data_stats["default"])
        processed_patches = []

        #save_dir = "C:\\Users\\florent.brondolo\\OneDrive - Akkodis\\Documents\\SCHISM\\data\\test_pred\\sample2\\test"
        patch_index = 0

        for i in range(patches.shape[1]):  # Loop through patch grid
            for j in range(patches.shape[2]):
                patch = patches[0, i, j]

                patch_tensor = torch.tensor(patch).float()  

                # Resize patch using nn.functional.interpolate
                patch_resized = nn_func.interpolate(patch_tensor.unsqueeze(0), size=(self.img_res, self.img_res),
                                                    mode="bicubic", align_corners=False).squeeze(0)  # Remove the batch dimension

                # Normalize the tensor
                patch_normalized = torchvision.transforms.functional.normalize(patch_resized, mean=m, std=s).float()

                # Save the patch for visualization/debugging
                #patch_filename = os.path.join(save_dir, f"patch_{patch_index}.png")
                #T.functional.to_pil_image(patch_resized).save(patch_filename)
                patch_index += 1

                processed_patches.append(patch_normalized)

        return processed_patches

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the normalized image tensor, mask tensor (if not inference mode),
                   dataset ID, and image path.

        Raises:
            AssertionError: If the dimensions of the image and mask do not match.
        """
        dataset_id, sample_id = self.indices[idx]
        img_path = self.img_data[dataset_id][sample_id]

        if self.inference_mode:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255.0
            patches = self.extract_patches(img_tensor, dataset_id)
            return patches, dataset_id, img_path

        mask_path = self.mask_data[dataset_id][sample_id]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        assert img.shape[:2] == mask.shape, (
            f"Mismatch in dimensions: Image {img.shape} vs Mask {mask.shape} for {img_path}"
        )

        # Apply cropping and resizing as before
        i, j, h, w = self.get_random_crop_params()
        img = img[i:i + h, j:j + w, :].copy()
        mask = mask[i:i + h, j:j + w].copy()

        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255.0
        mask_tensor = torch.from_numpy(mask).contiguous() / 255.0

        img_resized = nn_func.interpolate(img_tensor.unsqueeze(0), size=(self.img_res, self.img_res),
                                          mode="bicubic", align_corners=False).squeeze()
        mask_resized = nn_func.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(self.img_res, self.img_res),
                                           mode="nearest").squeeze()

        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.hflip(img_resized)
            mask_resized = torchvision.transforms.functional.hflip(mask_resized)

        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.vflip(img_resized)
            mask_resized = torchvision.transforms.functional.vflip(mask_resized)

        m, s = self.data_stats.get(dataset_id, self.data_stats["default"])
        img_normalized = torchvision.transforms.functional.normalize(img_resized, mean=m, std=s).float()

        if self.num_classes >= 2:
            mask_resized = (mask_resized * self.num_classes).long() - 1

        return img_normalized, mask_resized, dataset_id, img_path

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.indices)
