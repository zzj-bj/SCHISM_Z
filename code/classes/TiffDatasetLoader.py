import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms.functional as func_torch
from torchvision.datasets import VisionDataset
import torch.nn.functional as F
import json

class TiffDatasetLoader(VisionDataset):
    
    def __init__(self,  img_data=None, mask_data=None, indices=None, data_stats=None, num_classes=None, img_res=560, crop_size=(224, 224), p=0.5, inference_mode=False):
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
        """
        dataset_id, sample_id = self.indices[0] 
        img_path = self.img_data[dataset_id][sample_id]
        with Image.open(img_path) as img:
            return img.size[::-1] 
    
    def get_random_crop_params(self):
        h, w = self.image_dims  
        th, tw = self.crop_size
           
        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
           
        if w == tw and h == th:
            return 0, 0, h, w
           
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
           
        return i, j, th, tw
    
    def __getitem__(self, idx):
        dataset_id, sample_id = self.indices[idx]
        img_path = self.img_data[dataset_id][sample_id]
        
        if self.inference_mode: 
            img_path = self.img_data[dataset_id][sample_id]
            img = np.array(Image.open(img_path).convert("RGB"))
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255.0
            img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(self.img_res, self.img_res), 
                                        mode="bicubic", align_corners=False).squeeze()
            m, s = self.data_stats.get(dataset_id, self.data_stats["default"])
            img_normalized = torchvision.transforms.functional.normalize(img_resized, mean=m, std=s).float()
            return img_normalized, dataset_id, img_path
       
        mask_path = self.mask_data[dataset_id][sample_id]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
    
        assert img.shape[:2] == mask.shape, (
            f"Mismatch in dimensions: Image {img.shape} vs Mask {mask.shape} for {img_path}"
        )
    
        # Apply cropping and resizing as before
        i, j, h, w = self.get_random_crop_params()
        img = img[i:i+h, j:j+w, :].copy()
        mask = mask[i:i+h, j:j+w].copy()
    
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255.0
        mask_tensor = torch.from_numpy(mask).contiguous() / 255.0
    
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(self.img_res, self.img_res), 
                                    mode="bicubic", align_corners=False).squeeze()
        mask_resized = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(self.img_res, self.img_res), 
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
        return len(self.indices)
  