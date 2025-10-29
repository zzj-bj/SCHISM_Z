# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""

# Standard library
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Local application imports
from tools import utils as ut
from tools.display_color import DisplayColor
from tools.constants import DISPLAY_COLORS as colors
from tools.constants import IMAGE_EXTENSIONS
from tools.hyperparameters import Hyperparameters
from training.training import Training
from tools.paramconverter import ParamConverter

class LaunchTraining:
    def __init__(self) -> None:
        self.display = DisplayColor()
        self.param_converter = ParamConverter()
        self.data: Dict[str, Any] = {}     

    def compare_number(
        self,
        dir1: str | Path,
        dir2: str | Path
    ) -> Tuple[bool, List[str], List[str]]:
        pattern = re.compile(r"(\d+)")
        nums1 = {
            pattern.findall(f.name)[-1]
            for f in Path(dir1).iterdir()
            if (m := pattern.findall(f.name))
        }
        nums2 = {
            pattern.findall(f.name)[-1]
            for f in Path(dir2).iterdir()
            if (m := pattern.findall(f.name))
        }
        if nums1 == nums2:
            return True, [], []
        return False, sorted(nums1 - nums2), sorted(nums2 - nums1)

    def check_data_integrity(self, total_images: int, hyperp: Hyperparameters) -> bool:
        """Check consistency between hyperparameters and available data."""

        # Initialize integrity flag 
        check_data_integrity = True

        # Store hyperparameters reference
        self.hyperp = hyperp
        params = hyperp.get_parameters()

        # Safely extract relevant parameter groups
        self.model_params     = params.get('Model', {})
        self.optimizer_params = params.get('Optimizer', {})
        self.scheduler_params = params.get('Scheduler', {})
        self.loss_params      = params.get('Loss', {})
        
        self.training_params  = params.get('Training', {})
        self.data             = params.get('Data', {})

        # Convert and initialize values with defaults if missing
        self.batch_size = self.param_converter._convert_param(self.training_params.get('batch_size', 8))
        self.val_split = self.param_converter._convert_param(self.training_params.get('val_split', 0.8))
        self.num_samples = self.param_converter._convert_param(self.data.get('num_samples', 500))

        # --- Integrity check #1: optimizer ---
        if self.model_params.get('model_type', '') == "":
            self.display.print(
                "Model not specified in hyperparameters. Check your configuration.",
                colors["error"]
            )
            check_data_integrity = False
            return check_data_integrity

        # --- Integrity check #2: optimizer ---
        if self.optimizer_params.get('optimizer', '') == "":
            self.display.print(
                "Optimizer not specified in hyperparameters. Check your configuration.",
                colors["error"]
            )
            check_data_integrity = False
            return check_data_integrity

        # --- Integrity check #3: scheduler ---
        if self.scheduler_params.get('scheduler', '') == "":
            self.display.print(
                "Scheduler not specified in hyperparameters. Check your configuration.",
                colors["error"]
            )
            check_data_integrity = False
            return check_data_integrity

        # --- Integrity check #4: loss function ---
        if self.loss_params.get('loss', '') == "":
            self.display.print(
                "Loss function not specified in hyperparameters. Check your configuration.",
                colors["error"]
            )
            check_data_integrity = False
            return check_data_integrity


        # --- Integrity check #5: num_samples vs total_images ---
        if self.num_samples >= total_images:
            # Adjust num_samples if it exceeds available data
            new_num_samples = int(total_images) - 1

            self.display.print(
                f"num_samples ({self.num_samples}) exceeds total images ({total_images}). "
                f"Setting num_samples to {new_num_samples}.",
                colors["warning"]
            )
            self.display.print(
                "Consider using 'total_images - 1' as a better choice for num_samples.",
                colors["warning"]
            )

            # Update internal values and hyperparameters
            self.num_samples = new_num_samples
            self.data['num_samples'] = new_num_samples
            params.setdefault('Data', {})['num_samples'] = new_num_samples

        # --- Integrity check #5: batch_size vs val_split and num_samples ---
        val_min_size = int(self.batch_size /(1 - self.val_split))
        if self.num_samples < val_min_size:

            self.display.print(
                f"'The 'num_samples' value ({self.num_samples}) is too small."
                f" The minimum required value must be greater than:\n"
                f"  'batch_size' / (1 - 'val_split')",
                colors["warning"]
                )
            self.display.print(
                "Consider using this value as a better choice for num_samples.",
                colors["warning"]
            )
            
            # Update internal values and hyperparameters
            self.num_samples = val_min_size
            self.data['num_samples'] = val_min_size
            params.setdefault('Data', {})['num_samples'] = val_min_size

        return check_data_integrity


    def train_model(self) -> None:
        ut.print_box("Training")

        data_dir = Path(ut.get_path_color("Enter directory with training data"))
        run_dir = Path(ut.get_path_color("Enter directory for training outputs"))
        hyper_dir = Path(ut.get_path_color("Enter directory with hyperparameters"))
        hyper_file = hyper_dir / "hyperparameters.ini"

        if not data_dir.is_dir():
            self.display.print(f"Invalid data directory {data_dir}", colors["error"])
            return
        if not run_dir.is_dir():
            self.display.print(f"Invalid run directory {run_dir}", colors["error"])
            return
        if not hyper_file.is_file():
            self.display.print(f"Missing hyperparameters.ini at {hyper_file}", colors["error"])
            return

        # Check the Hyperparameters file  
        try:
            hyperparameters = Hyperparameters(str(hyper_file))
        except Exception:
            ut.format_and_display_error('Hyperparameters')
            return
       
        # Validate each subfolder in the data directory
        valid_subfolders: List[str] = []
        total_images = 0
        for sub in data_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name
            images_dir = sub / "images"
            if not images_dir.is_dir():
                self.display.print(f"{name}/images not found", colors["error"])
                return

            masks_dir = sub / "masks"
            if not masks_dir.is_dir():
                self.display.print(f"{name}/masks not found", colors["error"])
                return

            img_files = []
            for ext in IMAGE_EXTENSIONS:
                img_files.extend(images_dir.glob(f"*{ext}"))
            img_files = sorted(img_files)

            if not img_files:
                self.display.print(f"No tif files in {name}/images", colors["error"])
                return
            msk_files = list(masks_dir.glob("*.tif"))
            if not msk_files:
                self.display.print(f"No tif files in {name}/masks", colors["error"])
                return

            if len(img_files) != len(msk_files):
                self.display.print(
                    f"Count mismatch in {name}- nb images: {len(img_files)} / nb masks:"
                    f" {len(msk_files)}",
                    colors["error"]
                )
                return

            ok, miss_img, miss_msk = self.compare_number(images_dir, masks_dir)
            if not ok:
                self.display.print(f"Missing masks for images in {name}: {miss_img}",
                                    colors["error"])
                self.display.print(f"Missing images for masks in {name}: {miss_msk}",
                                   colors["error"])
                return

            total_images += len(img_files)                              
            valid_subfolders.append(name)

        # Check data integrity before proceeding
        if not self.check_data_integrity(total_images, hyperparameters):
            return

        # Ensure there are valid subfolders to train on
        if not valid_subfolders:
            self.display.print("No valid subfolders to train on", colors["error"])
            return


        self.display.print("Starting training", colors["warning"])

        try:
            trainer = Training(
                data_dir=str(data_dir),
                subfolders=valid_subfolders,
                run_dir=str(run_dir),
                hyperparameters=hyperparameters,
            )
        except Exception :
            ut.format_and_display_error('Training Loader')
            return

        try:
            trainer.load_segmentation_data()
            trainer.train()
        except Exception :
            ut.format_and_display_error('Training')
            return

        self.display.print("Training completed", colors["ok"])
