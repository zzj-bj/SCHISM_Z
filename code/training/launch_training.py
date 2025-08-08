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
from AI.hyperparameters import Hyperparameters
from training.training import Training
from tools.constants import IMAGE_EXTENSIONS

class LaunchTraining:
    def __init__(self) -> None:
        self.display = DisplayColor()

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

        try:
            hyperparameters = Hyperparameters(str(hyper_file))
        except Exception as e:
            self.display.print(f"Cannot load hyperparameters {e}", colors["error"])
            return

        valid_subfolders: List[str] = []

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
                    f"Count mismatch in {name}- nb images: {len(img_files)} / nb masks: {len(msk_files)}",
                    colors["error"]
                )
                return

            ok, miss_img, miss_msk = self.compare_number(images_dir, masks_dir)
            if not ok:
                self.display.print(f"Missing masks for images in {name}: {miss_img}", colors["error"])
                self.display.print(f"Missing images for masks in {name}: {miss_msk}", colors["error"])
                return

            valid_subfolders.append(name)

        if not valid_subfolders:
            self.display.print("No valid subfolders to train on", colors["error"])
            return

        self.display.print("Starting training", colors["warning"])
        trainer = Training(
            data_dir=str(data_dir),
            subfolders=valid_subfolders,
            run_dir=str(run_dir),
            hyperparameters=hyperparameters,
        )
        try:
            trainer.load_segmentation_data()
            trainer.train()

        except Exception as e:
            self.display.print(f"Training failed {e}", colors["error"])
            return

        self.display.print("Training completed", colors["ok"])
