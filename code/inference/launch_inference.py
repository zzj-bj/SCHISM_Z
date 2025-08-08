# Standard library
from pathlib import Path
from typing import Any, Dict, List

# Local application imports
from AI.hyperparameters import Hyperparameters
from tools import menu, utils as ut
from tools.display_color import DisplayColor
from tools.constants import DISPLAY_COLORS as colors
from inference.inference import Inference

class LaunchInference:
    """
    Responsible for launching the inference process.
    Validates inputs, selects metric, then runs prediction.
    """

    def __init__(self) -> None:
        self.display = DisplayColor()

    def run_inference(self) -> None:
        ut.print_box("Inference")

        # gather paths
        data_dir = Path(ut.get_path_color("Enter directory containing data to predict"))
        run_dir  = Path(ut.get_path_color("Enter directory containing model weights"))
        hyper_path = run_dir / "hyperparameters.ini"

        # collect subfolders and weight files
        subfolders = [d.name for d in data_dir.iterdir() if d.is_dir()] if data_dir.is_dir() else []
        weight_files = list(run_dir.glob("*.pth")) if run_dir.is_dir() else []

        # validate everything at once
        errors = []
        if not data_dir.is_dir():
            errors.append("Data directory is missing or invalid")
        if not subfolders:
            errors.append("No subfolders found in data directory")
        if not run_dir.is_dir():
            errors.append("Run directory is missing or invalid")
        if not hyper_path.exists():
            errors.append("hyperparameters.ini not found in run directory")
        if not weight_files:
            errors.append("No .pth weight files found in run directory")

        if errors:
            for e in errors:
                self.display.print(e, colors["error"])
            return

        # load hyperparameters and extract metrics
        hyper = Hyperparameters(str(hyper_path))
        params = hyper.get_parameters().get("Training", {})
        metrics = [
            m.strip()
            for m in params.get("metrics", "").split(",")
            if m.strip() and m.strip() != "ConfusionMatrix"
        ]

        if not metrics:
            self.display.print("No valid metrics defined in hyperparameters", colors["error"])
            return

        # choose metric
        if len(metrics) > 1:
            menu_items = ["Metrics"] + metrics
            mm = menu.Menu("Dynamic", menu_items)
            mm.display_menu()
            choice = mm.selection()
            selected_metric = metrics[choice - 1]
        else:
            selected_metric = metrics[0]
        
        expected_ckpt = run_dir / f"model_best_{selected_metric}.pth"
        if not expected_ckpt.exists():
            self.display.print(
                f"Checkpoint for metric {selected_metric} not found at {expected_ckpt}",
                colors["error"]
            )
            return
        
        self.display.print(f"Starting inference using {selected_metric}", colors["warning"])

        # run inference
        inf = Inference(
            data_dir=str(data_dir),
            subfolders=subfolders,
            run_dir=str(run_dir),
            selected_metric=selected_metric,
            hyperparameters=hyper,
        )
        inf.predict()
        #TODO
        self.display.print(f"Inference completed using {selected_metric}", colors["ok"])
