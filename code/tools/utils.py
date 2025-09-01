# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.
@author: Pierre.FANCELLI
"""

# Standard library
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Third-party
import numpy as np
from colorama import init, Style
from jsonschema import Draft7Validator

# Local application imports
import preprocessing.launch_preprocessing as lp
import tools.constants as ct
from tools.constants import DISPLAY_COLORS as colors
import tools.display_color as dc

# Initialize colorama
init(autoreset=True)

display = dc.DisplayColor()

#==============================================================================
def rgb_to_ansi(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB color to ANSI escape code."""
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"

def chck_color(color_key: str) -> Tuple[int, int, int]:
    """
    Check if the color key exists in the DISPLAY_COLORS dictionary.
    If it does, return the corresponding RGB value.
    If not, return a default color (Light Green).
    """
    try:
        color = colors[color_key]
    except KeyError:
        color = (153, 204, 51)  # Default to Light Green

    # Check if the color is a valid RGB tuple.
    if not (isinstance(color, tuple) and len(color) == 3 and
             all(isinstance(c, int) and 0 <= c <= 255 for c in color)):
        color = (153, 204, 51)

    return color


def get_path(prompt: str) -> str:
    """Requests a valid path from the user."""
    display = dc.DisplayColor()
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        text = f"Invalid path: {path}. Please try again. {ct.BELL}"
        display.print(text, colors['error'])

def get_path_color(prompt: str, color_key: str = 'input') -> str:
    """
    Requests a valid path from the user.
    Displays the prompt in the specified color.
    If the specified color key is invalid, the prompt will be displayed in Light Green.
    """
    display = dc.DisplayColor()

    color = chck_color(color_key)
    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {prompt}: {Style.RESET_ALL}"
        path = input(colored_prompt).strip()
        if os.path.exists(path):
            return path
        text = f"Invalid path: {path}. Please try again."
        display.print(text, colors['error'])

def answer_yes_or_no(message: str, color_key: str = 'input') -> bool:
    """
    This function returns
        - True for yes, y, oui, o.
        - False for no, non n.
        - The input does not take uppercase letters into account."
    Displays the prompt in the specified color.
    If the specified color key is invalid, the prompt will be displayed in Light Green.

    """
    display = dc.DisplayColor()

    color = chck_color(color_key)
    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {message} (y/n) ? : {Style.RESET_ALL}"

        reponse = input(colored_prompt).strip()
        if reponse in ['yes', 'y', 'YES', 'Y', 'Yes']:
            return True
        if reponse in ['no', 'n', 'NO', 'N', 'No']:
            return False
        text = f"Please provide a valid answer (y/n) {ct.BELL}"
        display.print(text, colors['error'])

def brigth_mode(message: str, color_key: str = 'input') -> str :
    """
    This function returns
        - single
        - all

    Displays the prompt in the specified color.
    If the specified color key is invalid, the prompt will be displayed in Light Green.

    """
    display = dc.DisplayColor()

    color = chck_color(color_key)
    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {message} : {Style.RESET_ALL}"

        reponse = input(colored_prompt).strip()
        if reponse in ['s', 'S']:
            return 'single'
        if reponse in ['a', 'A']:
            return "all"
        text = f"Please provide a valid answer (s/a) {ct.BELL}"
        display.print(text, colors['error'])





def format_and_display_error(texte : str) -> None  :
    """
    Handles errors based on the specified level of detail.   
    """

    display = dc.DisplayColor()

    # Retrieve the type, value, and traceback of the most recent exception
    exc_type, exc_value, exc_traceback = sys.exc_info()

    tb = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_type = traceback.format_exception_only(exc_type, exc_value)

    # Display the error message
    if ct.DEBUG_MODE:
        # Display the complete traceback
        prompt =  f"{texte} :\n {''.join(tb)}"
    else:
        # Display the type of exception only
        prompt = f"{texte} :\n {''.join(tb_type)}"

    display.print(prompt, colors['error'])



def input_percentage(message: str, color_key: str = 'input') -> float:
    """
    This function returns a real number between 0 and 1
    that corresponds to a percentage.
    """
    #TODO
    display = dc.DisplayColor()

    color = chck_color(color_key)

    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {message} : {Style.RESET_ALL}"

        try:
            # enter = input(colored_prompt).strip()
            value = float(input(colored_prompt).strip())
            if 1 <= value <= 100:
                return value / 100

            text = f"The value must be between 1 and 100. {ct.BELL}"
            display.print(text, colors['error'])
        except ValueError:
            text = f"Please enter a valid number. {ct.BELL}"
            display.print(text, colors['error'])

def print_box(text: str) -> None:
    """
    Prints a text string inside a decorative box.

    The box is created with a width that is determined by the length of the input text,
    with additional padding on both sides. The box is framed with special characters
    to enhance its appearance.

    Parameters:
    text (str): The text to be displayed inside the box.

    """
    # Determine the width of the box based on the string length
    box_width = len(text) + 2  # Add padding for the box

    # Create the box
    print(f"╔{'═' * (box_width)}╗")
    print(f"║{text.center(box_width)}║")
    print(f"╚{'═' * (box_width)}╝")


def load_data_stats(
    json_dir: Union[str, Path],
    data_dir: Union[str, Path]
) -> Dict[str, List[np.ndarray]]:
    """
    1. Load+validate data_stats.json
    2. Check for missing folders
    3. Optionally regenerate JSON
    4. Return data_stats dict (or default)
    """
    display = dc.DisplayColor()
    neutral_stats = ct.DEFAULT_MEAN_STD
    json_path = Path(json_dir) / "data_stats.json"

    # 1) File exists?
    if not json_path.is_file():
        if answer_yes_or_no("data_stats.json not found; generate a new one?"):
            lp.LaunchPreprocessing().launch_json_generation(
                data_dir=str(data_dir),
                file_name_report=str(json_path),
                append=False,
            )
            return load_data_stats(json_dir, data_dir)
        else:
            display.print("Using default normalization stats.", ct.DISPLAY_COLORS["warning"])
            return {"default": neutral_stats}

    # 2) Read & parse JSON
    try:
        raw_text = json_path.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        display.print(f"JSON parse error: {e}", ct.DISPLAY_COLORS["error"])
        if answer_yes_or_no("Invalid JSON; regenerate data_stats.json?"):
            lp.LaunchPreprocessing().launch_json_generation(
                data_dir=str(data_dir),
                file_name_report=str(json_path),
                append=False,
            )
            return load_data_stats(json_dir, data_dir)
        else:
            display.print("Using default normalization stats.", ct.DISPLAY_COLORS["warning"])
            return {"default": neutral_stats}

    # 3) Schema validation
    validator = Draft7Validator(ct.DATA_STATS_SCHEMA)
    errors = sorted(validator.iter_errors(raw), key=lambda e: e.path)
    if errors:
        for err in errors:
            display.print(f"Schema error: {err.message}", ct.DISPLAY_COLORS["error"])
        if answer_yes_or_no("Schema invalid; regenerate data_stats.json?"):
            lp.LaunchPreprocessing().launch_json_generation(
                data_dir=str(data_dir),
                file_name_report=str(json_path),
                append=True,
            )
            return load_data_stats(json_dir, data_dir)
        else:
            display.print("Using default normalization stats.", ct.DISPLAY_COLORS["warning"])
            return {"default": neutral_stats}

    # 4) Convert to numpy
    try:
        data_stats = {
            key: [np.array(vals[0], float), np.array(vals[1], float)]
            for key, vals in raw.items()
        }
    except Exception as e:
        display.print(f"Error converting stats to arrays: {e}", ct.DISPLAY_COLORS["error"])
        return {"default": neutral_stats}

    # 5) Check for missing subfolders — **use data_dir**, not dir!
    subfolders = [d.name for d in Path(data_dir).iterdir() if d.is_dir()]
    missing = [d for d in subfolders if d not in data_stats]
    if missing:
        display.print(f"Datasets without stats: {missing}", ct.DISPLAY_COLORS["warning"])
        if answer_yes_or_no("Generate updated data_stats.json including them?"):
            lp.LaunchPreprocessing().launch_json_generation(
                data_dir=str(data_dir),
                file_name_report=str(json_path),
                append=True,
            )
            return load_data_stats(json_dir, data_dir)
        else:
            display.print("Using default normalization stats.", ct.DISPLAY_COLORS["warning"])
            return {"default": neutral_stats}

    # 6) Success
    return data_stats
